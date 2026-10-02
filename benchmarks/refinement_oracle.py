"""Phase 1B: precision-refinement decomposition search (oracle only, no runtime).

    uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/<name> [--num-prompts N]

Reuses the prompts, model and numerics of the Phase 1A run named by `source_run`
in configs/phase1b-refinement.yaml. For every prompt, decomposition, mode and
(bound tier, tie break) pair, it simulates progressive refinement of the LM head
and records when the certificate would fire and how many bytes would be read.

Writes, into the output directory:
  config.yaml          the Phase 1B config as given
  environment.json     versions, hardware, numerics flags, source run
  decompositions.json  per decomposition: states, bytes per row, metadata, storage, code digest
  reference.jsonl      per prompt: reference token, top-2 gap, prefix and fallback bitwise checks
  validation.jsonl     per prompt × decomposition: envelope violations per tier, masked-fallback checks
  records.jsonl.gz     per prompt × decomposition × mode × (bound, tie break)
  digest.json          sha256 of the four files above
It stops at the first hard failure (unless --keep-going) and saves it to failure.json:
a prefix or fallback mismatch, an envelope violation, or a certified mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.certificate import TieBreak  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.decomposition import RefinementDecomposition  # noqa: E402
from awpmi.models.smollm2 import ModelSpec, final_hidden_state, lm_head_weight, load_model, resolve_dtype  # noqa: E402
from awpmi.oracle.refinement import IO_BLOCK_BYTES, BoundTier, Mode, RefinementBatch, simulate  # noqa: E402
from awpmi.reference import ReferenceRunner  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl  # noqa: E402

OUTPUT_FILES = ("decompositions.json", "reference.jsonl", "validation.jsonl", "records.jsonl.gz")


class HardFailure(Exception):
    pass


def tensor_digest(*tensors: torch.Tensor | None) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_pass(model, runner, weight, prompts, device, failures, keep_going, log):
    """Reference logits and LM-head inputs; the fallback form F.linear(h, W) must reproduce the reference bitwise."""
    hidden_rows, logits_rows, tokens = [], [], []
    for prompt in prompts:
        input_ids = torch.tensor([prompt["token_ids"]], device=device)
        reference = runner.next_token(input_ids)
        hidden = final_hidden_state(model, input_ids)
        # The oracle stores inputs as contiguous rows; validate exactly that form.
        row = hidden.reshape(-1).contiguous()
        fallback = F.linear(row.view(1, 1, -1), weight).reshape(-1)
        top2 = torch.topk(reference.logits.to(torch.float32), 2).values.tolist()
        entry = {
            "prompt_id": prompt["prompt_id"],
            "reference_token": reference.token_id,
            "reference_top2_gap": top2[0] - top2[1],
            "prefix_bitwise_equal": bool(torch.equal(hidden, reference.hidden_state)),
            "fallback_bitwise_equal": bool(torch.equal(fallback, reference.logits)),
            "fallback_token": int(torch.argmax(fallback)),
        }
        log.write(entry)
        if not (entry["prefix_bitwise_equal"] and entry["fallback_bitwise_equal"]) or (
            entry["fallback_token"] != reference.token_id
        ):
            failures.append({"kind": "reference", **entry})
            if not keep_going:
                raise HardFailure
        hidden_rows.append(row)
        logits_rows.append(reference.logits)
        tokens.append(reference.token_id)
    return torch.stack(hidden_rows), torch.stack(logits_rows), tokens


class MaskedFallbackChecker:
    """Diagnostic: does F.linear(h, W with only the final contenders kept) reproduce those rows bitwise?

    If it does, a fallback would not need the rows already eliminated. Not part of
    any certificate or of the primary byte accounting.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self._cache: dict[tuple[int, str], bool] = {}

    def check(self, prompt_id: int, hidden: torch.Tensor, alive: torch.Tensor, reference_logits, token) -> bool:
        key = (prompt_id, hashlib.sha256(alive.cpu().numpy().tobytes()).hexdigest())
        if key not in self._cache:
            masked = torch.where(alive[:, None], self.weight, torch.zeros((), dtype=self.weight.dtype, device=alive.device))
            logits = F.linear(hidden.view(1, 1, -1), masked).reshape(-1)
            rows_equal = torch.equal(logits[alive], reference_logits[alive])
            choice = int(torch.argmax(torch.where(alive, logits, torch.full_like(logits, -torch.inf))))
            self._cache[key] = rows_equal and choice == token
        return self._cache[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase1b-refinement.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts of the source run")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source_run = REPO_ROOT / raw_config["source_run"]
    source_config, _ = load_config(source_run / "config.yaml")
    specs = list(raw_config["decompositions"])
    modes = [Mode(name) for name in raw_config["modes"]]
    configurations = [(BoundTier(tier), TieBreak(tie)) for tier, tie in raw_config["configurations"]]
    batch_size = int(raw_config["batch_size"])

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(source_config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(
        source_config.model.repository,
        source_config.model.revision,
        resolve_dtype(source_config.model.dtype, device),
        device,
    )
    model, _ = load_model(spec)
    runner = ReferenceRunner(model)
    weight = lm_head_weight(model)
    numerics = ReferenceNumerics(weight.dtype, source_config.numerics.accumulation_unit_roundoff, weight.shape[1])
    prompts = read_jsonl(source_run / "prompts.jsonl")
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    model_info = {
        "repository": spec.repository,
        "revision": spec.revision,
        "dtype": str(spec.dtype),
        "device": str(device),
        "reference_numerics": {
            "output_dtype": str(numerics.output_dtype),
            "accumulation_unit_roundoff": numerics.accumulation_unit_roundoff,
            "reduction_length": numerics.reduction_length,
        },
    }
    environment = environment_metadata(REPO_ROOT, model_info, NUMERICS_FLAGS)
    environment["source_run"] = raw_config["source_run"]
    environment["num_prompts"] = len(prompts)
    environment["io_block_bytes"] = IO_BLOCK_BYTES
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    failures: list[dict] = []
    started = time.perf_counter()
    decompositions: dict[str, dict] = {}
    try:
        with JsonlWriter(output / "reference.jsonl") as reference_log:
            hidden, reference_logits, reference_tokens = reference_pass(
                model, runner, weight, prompts, device, failures, args.keep_going, reference_log
            )
        print(f"reference pass: {len(prompts)} prompts, {time.perf_counter() - started:.0f}s", flush=True)
        prompt_ids = [prompt["prompt_id"] for prompt in prompts]
        checker = MaskedFallbackChecker(weight)

        with (
            JsonlWriter(output / "validation.jsonl") as validation_log,
            JsonlWriter(output / "records.jsonl.gz") as record_log,
        ):
            for name in specs:
                decomposition = RefinementDecomposition.build(weight, name)
                states = [decomposition.state_name(s) for s in range(decomposition.num_states)]
                row_bytes = [sum(decomposition.row_bytes(s)) for s in range(decomposition.num_states)]
                decompositions[name] = {
                    **decomposition.describe(),
                    "reconstruction_exact": True,  # build() raises otherwise
                    "levels_sha256": tensor_digest(
                        *[t for level in decomposition.levels for t in (level.codes, level.scales)],
                        *[t for norms in decomposition.remainder_norms for t in (norms.l2, norms.linf)],
                    ),
                }
                for start in range(0, len(prompts), batch_size):
                    columns = slice(start, start + batch_size)
                    batch = RefinementBatch(decomposition, hidden[columns], numerics)
                    reference_columns = reference_logits[columns].T
                    intervals = {tier: batch.intervals(tier) for tier in dict.fromkeys(t for t, _ in configurations)}
                    violations = {tier.value: iv.violations(reference_columns).tolist() for tier, iv in intervals.items()}
                    masked = {offset: [0, 0] for offset in range(reference_columns.shape[1])}
                    for mode in modes:
                        for tier, tie_break in configurations:
                            simulation = simulate(intervals[tier], mode, tie_break, row_bytes)
                            certified = simulation.certified.tolist()
                            decision = simulation.decision_state.tolist()
                            winners = simulation.winner.tolist()
                            competitors = simulation.competitor.tolist()
                            margins = simulation.margin.tolist()
                            contender_counts = simulation.contenders.T.tolist()
                            rows_loaded = simulation.rows_loaded.T.tolist()
                            io_blocks = simulation.io_blocks.T.tolist()
                            for offset in range(len(certified)):
                                index = start + offset
                                fallback = not certified[offset]
                                fallback_rows = (
                                    decomposition.out_features - rows_loaded[offset][-1] if fallback else 0
                                )
                                breakdown = decomposition.materialized_bytes(rows_loaded[offset], fallback_rows)
                                masked_exact = None
                                if fallback and mode is Mode.ROW_SELECTIVE:
                                    masked_exact = checker.check(
                                        prompt_ids[index],
                                        hidden[index],
                                        simulation.final_contenders[:, offset],
                                        reference_logits[index],
                                        reference_tokens[index],
                                    )
                                    masked[offset][0] += 1
                                    masked[offset][1] += not masked_exact
                                io_bytes = (
                                    breakdown["metadata"]
                                    + IO_BLOCK_BYTES * sum(io_blocks[offset])
                                    + (decomposition.weight_bytes if fallback_rows else 0)
                                )
                                token = winners[offset] if certified[offset] else reference_tokens[index]
                                record = {
                                    "prompt_id": prompt_ids[index],
                                    "decomposition": name,
                                    "mode": mode.value,
                                    "bound": tier.value,
                                    "tie_break": tie_break.value,
                                    "reference_token": reference_tokens[index],
                                    "token": token,
                                    "certified": certified[offset],
                                    "fallback": fallback,
                                    "decision_state": states[decision[offset]] if certified[offset] else None,
                                    "decision_state_index": decision[offset] if certified[offset] else None,
                                    "winner": winners[offset],
                                    "competitor": competitors[offset],
                                    "certificate_margin": margins[offset],
                                    "contenders": contender_counts[offset],
                                    "rows_loaded": rows_loaded[offset],
                                    "bytes": breakdown,
                                    "fraction": breakdown["total"] / decomposition.weight_bytes,
                                    "fraction_masked_fallback": (breakdown["total"] - breakdown["fallback_rows"])
                                    / decomposition.weight_bytes,
                                    "masked_fallback_exact": masked_exact,
                                    "io_bytes_4k": io_bytes,
                                }
                                record_log.write(record)
                                if token != reference_tokens[index]:
                                    failures.append({"kind": "certified_mismatch", **record})
                                    if not args.keep_going:
                                        raise HardFailure
                    for offset in range(reference_columns.shape[1]):
                        entry = {
                            "prompt_id": prompt_ids[start + offset],
                            "decomposition": name,
                            "envelope_violations": {tier: counts[offset] for tier, counts in violations.items()},
                            "masked_fallback_checks": masked[offset][0],
                            "masked_fallback_mismatches": masked[offset][1],
                        }
                        validation_log.write(entry)
                        if any(entry["envelope_violations"].values()):
                            failures.append({"kind": "envelope_violation", **entry})
                            if not args.keep_going:
                                raise HardFailure
                    del batch, intervals
                print(f"{name}: done, {time.perf_counter() - started:.0f}s", flush=True)
                del decomposition
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except HardFailure:
        pass

    (output / "decompositions.json").write_text(json.dumps(decompositions, indent=2), encoding="utf-8")
    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    digest = {
        "decompositions_sha256": canonical_digest([decompositions]),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl")),
        "validation_sha256": canonical_digest(read_jsonl(output / "validation.jsonl"))
        if (output / "validation.jsonl").exists()
        else None,
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl.gz"))
        if (output / "records.jsonl.gz").exists()
        else None,
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - started:.0f}s; hard failures: {len(failures)}; output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
