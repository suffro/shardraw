"""Phase 2: an adaptive suffix in the last MLP, ahead of the q6+q4 LM head, against the full reference.

    uv run python benchmarks/suffix_runtime.py --output experiments/phase2/<name> [--num-prompts N]

Reuses the prompts, model and reference numerics of `source_run` (configs/phase2-suffix.yaml).
For every prompt:

  reference  the unmodified forward, with hooks on every intermediate of the last MLP and the
             final norm (`ReferenceRunner.next_token(intermediates=True)`); every stage's exact
             prefix must equal it bitwise
  lm_head    the Phase 1C runtime on the exact LM-head input (depth 0)
  down, mlp  the adaptive-suffix runtime at every budget, under the certified (faithful) rounding
             model and, as what-ifs that never certify, the experimental ones; with the row limits
             of the LM-head pass on an enclosure ("limited" policy), and without them at the
             ceiling budgets ("unlimited")

Writes, into the output directory:
  config.yaml, environment.json
  store.json          LM-head and MLP stores: layouts, metadata, digests, masked-fallback self-test
  reference.jsonl     per prompt: reference token, top-2 gap, prefix and fallback bitwise checks
  validation.jsonl.gz per prompt × stage × budget × rounding model: enclosure violations of every
                      intermediate and logit state, pairwise-bound violations, exact-suffix and
                      byte checks
  records.jsonl.gz    per prompt × stage × budget × rounding model: decision, path, bytes, baseline,
                      contenders, bound terms, widths, wall time
  digest.json         sha256 of store, reference, validation and records (timing fields excluded)
It stops at the first hard failure (unless --keep-going) and saves it to failure.json: a prefix,
fallback or exact-suffix mismatch; a certified or fallback token mismatch; under the certified
model, an enclosure that misses a reference value or a pairwise bound above the reference's value;
a byte audit failure; a page or row read twice in one token.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from awpmi.bounds.coarse import CoarseArithmetic  # noqa: E402
from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.bounds.rounding import ASSUMPTIONS  # noqa: E402
from awpmi.certificate import TieBreak  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.decomposition import RefinementDecomposition  # noqa: E402
from awpmi.models.smollm2 import (  # noqa: E402
    ModelSpec,
    final_hidden_state,
    last_layer,
    lm_head_weight,
    load_model,
    resolve_dtype,
    suffix_prefix,
)
from awpmi.reference import ReferenceRunner  # noqa: E402
from awpmi.refinement_head import FallbackMode, RefinementLMHead  # noqa: E402
from awpmi.stores.refinement import PackedRefinementStore  # noqa: E402
from awpmi.stores.suffix import MLPStore  # noqa: E402
from awpmi.suffix_runtime import AdaptiveSuffixRuntime, SuffixRunResult, SuffixStage  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl, tensor_digest  # noqa: E402

TIMING_FIELDS = ("wall_ms", "timings_ms")
OUTPUT_FILES = ("store.json", "reference.jsonl", "validation.jsonl.gz", "records.jsonl.gz")


class HardFailure(Exception):
    pass


def last(values: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return values[name][0, -1].to(torch.float64)


def relative_width(lower: torch.Tensor, upper: torch.Tensor, value: torch.Tensor) -> float:
    scale = value.abs().clamp_min(2.0**-126)
    return float(((upper - lower) / scale).median())


# Validation


def head_checks(result, head_store: PackedRefinementStore, decomposition: RefinementDecomposition) -> dict:
    """Each row of each LM-head state read once per token; the store's log equals the decomposition's accounting."""
    per_state = [0] * decomposition.num_states
    marks: dict[tuple[str, int], torch.Tensor] = {}
    twice = 0
    for read in head_store.reads:
        rows = torch.arange(head_store.out_features, device=head_store.device) if read.rows is None else read.rows
        key = ("weight" if read.kind in ("exact", "fallback") else read.kind, read.state)
        mask = marks.setdefault(key, torch.zeros(head_store.out_features, dtype=torch.bool, device=head_store.device))
        twice += int(mask[rows].sum())
        mask[rows] = True
        if read.kind in ("level", "exact"):
            per_state[read.state] += read.row_count
    fallback_rows = result.head_bytes["fallback_rows"] // decomposition.exact_row_bytes
    audit = result.head_bytes == decomposition.materialized_bytes(per_state, fallback_rows)
    return {"head_rows_read_twice": twice, "head_byte_audit": audit}


def suffix_checks(result: SuffixRunResult, runtime: AdaptiveSuffixRuntime) -> dict:
    store = runtime.store
    seen = {role: torch.zeros(store.neurons, dtype=torch.bool, device=store.device) for role in store.adaptive}
    twice = 0
    for read in store.reads:
        twice += int(seen[read.role][read.neurons].sum())
        seen[read.role][read.neurons] = True
    expected = math.ceil(runtime.budget * store.neurons - 1e-9)
    budget_ok = all(count == (store.neurons if role == "gate" else expected) for role, count in result.neurons_read.items())
    pages = sum(read.bytes for read in store.reads)
    audit = result.suffix_bytes["total"] == store.metadata_bytes + runtime.resident_bytes + pages
    return {"suffix_pages_read_twice": twice, "suffix_byte_audit": audit, "suffix_budget_respected": budget_ok}


def logit_violations(states, logits: torch.Tensor) -> int:
    reference = logits.to(torch.float64)
    total = 0
    for bounds in states:
        covered = reference if bounds.rows is None else reference[bounds.rows]
        total += int(((covered < bounds.lower) | (covered > bounds.upper)).sum())
    return total


def enclosure_violations(result: SuffixRunResult, values: dict[str, torch.Tensor]) -> dict[str, int]:
    bounds = result.bounds
    if bounds is None:
        return {}
    read = bounds.read
    a = last(values, "a")
    counts = {
        "o": bounds.output.violations(last(values, "o")),
        "y": bounds.residual_sum.violations(last(values, "y")),
        "n": bounds.norm.normalized.violations(last(values, "n")),
        "h": bounds.norm.output.violations(last(values, "h")),
        "q": int(not bounds.norm.scale_lower <= float(values["q"][0, -1, 0]) <= bounds.norm.scale_upper),
        "a_read": bounds.activation.select(read).violations(a[read]),
        "a_unread": int((a[~read].abs() > bounds.unread_activation[~read]).sum()),
    }
    if bounds.gate is not None:
        counts["g"] = bounds.gate.violations(last(values, "g"))
        counts["s"] = bounds.activation_function.violations(last(values, "s"))
        counts["u"] = bounds.up.violations(last(values, "u")[read])
    return counts


def pairwise_violations(result: SuffixRunResult, values, lm_weight64, gain64) -> dict[str, int]:
    pairwise = result.pairwise
    if pairwise is None:
        return {}
    delta = (lm_weight64[pairwise.winner][None, :] - lm_weight64[pairwise.contenders]) * gain64[None, :]
    y = last(values, "y")
    actual = delta @ y
    tolerance = 1e-9 * (delta.abs() @ y.abs())
    return {
        "box": int((pairwise.box_bound > actual + tolerance).sum()),
        "decomposed": int((pairwise.decomposed_bound > actual + tolerance).sum()),
    }


# Records


def lm_head_record(prompt_id, result: SuffixRunResult, reference_token, wall_ms) -> dict:
    head = result.head
    return {
        "prompt_id": prompt_id,
        "stage": "lm_head",
        "depth": 0,
        "budget": 1.0,
        "assumptions": result.assumptions,
        "policy": "limited",
        "reference_token": reference_token,
        "token": result.token_id,
        "certified": result.certified,
        "would_certify": result.would_certify,
        "path": result.path,
        "reason": head.fallback_reason,
        "head_fallback": head.fallback.value if head.fallback else None,
        "head_guard_tripped": head.masked_guard_tripped,
        "head_decision_state": head.decision_state,
        "head_contenders": list(head.contenders),
        "head_rows_loaded": list(head.rows_loaded),
        "neurons_read": {},
        "suffix_bytes": result.suffix_bytes,
        "head_bytes": result.head_bytes,
        "bytes_total": result.bytes_total,
        "region_bytes": result.region_bytes,
        "fraction": result.fraction,
        "wall_ms": wall_ms,
        "timings_ms": result.timings_ms,
    }


def suffix_record(prompt_id, stage, result: SuffixRunResult, reference_token, values, baseline_bytes, wall_ms) -> dict:
    record = {
        "prompt_id": prompt_id,
        "stage": stage.value,
        "depth": stage.depth,
        "budget": result.budget,
        "assumptions": result.assumptions,
        "reference_token": reference_token,
        "token": result.token_id,
        "certified": result.certified,
        "would_certify": result.would_certify,
        "path": result.path,
        "reason": result.reason,
        "neurons_read": result.neurons_read,
        "suffix_bytes": result.suffix_bytes,
        "head_bytes": result.head_bytes,
        "bytes_total": result.bytes_total,
        "region_bytes": result.region_bytes,
        "fraction": result.fraction,
        "baseline_bytes": baseline_bytes,
        "wall_ms": wall_ms,
        "timings_ms": result.timings_ms,
    }
    if result.bounds is not None:
        bounds = result.bounds
        record["widths"] = {
            "o": relative_width(bounds.output.lower, bounds.output.upper, last(values, "o")),
            "y": relative_width(bounds.residual_sum.lower, bounds.residual_sum.upper, last(values, "y")),
            "h": relative_width(bounds.norm.output.lower, bounds.norm.output.upper, last(values, "h")),
            "scale": (bounds.norm.scale_upper - bounds.norm.scale_lower) / float(values["q"][0, -1, 0]),
        }
    if result.enclosure is not None:
        enclosure = result.enclosure
        record["enclosure"] = {
            "certified": enclosure.certified,
            "reason": enclosure.reason,
            "decision_state": enclosure.decision_state,
            "contenders": list(enclosure.contenders),
            "rows_loaded": list(enclosure.rows_loaded),
            "margin": enclosure.certificate_margin,
        }
    if result.pairwise is not None:
        pairwise = result.pairwise
        record["pairwise"] = {
            "certified": pairwise.certified,
            "contenders": int(pairwise.contenders.numel()),
            "min_margin": float(pairwise.margin.min()) if pairwise.contenders.numel() else None,
            "terms": pairwise.terms,
        }
    if result.head is not None:
        head = result.head
        record.update(
            head_fallback=head.fallback.value if head.fallback else None,
            head_guard_tripped=head.masked_guard_tripped,
            head_decision_state=head.decision_state,
            head_contenders=list(head.contenders),
            head_rows_loaded=list(head.rows_loaded),
        )
    return record


def timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1e3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase2-suffix.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts of the source run")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source_run = REPO_ROOT / raw_config["source_run"]
    source_config, _ = load_config(source_run / "config.yaml")
    runtime_config = raw_config["runtime"]
    stages = [SuffixStage(name) for name in raw_config["stages"]]
    budgets = [float(b) for b in raw_config["budgets"]]
    certified_name = raw_config["certified_assumptions"]
    if not ASSUMPTIONS[certified_name].certified:
        parser.error("certified_assumptions must be the faithful model")
    assumption_names = [certified_name] + list(raw_config["experimental_assumptions"])
    policies = {"limited": tuple(int(v) for v in runtime_config["enclosure_row_limits"]), "unlimited": None}
    ceiling_budgets = [float(b) for b in runtime_config["ceiling_budgets"]]

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(source_config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(
        source_config.model.repository, source_config.model.revision, resolve_dtype(source_config.model.dtype, device), device
    )
    model, _ = load_model(spec)
    runner = ReferenceRunner(model)
    weight = lm_head_weight(model)
    unit_roundoff = float(runtime_config["coarse_unit_roundoff"])
    numerics = ReferenceNumerics(weight.dtype, source_config.numerics.accumulation_unit_roundoff, weight.shape[1])
    prompts = read_jsonl(source_run / "prompts.jsonl")
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    decomposition = RefinementDecomposition.build(weight, raw_config["decomposition"])
    head_store = PackedRefinementStore.from_decomposition(decomposition)
    head = RefinementLMHead(
        head_store,
        numerics,
        CoarseArithmetic(unit_roundoff, weight.shape[1]),
        tie_break=TieBreak(raw_config["tie_break"]),
        fallback=FallbackMode(raw_config["lm_head_fallback"]),
        chunk_rows=int(runtime_config["chunk_rows"]),
        self_test_trials=int(runtime_config["masked_self_test_trials"]),
    )
    layer = last_layer(model)
    layer_index = len(model.model.layers) - 1
    mlp_prefix = f"model.layers.{layer_index}.mlp"
    norm = model.model.norm
    stores = {
        stage: MLPStore.from_mlp(layer.mlp, stage.adaptive_roles, layer_index, mlp_prefix)
        for stage in stages
        if stage is not SuffixStage.LM_HEAD
    }
    lm_head_key = (SuffixStage.LM_HEAD, 1.0, certified_name, "limited")
    runtimes = {lm_head_key: AdaptiveSuffixRuntime(SuffixStage.LM_HEAD, head, norm)}
    cells = [(budget, "limited") for budget in budgets] + [(budget, "unlimited") for budget in ceiling_budgets]
    for stage, store in stores.items():
        for budget, policy in cells:
            for name in assumption_names:
                assumptions = ASSUMPTIONS[name]
                runtimes[(stage, budget, name, policy)] = AdaptiveSuffixRuntime(
                    stage,
                    head,
                    norm,
                    store,
                    layer.mlp.act_fn,
                    accumulation_unit_roundoff=source_config.numerics.accumulation_unit_roundoff,
                    budget=budget,
                    assumptions=assumptions,
                    experimental=not assumptions.certified,
                    enclosure_row_limits=policies[policy],
                )

    store_info = {
        "lm_head": {
            "decomposition": decomposition.spec,
            "storage_bytes": head_store.storage_bytes(),
            "weight_bytes": head_store.weight_bytes,
            "metadata_bytes": head_store.metadata_bytes,
            "store_sha256": head_store.content_digest(),
            "fallback": head.fallback.value,
            "masked_fallback_self_test": head.self_test,
            "masked_enabled": head.masked_enabled,
        },
        "suffix": {
            stage.value: {
                "adaptive_roles": list(store.adaptive),
                "layer": store.layer,
                "pages": len(store.pages()),
                "page_bytes": store.page_bytes,
                "weight_bytes": store.weight_bytes,
                "metadata_bytes": store.metadata_bytes,
                "resident_bytes": runtimes[(stage, budgets[0], certified_name, "limited")].resident_bytes,
                "region_bytes": runtimes[(stage, budgets[0], certified_name, "limited")].region_bytes,
                "store_sha256": store.content_digest(),
            }
            for stage, store in stores.items()
        },
        "final_norm_sha256": tensor_digest(norm.weight.detach()),
    }
    (output / "store.json").write_text(json.dumps(store_info, indent=2), encoding="utf-8")
    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    model_info = {
        "repository": spec.repository,
        "revision": spec.revision,
        "dtype": str(spec.dtype),
        "device": str(device),
        "accumulation_unit_roundoff": source_config.numerics.accumulation_unit_roundoff,
        "final_norm_eps": norm.variance_epsilon,
    }
    environment = environment_metadata(REPO_ROOT, model_info, NUMERICS_FLAGS)
    environment.update(source_run=raw_config["source_run"], num_prompts=len(prompts))
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    lm_weight64 = weight.to(torch.float64)
    gain64 = norm.weight.detach().to(torch.float64)

    def prefixes(ids):
        return {stage: suffix_prefix(model, ids, stage.boundary) for stage in stores}

    for prompt in prompts[: int(raw_config["timing"]["warmup_prompts"])]:
        ids = torch.tensor([prompt["token_ids"]], device=device)
        runtimes[lm_head_key].run(final_hidden_state(model, ids))
        for (stage, _, _, _), runtime in runtimes.items():
            if stage is not SuffixStage.LM_HEAD:
                runtime.run(suffix_prefix(model, ids, stage.boundary))

    failures: list[dict] = []
    started = time.perf_counter()
    with (
        JsonlWriter(output / "reference.jsonl") as reference_log,
        JsonlWriter(output / "validation.jsonl.gz") as validation_log,
        JsonlWriter(output / "records.jsonl.gz") as record_log,
    ):

        def fail(kind: str, entry: dict) -> None:
            failures.append({"kind": kind, **entry})
            if not args.keep_going:
                raise HardFailure

        try:
            for index, prompt in enumerate(prompts):
                prompt_id = prompt["prompt_id"]
                ids = torch.tensor([prompt["token_ids"]], device=device)
                reference = runner.next_token(ids, intermediates=True)
                values = reference.intermediates
                token = reference.token_id
                hidden = final_hidden_state(model, ids)
                full_logits = F.linear(hidden, weight).reshape(-1)
                boundaries = prefixes(ids)
                top2 = torch.topk(reference.logits.to(torch.float32), 2).values.tolist()
                prefix_checks = {"lm_head": bool(torch.equal(hidden, reference.hidden_state))}
                for stage, prefix in boundaries.items():
                    same = torch.equal(prefix.residual, values["residual"]) and torch.equal(prefix.mlp_input, values["x"])
                    if prefix.activation is not None:
                        same = same and torch.equal(prefix.activation, values["a"])
                    prefix_checks[stage.value] = bool(same)
                reference_entry = {
                    "prompt_id": prompt_id,
                    "reference_token": token,
                    "reference_top2_gap": top2[0] - top2[1],
                    "prefix_bitwise_equal": prefix_checks,
                    "fallback_bitwise_equal": bool(torch.equal(full_logits, reference.logits)),
                }
                reference_log.write(reference_entry)
                if not all(prefix_checks.values()) or not reference_entry["fallback_bitwise_equal"]:
                    fail("reference", reference_entry)

                # Depth 0: the Phase 1C LM head on the exact input.
                result, wall = timed(lambda: runtimes[lm_head_key].run(hidden, trace=True), device)
                record_log.write(lm_head_record(prompt_id, result, token, wall))
                head_result = result.head
                validation = {
                    "prompt_id": prompt_id,
                    "stage": "lm_head",
                    "budget": 1.0,
                    "assumptions": certified_name,
                    "policy": "limited",
                    "logit_violations": logit_violations(head_result.trace.states, reference.logits) if head_result.trace else 0,
                    **head_checks(result, head_store, decomposition),
                }
                if head_result.fallback is FallbackMode.FULL:
                    validation["fallback_bitwise"] = bool(torch.equal(head_result.fallback_logits, reference.logits))
                elif head_result.fallback is FallbackMode.MASKED:
                    alive = head_result.trace.final_contenders
                    validation["fallback_bitwise"] = bool(torch.equal(head_result.fallback_logits[alive], reference.logits[alive]))
                validation_log.write(validation)
                if result.token_id != token:
                    fail("certified_mismatch" if result.certified else "fallback_mismatch", validation)
                if validation["logit_violations"] or validation["head_rows_read_twice"] or not validation["head_byte_audit"]:
                    fail("lm_head_validation", validation)
                if validation.get("fallback_bitwise") is False:
                    fail("fallback_not_bitwise", validation)
                lm_head_bytes = result.bytes_total

                for (stage, budget, name, policy), runtime in runtimes.items():
                    if stage is SuffixStage.LM_HEAD:
                        continue
                    baseline = lm_head_bytes + runtime.store.weight_bytes + runtime.resident_bytes
                    result, wall = timed(lambda: runtime.run(boundaries[stage], trace=True), device)
                    record = suffix_record(prompt_id, stage, result, token, values, baseline, wall)
                    record["policy"] = policy
                    record_log.write(record)
                    certified_model = ASSUMPTIONS[name].certified
                    validation = {
                        "prompt_id": prompt_id,
                        "stage": stage.value,
                        "budget": budget,
                        "assumptions": name,
                        "policy": policy,
                        "enclosure_violations": enclosure_violations(result, values),
                        "pairwise_violations": pairwise_violations(result, values, lm_weight64, gain64),
                        "logit_violations": 0,
                        **head_checks(result, head_store, decomposition),
                        **suffix_checks(result, runtime),
                    }
                    if result.enclosure is not None and result.enclosure.trace is not None:
                        validation["logit_violations"] += logit_violations(result.enclosure.trace.states, reference.logits)
                    if result.head is not None and result.head.trace is not None:
                        validation["logit_violations"] += logit_violations(result.head.trace.states, reference.logits)
                    if result.exact_suffix is not None:
                        validation["exact_suffix_bitwise"] = all(
                            bool(torch.equal(result.exact_suffix[key], values[key])) for key in ("o", "y", "h")
                        )
                    if result.head is not None and result.head.fallback is FallbackMode.FULL:
                        validation["fallback_bitwise"] = bool(torch.equal(result.head.fallback_logits, reference.logits))
                    elif result.head is not None and result.head.fallback is FallbackMode.MASKED:
                        alive = result.head.trace.final_contenders
                        validation["fallback_bitwise"] = bool(
                            torch.equal(result.head.fallback_logits[alive], reference.logits[alive])
                        )
                    validation_log.write(validation)
                    if certified_model:
                        if result.token_id != token:
                            fail("certified_mismatch" if result.certified else "fallback_mismatch", record)
                        if any(validation["enclosure_violations"].values()) or any(validation["pairwise_violations"].values()):
                            fail("enclosure_violation", validation)
                        if validation["logit_violations"]:
                            fail("logit_violation", validation)
                    if validation.get("exact_suffix_bitwise") is False:
                        fail("exact_suffix_not_bitwise", validation)
                    if validation.get("fallback_bitwise") is False:
                        fail("fallback_not_bitwise", validation)
                    if validation["head_rows_read_twice"] or validation["suffix_pages_read_twice"]:
                        fail("read_twice", validation)
                    if not (validation["head_byte_audit"] and validation["suffix_byte_audit"] and validation["suffix_budget_respected"]):
                        fail("byte_audit", validation)
                if (index + 1) % 50 == 0:
                    print(f"{index + 1}/{len(prompts)} prompts, {time.perf_counter() - started:.0f}s", flush=True)
        except HardFailure:
            pass

    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
    digest = {
        "store_sha256": canonical_digest([json.loads((output / "store.json").read_text(encoding="utf-8"))]),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl")),
        "validation_sha256": canonical_digest(read_jsonl(output / "validation.jsonl.gz")),
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl.gz"), TIMING_FIELDS),
        "excluded_fields": list(TIMING_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    walls = [r["wall_ms"] for r in read_jsonl(output / "records.jsonl.gz")]
    print(
        f"done in {time.perf_counter() - started:.0f}s; median run {statistics.median(walls) if walls else 0:.0f} ms; "
        f"hard failures: {len(failures)}; output: {output}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
