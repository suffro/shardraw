"""Phase 1C: the q6+q4 LM-head runtime against the full reference and the Phase 1B oracle.

    uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/<name> [--num-prompts N]

Reuses the prompts, model and reference numerics of `source_run` and compares with the
Phase 1B records of `oracle_run` (configs/phase1c-runtime.yaml). For every prompt it
runs the runtime once per fallback mode, validates it against the reference, and
records the bytes the store actually handed out and the time of every stage.

Writes, into the output directory:
  config.yaml        the Phase 1C config as given
  environment.json   versions, hardware, numerics flags, runtime arithmetic, source and oracle runs
  store.json         packed layout: bytes per row and state, storage, digests, masked-fallback self-test
  reference.jsonl    per prompt: reference token, top-2 gap, prefix and full-fallback bitwise checks
  validation.jsonl   per prompt: envelope violations per state, coarse-arithmetic check, masked-fallback
                     check, and the final contenders' exact-state intervals of every fallback
  records.jsonl      per prompt × fallback mode: decision, contenders, rows, bytes, oracle record, timings
  profile.json       micro-benchmarks and a kernel-level profile (timing only, not digested)
  digest.json        sha256 of store, reference, validation and records (timing fields excluded)
It stops at the first hard failure (unless --keep-going) and saves it to failure.json:
a prefix or full-fallback mismatch, a certified or fallback mismatch, a masked fallback
that is not bitwise on the surviving rows, an envelope violation, a coarse-arithmetic
violation, or bytes that differ from the decomposition's accounting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from awpmi.bounds.coarse import CoarseArithmetic  # noqa: E402
from awpmi.bounds.linear import absolute_mass_upper  # noqa: E402
from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.certificate import TieBreak  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.decomposition import RefinementDecomposition  # noqa: E402
from awpmi.decomposition.accounting import IO_BLOCK_BYTES, blocks_touched  # noqa: E402
from awpmi.decomposition.packing import CodeLayout  # noqa: E402
from awpmi.models.smollm2 import ModelSpec, final_hidden_state, lm_head_weight, load_model, resolve_dtype  # noqa: E402
from awpmi.reference import ReferenceRunner  # noqa: E402
from awpmi.refinement_head import FallbackMode, RefinementLMHead, RefinementRunResult, coarse_matvec  # noqa: E402
from awpmi.stores.refinement import PackedRefinementStore  # noqa: E402
from awpmi.tracing import (  # noqa: E402
    JsonlWriter,
    StageTimer,
    canonical_digest,
    environment_metadata,
    read_jsonl,
    tensor_digest,
)

TIMING_FIELDS = ("timings_ms", "reference_ms")
ORACLE_PRIMARY = ("row_selective", "realistic", "lowest_index")
OUTPUT_FILES = ("store.json", "reference.jsonl", "validation.jsonl", "records.jsonl")


class HardFailure(Exception):
    pass


def oracle_records(oracle_run: Path, spec: str) -> dict[int, dict]:
    """The oracle's primary-configuration record of every prompt, for one decomposition."""
    return {
        record["prompt_id"]: record
        for record in read_jsonl(oracle_run / "records.jsonl.gz")
        if record["decomposition"] == spec and (record["mode"], record["bound"], record["tie_break"]) == ORACLE_PRIMARY
    }


def describe_store(store: PackedRefinementStore, decomposition: RefinementDecomposition, oracle_run: Path) -> dict:
    oracle = json.loads((oracle_run / "decompositions.json").read_text(encoding="utf-8"))[decomposition.spec]
    levels_sha256 = tensor_digest(
        *[t for level in decomposition.levels for t in (level.codes, level.scales)],
        *[t for norms in decomposition.remainder_norms for t in (norms.l2, norms.linf)],
    )
    states = range(store.num_states)
    return {
        "decomposition": decomposition.spec,
        "states": [decomposition.state_name(s) for s in states],
        "row_bytes": {f"{s}:{decomposition.state_name(s)}": list(store.row_bytes(s)) for s in states},
        "metadata_bytes": store.metadata_bytes,
        "storage_bytes": store.storage_bytes(),
        "weight_bytes": store.weight_bytes,
        "packing_lossless": True,  # PackedRefinementStore.from_decomposition raises otherwise
        "accounting_matches_decomposition": store.storage_bytes() == decomposition.storage_bytes()
        and all(store.row_bytes(s) == decomposition.row_bytes(s) for s in states),
        "levels_sha256": levels_sha256,
        "levels_match_oracle_run": levels_sha256 == oracle["levels_sha256"],
        "store_sha256": store.content_digest(),
    }


def io_bytes_4k(store: PackedRefinementStore, result: RefinementRunResult) -> int:
    """Block-granular I/O, computed exactly as the Phase 1B oracle does (a fallback reads the whole BF16 file)."""
    blocks = 0
    for state in range(store.num_states):
        rows = torch.zeros(store.out_features, dtype=torch.bool, device=store.device)
        for read in result.reads:
            if read.kind in ("level", "exact") and read.state == state:
                rows[slice(None) if read.rows is None else read.rows] = True
        blocks += int(blocks_touched(rows[:, None], sum(store.row_bytes(state)))[0])
    fallback = any(read.kind == "fallback" and read.row_count for read in result.reads)
    return store.metadata_bytes + IO_BLOCK_BYTES * blocks + (store.weight_bytes if fallback else 0)


def validate(result: RefinementRunResult, reference_logits, base_values, vector64, numerics) -> dict:
    """Checks of a traced run that need the full reference: envelopes, coarse arithmetic, final contenders."""
    trace = result.trace
    if trace is None:
        raise ValueError(f"run without a trace ({result.fallback_reason}): nothing was bounded")
    reference64 = reference_logits.to(torch.float64)
    envelope = {}
    for bounds in trace.states:
        covered = reference64 if bounds.rows is None else reference64[bounds.rows]
        envelope[str(bounds.state)] = int(((covered < bounds.lower) | (covered > bounds.upper)).sum())
    coarse = trace.states[0]
    exact_center = base_values @ vector64
    float64_error = numerics.float64_gamma * absolute_mass_upper(base_values, vector64)
    deviation = (coarse.center - exact_center).abs()
    bounded = trace.coarse_error > 0
    entry = {
        "envelope_violations": envelope,
        "rows_checked": {str(b.state): reference64.numel() if b.rows is None else b.rows.numel() for b in trace.states},
        "coarse_arithmetic_violations": int((deviation > trace.coarse_error + float64_error).sum()),
        # Observed binary32 error over its bound (rows with a zero bound are covered by the violation count).
        "coarse_arithmetic_max_ratio": float((deviation[bounded] / trace.coarse_error[bounded]).max()),
        "final_contenders": None,
    }
    if result.fallback_reason == "uncertified":  # only reached after the exact state
        exact = trace.states[-1]
        alive = trace.final_contenders[exact.rows]
        entry["final_contenders"] = [
            {
                "row": int(row),
                "center": float(center),
                "radius": float(radius),
                "lower": float(trace.lower[row]),
                "upper": float(trace.upper[row]),
                "reference_logit": float(reference64[row]),
            }
            for row, center, radius in zip(
                exact.rows[alive].tolist(), exact.center[alive].tolist(), exact.radius[alive].tolist()
            )
        ]
    return entry


def run_record(prompt_id, mode, result, reference_token, store, decomposition, oracle, reference_ms) -> dict:
    states = store.num_states
    decision = result.decision_state
    timings = dict(result.timings_ms)
    timings["total"] = sum(result.timings_ms.values())
    return {
        "prompt_id": prompt_id,
        "fallback_mode": mode.value,
        "reference_token": reference_token,
        "token": result.token_id,
        "certified": result.certified,
        "fallback": result.fallback.value if result.fallback else None,
        "fallback_reason": result.fallback_reason,
        "masked_guard_tripped": result.masked_guard_tripped,
        "decision_state": decomposition.state_name(decision) if decision is not None else None,
        "decision_state_index": decision,
        "winner": result.winner,
        "competitor": result.competitor,
        "certificate_margin": result.certificate_margin,
        "contenders": list(result.contenders) + [-1] * (states - len(result.contenders)),
        "rows_loaded": list(result.rows_loaded),
        "bytes": result.bytes,
        "fraction": result.bytes["total"] / store.weight_bytes,
        "io_bytes_4k": io_bytes_4k(store, result),
        "oracle": {
            "token": oracle["token"],
            "certified": oracle["certified"],
            "decision_state_index": oracle["decision_state_index"],
            "contenders": oracle["contenders"],
            "rows_loaded": oracle["rows_loaded"],
            "fraction": oracle["fraction"],
            "fraction_masked_fallback": oracle["fraction_masked_fallback"],
            "io_bytes_4k": oracle["io_bytes_4k"],
        },
        "timings_ms": timings,
        "reference_ms": reference_ms,
    }


def mean_ms(function, repetitions: int, device: torch.device) -> float:
    function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repetitions * 1e3


def profile_runtime(head, store, weight, vectors, repetitions, device) -> dict:
    """Micro-benchmarks of the pieces of a run, and kernel time against wall time for a few runs."""
    chunk = head.chunk_rows
    store.begin_run()
    payload, _ = store.read_level(0)
    layout = CodeLayout(store.level_bits(0), store.in_features, device)
    vector = vectors[0][1]
    vector32 = vector.to(torch.float32)
    decoded = torch.cat([layout.unpack(payload[s : s + chunk]) for s in range(0, payload.shape[0], chunk)]).to(
        torch.float32
    )

    def decode() -> None:
        for start in range(0, payload.shape[0], chunk):
            layout.unpack(payload[start : start + chunk]).to(torch.float32)

    micro = {
        "reference_lm_head_ms": mean_ms(lambda: F.linear(vector.view(1, 1, -1), weight), repetitions, device),
        "coarse_decode_ms": mean_ms(decode, repetitions, device),
        "coarse_mv_on_decoded_ms": mean_ms(lambda: torch.mv(decoded, vector32), repetitions, device),
        "coarse_matvec_ms": mean_ms(lambda: coarse_matvec(payload, layout, vector32, chunk), repetitions, device),
    }
    del decoded
    runs = []
    for position, row in vectors:
        wall = mean_ms(lambda: head.run(row), max(repetitions // 10, 5), device)
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        with profile(activities=activities) as profiler:
            result = head.run(row)
        # Kernels, copies and fills are the device-side events; operator events only re-attribute them.
        device_us = sum(e.device_time_total for e in profiler.events() if e.device_type == DeviceType.CUDA)
        launches = sum(e.count for e in profiler.key_averages() if e.key.startswith("cudaLaunchKernel"))
        runs.append(
            {
                "prompt_position": position,
                "decision_state_index": result.decision_state,
                "fallback": result.fallback.value if result.fallback else None,
                "rows_loaded": list(result.rows_loaded),
                "wall_ms_without_stage_sync": wall,
                "device_kernel_ms": device_us / 1e3,
                "kernel_launches": launches,
            }
        )
    store.begin_run()
    return {"repetitions": repetitions, "micro": micro, "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase1c-runtime.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts of the source run")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source_run = REPO_ROOT / raw_config["source_run"]
    oracle_run = REPO_ROOT / raw_config["oracle_run"]
    source_config, _ = load_config(source_run / "config.yaml")
    spec_name = raw_config["decomposition"]
    tie_break = TieBreak(raw_config["tie_break"])
    modes = [FallbackMode(name) for name in raw_config["fallbacks"]]
    runtime_config, timing_config = raw_config["runtime"], raw_config["timing"]

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
    coarse = CoarseArithmetic(float(runtime_config["coarse_unit_roundoff"]), weight.shape[1])
    prompts = read_jsonl(source_run / "prompts.jsonl")
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    oracle = oracle_records(oracle_run, spec_name)

    decomposition = RefinementDecomposition.build(weight, spec_name)
    store = PackedRefinementStore.from_decomposition(decomposition)
    heads = {
        mode: RefinementLMHead(
            store,
            numerics,
            coarse,
            tie_break=tie_break,
            fallback=mode,
            chunk_rows=int(runtime_config["chunk_rows"]),
            self_test_trials=int(runtime_config["masked_self_test_trials"]),
        )
        for mode in modes
    }
    store_info = describe_store(store, decomposition, oracle_run)
    store_info["masked_fallback_self_test"] = {mode.value: head.self_test for mode, head in heads.items()}
    (output / "store.json").write_text(json.dumps(store_info, indent=2), encoding="utf-8")
    base_values = decomposition.levels[0].values()

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
        "coarse_arithmetic": {"unit_roundoff": coarse.unit_roundoff, "gamma": coarse.gamma},
    }
    environment = environment_metadata(REPO_ROOT, model_info, NUMERICS_FLAGS)
    environment["source_run"] = raw_config["source_run"]
    environment["oracle_run"] = raw_config["oracle_run"]
    environment["num_prompts"] = len(prompts)
    environment["io_block_bytes"] = IO_BLOCK_BYTES
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    def lm_head_input(prompt) -> tuple:
        input_ids = torch.tensor([prompt["token_ids"]], device=device)
        reference = runner.next_token(input_ids)
        hidden = final_hidden_state(model, input_ids)
        return reference, hidden, hidden.reshape(-1).contiguous()

    timer = StageTimer(device, synchronize=bool(timing_config["synchronize"]))
    for prompt in prompts[: int(timing_config["warmup_prompts"])]:
        _, _, vector = lm_head_input(prompt)
        F.linear(vector.view(1, 1, -1), weight)
        for head in heads.values():
            head.run(vector, timer=timer, trace=True)
    profiled = [
        (position, lm_head_input(prompts[position])[2])
        for position in timing_config["profile_prompts"]
        if position < len(prompts)
    ]
    if profiled:
        profile_info = profile_runtime(
            heads[modes[0]], store, weight, profiled, int(timing_config["profile_repetitions"]), device
        )
        (output / "profile.json").write_text(json.dumps(profile_info, indent=2), encoding="utf-8")

    failures: list[dict] = []
    started = time.perf_counter()
    with (
        JsonlWriter(output / "reference.jsonl") as reference_log,
        JsonlWriter(output / "validation.jsonl") as validation_log,
        JsonlWriter(output / "records.jsonl") as record_log,
    ):

        def fail(kind: str, entry: dict) -> None:
            failures.append({"kind": kind, **entry})
            if not args.keep_going:
                raise HardFailure

        try:
            for index, prompt in enumerate(prompts):
                prompt_id = prompt["prompt_id"]
                reference, hidden, vector = lm_head_input(prompt)
                timer.start()
                full_logits = F.linear(vector.view(1, 1, -1), weight).reshape(-1)
                timer.mark("reference")
                reference_ms = timer.stages["reference"]
                top2 = torch.topk(reference.logits.to(torch.float32), 2).values.tolist()
                reference_entry = {
                    "prompt_id": prompt_id,
                    "reference_token": reference.token_id,
                    "reference_top2_gap": top2[0] - top2[1],
                    "prefix_bitwise_equal": bool(torch.equal(hidden, reference.hidden_state)),
                    "fallback_bitwise_equal": bool(torch.equal(full_logits, reference.logits)),
                    "fallback_token": int(torch.argmax(full_logits)),
                }
                reference_log.write(reference_entry)
                if not (reference_entry["prefix_bitwise_equal"] and reference_entry["fallback_bitwise_equal"]) or (
                    reference_entry["fallback_token"] != reference.token_id
                ):
                    fail("reference", reference_entry)

                validation = {"prompt_id": prompt_id}
                for mode in modes:
                    result = heads[mode].run(vector, timer=timer, trace=True)
                    record = run_record(
                        prompt_id, mode, result, reference.token_id, store, decomposition, oracle[prompt_id], reference_ms
                    )
                    record_log.write(record)
                    if result.token_id != reference.token_id:
                        fail("certified_mismatch" if result.certified else "fallback_mismatch", record)
                    fallback_rows = result.bytes["fallback_rows"] // store.exact_row_bytes
                    if result.bytes != decomposition.materialized_bytes(list(result.rows_loaded), fallback_rows):
                        fail("byte_audit", record)
                    if result.fallback is FallbackMode.FULL and not torch.equal(result.fallback_logits, reference.logits):
                        fail("full_fallback_not_bitwise", record)
                    if mode is FallbackMode.MASKED:
                        alive = result.trace.final_contenders
                        masked = result.fallback is FallbackMode.MASKED
                        validation["masked_fallback_checked"] = masked
                        validation["masked_fallback_bitwise"] = (
                            bool(torch.equal(result.fallback_logits[alive], reference.logits[alive])) if masked else None
                        )
                        validation["masked_guard_tripped"] = result.masked_guard_tripped
                        if masked and not validation["masked_fallback_bitwise"]:
                            fail("masked_fallback_not_bitwise", record)
                    if mode is modes[0]:
                        validation.update(validate(result, reference.logits, base_values, vector.to(torch.float64), numerics))
                validation_log.write(validation)
                if any(validation["envelope_violations"].values()) or validation["coarse_arithmetic_violations"]:
                    fail("envelope_violation", validation)
                if (index + 1) % 100 == 0:
                    print(f"{index + 1}/{len(prompts)} prompts, {time.perf_counter() - started:.0f}s", flush=True)
        except HardFailure:
            pass

    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    digest = {
        "store_sha256": canonical_digest([json.loads((output / "store.json").read_text(encoding="utf-8"))]),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl")),
        "validation_sha256": canonical_digest(read_jsonl(output / "validation.jsonl")),
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl"), TIMING_FIELDS),
        "excluded_fields": list(TIMING_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - started:.0f}s; hard failures: {len(failures)}; output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
