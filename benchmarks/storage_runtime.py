"""Phase 3, step 2: the Phase 1C LM head on physical storage, against the full BF16 head and the resident runtime.

    uv run python benchmarks/storage_runtime.py --output experiments/phase3/<name> [--num-prompts N]

Reuses the prompts, model and reference numerics of `source_run` (configs/phase3-storage.yaml).
Builds (or re-opens and re-hashes) a refinement pack: level records in an AWPMI safetensors
file, exact rows read straight from the published checkpoint. For every prompt it runs every
configuration of the config:

  A  the full BF16 LM head read from a tier, moved to the device, reference GEMV
  B  the resident Phase 1C runtime
  C  the same runtime reading its segments from a tier (direct I/O, pinned host memory, cached base)

and records, per run, the logical bytes (the store's log), the physical bytes read, read
calls, extents and 4 KiB blocks, the bytes moved host-to-device, the cache's work, the time
of every stage, peak device memory and process memory.

Writes, into the output directory:
  config.yaml, environment.json   as every benchmark (environment adds the storage device)
  pack.json         the pack's manifest, verification, and per-configuration resident bytes
  reference.jsonl   per prompt: reference token, top-2 gap, prefix and full-GEMV bitwise checks
  validation.jsonl  per prompt: envelopes and coarse arithmetic (B, masked), masked-fallback
                    check, and every storage audit
  records.jsonl.gz  per prompt × configuration × fallback mode
  ranges.jsonl.gz   the read extents of the primary configuration for the first prompts
  profile.json      storage and transfer micro-benchmarks (timing only, not digested)
  digest.json       sha256 of pack, reference, validation and records (timings and system fields excluded)
It stops at the first hard failure (unless --keep-going) and saves it to failure.json: any
Phase 1C failure, a full-head GEMV that is not bitwise, a C run that differs from B, a B
run that differs from the Phase 1C record, or a storage audit that does not hold.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from awpmi.bounds.coarse import CoarseArithmetic  # noqa: E402
from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.certificate import TieBreak  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.decomposition import RefinementDecomposition  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.models.smollm2 import ModelSpec, final_hidden_state, lm_head_weight, load_model, resolve_dtype  # noqa: E402
from awpmi.reference import ReferenceRunner  # noqa: E402
from awpmi.refinement_head import FallbackMode, RefinementLMHead, RefinementRunResult  # noqa: E402
from awpmi.storage.cache import LRUPolicy, PageCache  # noqa: E402
from awpmi.storage.fileio import DIRECT_ALIGNMENT, aligned_host_buffer, process_memory  # noqa: E402
from awpmi.storage.pack import MANIFEST, Pack, SourceFile, open_pack  # noqa: E402
from awpmi.storage.store import IO_BLOCK_BYTES  # noqa: E402
from awpmi.stores.refinement import EXACT_SEGMENT, PackedRefinementStore, level_segment, write_refinement_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import JsonlWriter, StageTimer, canonical_digest, environment_metadata, read_jsonl  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from refinement_runtime import io_bytes_4k, mean_ms, validate  # noqa: E402

EXCLUDED_FIELDS = ("timings_ms", "system")
PHASE1C_FIELDS = (
    "token", "certified", "fallback", "fallback_reason", "masked_guard_tripped", "decision_state_index",
    "winner", "competitor", "certificate_margin", "contenders", "rows_loaded", "bytes", "fraction", "io_bytes_4k",
)


class HardFailure(Exception):
    pass


@dataclass
class Configuration:
    name: str
    kind: str  # full_head, resident or file
    tier: str | None = None  # direct, buffered or host (full_head and file)
    cache_base: bool = False
    fallbacks: list[FallbackMode] = field(default_factory=list)
    backend: MaterializationBackend | None = None
    store: PackedRefinementStore | None = None
    heads: dict[FallbackMode, RefinementLMHead] = field(default_factory=dict)


def describe_disk(path: Path) -> dict | None:
    """Best effort: the physical disk under `path` (Windows only)."""
    if sys.platform != "win32":
        return None
    drive = path.resolve().drive.rstrip(":")
    command = (
        f"$d = Get-Partition -DriveLetter {drive} | Get-Disk; $p = Get-PhysicalDisk | Where-Object DeviceId -eq $d.Number; "
        "[pscustomobject]@{model=$d.FriendlyName; bus=[string]$d.BusType; media=[string]$p.MediaType; "
        "logical_sector=$p.LogicalSectorSize; physical_sector=$p.PhysicalSectorSize} | ConvertTo-Json"
    )
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=60)
        return json.loads(output.stdout) if output.returncode == 0 and output.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def ensure_pack(config: dict, decomposition: RefinementDecomposition, model_spec: ModelSpec) -> tuple[Pack, dict]:
    """Open the configured pack (re-hashing every segment), or write it first."""
    directory = REPO_ROOT / config["directory"]
    started = time.perf_counter()
    built = not (directory / MANIFEST).exists()
    if built:
        source = SourceFile(model_spec.repository, model_spec.revision, config["source_file"])
        write_refinement_pack(
            decomposition,
            directory,
            source=(source, config["source_tensor"]),
            packing={"alignment": DIRECT_ALIGNMENT, "tool": "benchmarks/storage_runtime.py"},
        )
    pack = open_pack(directory, verify="segments")
    info = {"directory": config["directory"], "built_now": built, "open_and_verify_s": time.perf_counter() - started}
    return pack, info


def make_backend(
    pack: Pack, device: torch.device, storage: dict, tier: str, cache_base: bool, segments: list[str] | None = None
) -> MaterializationBackend:
    if tier == "host":
        store = pack.load(segments)
    elif tier in ("direct", "buffered"):
        store = pack.store(
            direct=tier == "direct",
            alignment=int(storage["alignment"]),
            max_gap=int(storage["max_gap"]),
            workers=int(storage["workers"]),
            max_read_bytes=int(storage["max_read_bytes"]),
            max_extent_bytes=int(storage["max_extent_bytes"]),
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")
    cache = None
    if cache_base:
        # Exactly the base level's bytes: nothing else is ever admitted.
        cache = PageCache(pack.segments[level_segment(0)].nbytes, LRUPolicy())
    backend = MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])), cache)
    if cache_base:
        backend.pin(level_segment(0))
    return backend


def same_run(a: RefinementRunResult, b: RefinementRunResult) -> list[str]:
    """Fields on which two runs on the same input differ (decision, reads, bounds, fallback logits), bit for bit."""
    differences = []
    for name in (
        "token_id", "certified", "fallback", "fallback_reason", "masked_guard_tripped", "decision_state",
        "winner", "competitor", "certificate_margin", "contenders", "rows_loaded", "bytes",
    ):
        if getattr(a, name) != getattr(b, name):
            differences.append(name)
    if [(r.kind, r.state, r.row_count) for r in a.reads] != [(r.kind, r.state, r.row_count) for r in b.reads]:
        differences.append("reads")
    else:
        for ra, rb in zip(a.reads, b.reads):
            if (ra.rows is None) != (rb.rows is None) or (ra.rows is not None and not torch.equal(ra.rows, rb.rows)):
                differences.append("read_rows")
                break
    if (a.fallback_logits is None) != (b.fallback_logits is None) or (
        a.fallback_logits is not None and not torch.equal(a.fallback_logits, b.fallback_logits)
    ):
        differences.append("fallback_logits")
    ta, tb = a.trace, b.trace
    if not (torch.equal(ta.coarse_sum, tb.coarse_sum) and torch.equal(ta.lower, tb.lower) and torch.equal(ta.upper, tb.upper)):
        differences.append("final_bounds")
    if len(ta.states) != len(tb.states) or any(
        not torch.equal(getattr(sa, f), getattr(sb, f)) for sa, sb in zip(ta.states, tb.states) for f in ("center", "radius", "lower", "upper")
    ):
        differences.append("state_bounds")
    return differences


def storage_audit(tier: str, report: dict, requested_bytes: int, cuda: bool) -> list[str]:
    """What must hold for a run served through a materialization backend (empty when everything holds)."""
    problems = []
    served, storage = report["materialization"], report["storage"]
    transfer = report.get("transfer", {})
    if served["requested_bytes"] != requested_bytes:
        problems.append("requested_bytes != store log")
    if served["cache_hit_bytes"] + served["fetched_bytes"] != served["requested_bytes"]:
        problems.append("cache hits + fetched != requested")
    if storage["logical_bytes"] != served["fetched_bytes"]:
        problems.append("storage logical != fetched")
    if cuda and transfer.get("h2d_bytes") != storage["logical_bytes"]:
        problems.append("h2d != storage logical")
    if tier == "host":
        if storage["physical_bytes"] != storage["logical_bytes"]:
            problems.append("host-memory gathers != requested rows")
        return problems
    blocks = storage["blocks_4k"] * IO_BLOCK_BYTES
    if not 0 <= blocks - storage["physical_bytes"] < IO_BLOCK_BYTES * max(1, storage["requests"]):
        problems.append("physical bytes are not the requested rows' 4 KiB blocks")
    if tier == "direct" and storage["os_read_bytes"] is not None and (
        storage["os_read_bytes"] != storage["physical_bytes"] or storage["os_read_calls"] != storage["read_calls"]
    ):
        problems.append("OS counters differ from the store's reads")
    return problems


def system_fields(report: dict, peak_vram: int, memory: dict | None) -> dict:
    storage = report["storage"]
    transfer = report.get("transfer", {})
    return {
        "io_ms": storage.get("io_ms", 0.0),
        "copy_ms": transfer.get("copy_ms", 0.0),
        "os_read_calls": storage.get("os_read_calls"),
        "os_read_bytes": storage.get("os_read_bytes"),
        "peak_vram_bytes": peak_vram,
        "process_resident_bytes": None if memory is None else memory["resident_bytes"],
        "process_peak_resident_bytes": None if memory is None else memory["peak_resident_bytes"],
    }


def transfer_record(report: dict, configuration: Configuration, weight_bytes: int) -> dict:
    """Bytes per token by where they came from: the drive (`physical`), host memory, host-to-device."""
    storage = report["storage"]
    transfer = report.get("transfer", {})
    from_drive = configuration.tier in ("direct", "buffered")
    physical = storage["physical_bytes"] if from_drive else 0
    host_memory = storage["physical_bytes"] if configuration.tier == "host" else 0
    h2d = transfer.get("h2d_bytes", 0) if configuration.kind != "resident" else 0
    return {
        "tier": configuration.tier or "device",
        "materialization": report["materialization"],
        "storage": {key: storage[key] for key in ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k", "by_segment")},
        "transfer": {key: transfer.get(key, 0) for key in ("h2d_bytes", "h2d_copies", "pieces", "gathered_bytes")},
        "cache": report.get("cache"),
        "physical_fraction": physical / weight_bytes,
        "host_memory_fraction": host_memory / weight_bytes,
        "h2d_fraction": h2d / weight_bytes,
        "blocks_4k_fraction": storage["blocks_4k"] * IO_BLOCK_BYTES / weight_bytes if from_drive else 0.0,
    }


def measure(function, device: torch.device):
    """Run `function`, returning its result, peak device memory above the start, and process memory after."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = function()
    peak = (torch.cuda.max_memory_allocated(device) - start) if cuda else 0
    return result, peak, process_memory()


def storage_profile(pack: Pack, device: torch.device, storage: dict, repetitions: int) -> dict:
    """Throughput of the pieces of a run: sequential and random direct reads, pinned host-to-device copies."""
    store = pack.store(direct=True, workers=int(storage["workers"]), max_read_bytes=int(storage["max_read_bytes"]))
    level0, level1 = pack.segments[level_segment(0)], pack.segments[level_segment(1)]
    staging = aligned_host_buffer(level0.nbytes + 2 * DIRECT_ALIGNMENT)
    generator = torch.Generator().manual_seed(0)
    profile = {}
    try:
        plan = store.plan(level0.name, None)
        times = []
        for _ in range(repetitions):
            started = time.perf_counter()
            store.read_extents(plan, plan.extents, staging)
            times.append(time.perf_counter() - started)
        profile["sequential_read_ms"] = 1e3 * sum(times) / len(times)
        profile["sequential_read_gbps"] = level0.nbytes / (sum(times) / len(times)) / 1e9
        for count in (32, 1024):
            times = []
            for _ in range(repetitions):
                rows = torch.randperm(level1.rows, generator=generator)[:count].sort().values
                plan = store.plan(level1.name, rows)
                started = time.perf_counter()
                store.read_extents(plan, plan.extents, staging)
                times.append((time.perf_counter() - started, plan.extents.shape[0]))
            mean = sum(t for t, _ in times) / len(times)
            extents = sum(e for _, e in times) / len(times)
            profile[f"random_{count}_rows_ms"] = 1e3 * mean
            profile[f"random_{count}_rows_us_per_extent"] = 1e6 * mean / extents
        if device.type == "cuda":
            target = torch.empty(level0.nbytes, dtype=torch.uint8, device=device)
            host = staging[: level0.nbytes]
            profile["h2d_pinned_ms"] = mean_ms(lambda: target.copy_(host, non_blocking=True), repetitions, device)
            profile["h2d_pinned_gbps"] = level0.nbytes / (profile["h2d_pinned_ms"] / 1e3) / 1e9
    finally:
        store.close()
    profile["repetitions"] = repetitions
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase3-storage.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts of the source run")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    args = parser.parse_args()

    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source_run = REPO_ROOT / raw_config["source_run"]
    phase1c_run = REPO_ROOT / raw_config["phase1c_run"]
    source_config, _ = load_config(source_run / "config.yaml")
    runtime_config, storage_config, timing_config = raw_config["runtime"], raw_config["storage"], raw_config["timing"]
    tie_break = TieBreak(raw_config["tie_break"])
    primary = (raw_config["primary"]["configuration"], FallbackMode(raw_config["primary"]["fallback"]))

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(source_config.model.device if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    spec = ModelSpec(source_config.model.repository, source_config.model.revision, resolve_dtype(source_config.model.dtype, device), device)
    model, _ = load_model(spec)
    runner = ReferenceRunner(model)
    weight = lm_head_weight(model)
    weight_bytes = weight.numel() * weight.element_size()
    numerics = ReferenceNumerics(weight.dtype, source_config.numerics.accumulation_unit_roundoff, weight.shape[1])
    coarse = CoarseArithmetic(float(runtime_config["coarse_unit_roundoff"]), weight.shape[1])
    prompts = read_jsonl(source_run / "prompts.jsonl")
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    phase1c = {(r["prompt_id"], r["fallback_mode"]): r for r in read_jsonl(phase1c_run / "records.jsonl")}

    decomposition = RefinementDecomposition.build(weight, raw_config["decomposition"])
    base_values = decomposition.levels[0].values()
    pack, pack_info = ensure_pack(raw_config["pack"], decomposition, spec)
    phase1c_store = json.loads((phase1c_run / "store.json").read_text(encoding="utf-8"))
    pack_info["levels_match_phase1c"] = pack.metadata["levels_sha256"] == phase1c_store["levels_sha256"]

    def heads_for(store: PackedRefinementStore, modes: list[FallbackMode]) -> dict[FallbackMode, RefinementLMHead]:
        return {
            mode: RefinementLMHead(
                store, numerics, coarse, tie_break=tie_break, fallback=mode,
                chunk_rows=int(runtime_config["chunk_rows"]), self_test_trials=int(runtime_config["masked_self_test_trials"]),
            )
            for mode in modes
        }

    configurations: list[Configuration] = []
    resident_digest = None
    for entry in raw_config["configurations"]:
        configuration = Configuration(entry["name"], entry["kind"], entry.get("tier"), bool(entry.get("cache_base", False)))
        configuration.fallbacks = [FallbackMode(name) for name in entry.get("fallbacks", [])]
        if configuration.kind == "full_head":
            configuration.backend = make_backend(pack, device, storage_config, configuration.tier, False, [EXACT_SEGMENT])
        elif configuration.kind == "resident":
            configuration.store = PackedRefinementStore.from_decomposition(decomposition)
            configuration.backend = configuration.store.backend
            resident_digest = configuration.store.content_digest()
        elif configuration.kind == "file":
            configuration.backend = make_backend(pack, device, storage_config, configuration.tier, configuration.cache_base)
            configuration.store = PackedRefinementStore.from_pack(pack, configuration.backend)
        else:
            raise ValueError(f"unknown configuration kind {configuration.kind!r}")
        if configuration.store is not None:
            configuration.heads = heads_for(configuration.store, configuration.fallbacks)
        configurations.append(configuration)
    if configurations[0].kind != "full_head" or not any(c.kind == "resident" for c in configurations):
        raise ValueError("the configurations must start with the full-head baselines and include the resident runtime")
    resident = next(c for c in configurations if c.kind == "resident")

    pack_info["manifest"] = pack.manifest
    pack_info["weight_bytes"] = weight_bytes
    pack_info["store_sha256"] = {
        c.name: c.store.content_digest() for c in configurations if c.store is not None
    }
    pack_info["store_matches_resident"] = all(d == resident_digest for d in pack_info["store_sha256"].values())
    pack_info["store_matches_phase1c"] = resident_digest == phase1c_store["store_sha256"]
    pack_info["masked_fallback_self_test"] = {
        c.name: {mode.value: head.self_test for mode, head in c.heads.items()} for c in configurations if c.heads
    }
    pack_info["resident_bytes"] = {
        c.name: {
            "device": c.backend.device_resident_bytes + (c.store.metadata_bytes if c.store is not None else 0),
            "device_metadata": c.store.metadata_bytes if c.store is not None else 0,
            "host": c.backend.host_resident_bytes,
        }
        for c in configurations
    }
    (output / "pack.json").write_text(json.dumps(pack_info, indent=2), encoding="utf-8")
    if not (pack_info["levels_match_phase1c"] and pack_info["store_matches_resident"] and pack_info["store_matches_phase1c"]):
        (output / "failure.json").write_text(json.dumps([{"kind": "pack", **{k: pack_info[k] for k in ("levels_match_phase1c", "store_matches_resident", "store_matches_phase1c")}}], indent=2), encoding="utf-8")
        print("the pack does not hold the Phase 1C store; stopping")
        return 1

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
    environment["phase1c_run"] = raw_config["phase1c_run"]
    environment["num_prompts"] = len(prompts)
    environment["storage"] = {
        "pack_files": {key: {"path": str(path), "bytes": path.stat().st_size} for key, path in pack.files.items()},
        "disk": describe_disk(pack.directory),
        "source_disk": describe_disk(pack.files["source"]),
    }
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    def lm_head_input(prompt) -> tuple:
        input_ids = torch.tensor([prompt["token_ids"]], device=device)
        reference = runner.next_token(input_ids)
        hidden = final_hidden_state(model, input_ids)
        return reference, hidden, hidden.reshape(-1).contiguous()

    timer = StageTimer(device, synchronize=bool(timing_config["synchronize"]))

    def run_full_head(configuration: Configuration, vector: torch.Tensor):
        backend = configuration.backend
        backend.reset_stats()

        def go():
            timer.start()
            data = backend.materialize(EXACT_SEGMENT)
            timer.mark("read")
            logits = F.linear(vector.view(1, 1, -1), data.view(weight.dtype).view(weight.shape)).reshape(-1)
            timer.mark("linear")
            return logits

        logits, peak, memory = measure(go, device)
        return logits, backend.report(), dict(timer.stages), peak, memory

    def run_head(configuration: Configuration, mode: FallbackMode, vector: torch.Tensor, record_ranges: bool):
        backend = configuration.backend
        backend.reset_stats(record_ranges=record_ranges)
        result, peak, memory = measure(lambda: configuration.heads[mode].run(vector, timer=timer, trace=True), device)
        return result, backend.report(), peak, memory

    # Let the OS finish closing the cached file objects of setup (header reads, model loading)
    # before direct reads are timed: until then each direct read pays for cache coherency.
    gc.collect()
    time.sleep(float(timing_config["settle_seconds"]))
    for prompt in prompts[: int(timing_config["warmup_prompts"])]:
        _, _, vector = lm_head_input(prompt)
        for configuration in configurations:
            if configuration.kind == "full_head":
                run_full_head(configuration, vector)
            for mode in configuration.fallbacks:
                run_head(configuration, mode, vector, False)
    profile = storage_profile(pack, device, storage_config, repetitions=20)
    (output / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    failures: list[dict] = []
    started = time.perf_counter()
    with (
        JsonlWriter(output / "reference.jsonl") as reference_log,
        JsonlWriter(output / "validation.jsonl") as validation_log,
        JsonlWriter(output / "records.jsonl.gz") as record_log,
        JsonlWriter(output / "ranges.jsonl.gz") as range_log,
    ):

        def fail(kind: str, entry: dict) -> None:
            failures.append({"kind": kind, **entry})
            if not args.keep_going:
                raise HardFailure

        try:
            for index, prompt in enumerate(prompts):
                prompt_id = prompt["prompt_id"]
                reference, hidden, vector = lm_head_input(prompt)
                full_logits = F.linear(vector.view(1, 1, -1), weight).reshape(-1)
                top2 = torch.topk(reference.logits.to(torch.float32), 2).values.tolist()
                reference_entry = {
                    "prompt_id": prompt_id,
                    "reference_token": reference.token_id,
                    "reference_top2_gap": top2[0] - top2[1],
                    "prefix_bitwise_equal": bool(torch.equal(hidden, reference.hidden_state)),
                    "fallback_bitwise_equal": bool(torch.equal(full_logits, reference.logits)),
                }
                reference_log.write(reference_entry)
                if not (reference_entry["prefix_bitwise_equal"] and reference_entry["fallback_bitwise_equal"]):
                    fail("reference", reference_entry)

                validation = {"prompt_id": prompt_id, "storage_audit": {}, "parity": {}}
                resident_results: dict[FallbackMode, RefinementRunResult] = {}
                for configuration in configurations:
                    if configuration.kind == "full_head":
                        logits, report, timings, peak, memory = run_full_head(configuration, vector)
                        timings["total"] = sum(timings.values())
                        record = {
                            "prompt_id": prompt_id,
                            "configuration": configuration.name,
                            "fallback_mode": None,
                            "reference_token": reference.token_id,
                            "token": int(torch.argmax(logits)),
                            "certified": False,
                            "fallback": None,
                            "logits_bitwise_equal": bool(torch.equal(logits, reference.logits)),
                            "fraction": 1.0,
                            **transfer_record(report, configuration, weight_bytes),
                            "timings_ms": timings,
                            "system": system_fields(report, peak, memory),
                        }
                        record_log.write(record)
                        problems = storage_audit(configuration.tier, report, weight_bytes, cuda)
                        validation["storage_audit"][configuration.name] = problems
                        if not record["logits_bitwise_equal"] or record["token"] != reference.token_id:
                            fail("full_head_not_bitwise", record)
                        if problems:
                            fail("storage_audit", {"configuration": configuration.name, "problems": problems, "prompt_id": prompt_id})
                        continue
                    for mode in configuration.fallbacks:
                        record_ranges = (configuration.name, mode) == primary and index < int(timing_config["record_ranges_prompts"])
                        result, report, peak, memory = run_head(configuration, mode, vector, record_ranges)
                        store = configuration.store
                        timings = dict(result.timings_ms)
                        timings["total"] = sum(result.timings_ms.values())
                        record = {
                            "prompt_id": prompt_id,
                            "configuration": configuration.name,
                            "fallback_mode": mode.value,
                            "reference_token": reference.token_id,
                            "token": result.token_id,
                            "certified": result.certified,
                            "fallback": result.fallback.value if result.fallback else None,
                            "fallback_reason": result.fallback_reason,
                            "masked_guard_tripped": result.masked_guard_tripped,
                            "decision_state": decomposition.state_name(result.decision_state) if result.decision_state is not None else None,
                            "decision_state_index": result.decision_state,
                            "winner": result.winner,
                            "competitor": result.competitor,
                            "certificate_margin": result.certificate_margin,
                            "contenders": list(result.contenders) + [-1] * (store.num_states - len(result.contenders)),
                            "rows_loaded": list(result.rows_loaded),
                            "bytes": result.bytes,
                            "fraction": result.bytes["total"] / weight_bytes,
                            "io_bytes_4k": io_bytes_4k(store, result),
                            **transfer_record(report, configuration, weight_bytes),
                            "timings_ms": timings,
                            "system": system_fields(report, peak, memory),
                        }
                        record_log.write(record)
                        if record_ranges:
                            range_log.write({"prompt_id": prompt_id, "configuration": configuration.name, "ranges": report["storage"].get("ranges", [])})
                        if result.token_id != reference.token_id:
                            fail("certified_mismatch" if result.certified else "fallback_mismatch", record)
                        fallback_rows = result.bytes["fallback_rows"] // store.exact_row_bytes
                        if result.bytes != decomposition.materialized_bytes(list(result.rows_loaded), fallback_rows):
                            fail("byte_audit", record)
                        if result.fallback is FallbackMode.FULL and not torch.equal(result.fallback_logits, reference.logits):
                            fail("full_fallback_not_bitwise", record)
                        if result.fallback is FallbackMode.MASKED:
                            alive = result.trace.final_contenders
                            if not torch.equal(result.fallback_logits[alive], reference.logits[alive]):
                                fail("masked_fallback_not_bitwise", record)
                        key = f"{configuration.name}/{mode.value}"
                        if configuration.kind == "resident":
                            resident_results[mode] = result
                            expected = phase1c.get((prompt_id, mode.value))
                            mismatched = [f for f in PHASE1C_FIELDS if expected is None or record[f] != expected[f]]
                            validation["parity"][key] = mismatched
                            if mismatched:
                                fail("phase1c_reproduction", {"prompt_id": prompt_id, "configuration": key, "fields": mismatched})
                            if mode is FallbackMode.MASKED:
                                validation.update(validate(result, reference.logits, base_values, vector.to(torch.float64), numerics))
                                validation["masked_fallback_checked"] = result.fallback is FallbackMode.MASKED
                                validation["masked_guard_tripped"] = result.masked_guard_tripped
                        else:
                            differences = same_run(result, resident_results[mode])
                            validation["parity"][key] = differences
                            if differences:
                                fail("storage_parity", {"prompt_id": prompt_id, "configuration": key, "fields": differences})
                            requested = result.bytes["total"] - result.bytes["metadata"]
                            problems = storage_audit(configuration.tier, report, requested, cuda)
                            validation["storage_audit"][key] = problems
                            if problems:
                                fail("storage_audit", {"prompt_id": prompt_id, "configuration": key, "problems": problems})
                validation_log.write(validation)
                if any(validation["envelope_violations"].values()) or validation["coarse_arithmetic_violations"]:
                    fail("envelope_violation", validation)
                if (index + 1) % 100 == 0:
                    print(f"{index + 1}/{len(prompts)} prompts, {time.perf_counter() - started:.0f}s", flush=True)
        except HardFailure:
            pass

    for configuration in configurations:
        configuration.backend.store.close()
        if configuration.backend.streamer is not None:
            configuration.backend.streamer.close()
    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
    pack_digest = dict(pack_info)
    pack_digest.pop("open_and_verify_s")
    pack_digest.pop("built_now")
    digest = {
        "pack_sha256": canonical_digest([pack_digest]),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl")),
        "validation_sha256": canonical_digest(read_jsonl(output / "validation.jsonl")),
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl.gz"), EXCLUDED_FIELDS),
        "ranges_sha256": canonical_digest(read_jsonl(output / "ranges.jsonl.gz")),
        "excluded_fields": list(EXCLUDED_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - started:.0f}s; hard failures: {len(failures)}; output: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
