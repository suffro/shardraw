"""Phase 4A: the first real out-of-VRAM mixture of experts, reproduced bit for bit (decision 0007).

    uv run python benchmarks/olmoe_runtime.py --output experiments/phase4a/<name> [--num-prompts N]

The model, prompts, storage options, GPU budget and configurations are in
configs/phase4a-olmoe.yaml (OLMoE-1B-7B: 12.9 GB of experts, an 8 GB GPU). The run has two
stages, each in its own process, started by this script:

  reference  the model as published, loaded by transformers with every weight in host memory;
             the source files are re-hashed against the sha256 the Hub declares, and every row of
             the expert index is compared with the loader's own fused expert tensors. Then greedy
             decoding with the KV cache, each experts layer materialized whole on the device just
             before it runs (`FullLayerOffload`); the first prompts also run with every layer
             offloaded, and must agree bit for bit with the run that keeps some layers resident.
  stream     a fresh process, under the configured device-memory cap: every weight but the experts
             read from the checkpoint (direct I/O), the experts served from the drive through the
             compact experts call, under each configured cache. Every step is compared with the
             reference.

Per step and layer the stages record digests of the router logits, the routed experts and their
weights, every (token, expert) output before weighting (computed again on the same weights:
the direct expert-output check), the experts module's output, and per step the logits, token
and the whole KV cache. The stream stage also records what was requested, served by the cache,
read from the drive (physical bytes, reads, extents, 4 KiB blocks), moved to the device, the
compact buffers and peak device memory, and audits every step.

Writes, into the output directory:
  config.yaml, environment.json   as every benchmark (with the stages' load reports)
  prompts.jsonl        the prompts as this model's token ids
  index.json           the expert index's manifest, its verification and its audit
  reference.jsonl.gz   per prompt and step: the reference's digests (and the residency check)
  records.jsonl.gz     per configuration, prompt and step: comparisons, bytes, cache, memory, audit
  ranges.jsonl.gz      raw physical-I/O traces: every extent read, for the first prompts of each configuration
  digest.json          sha256 of index, prompts, reference, records and ranges (timings and system fields excluded)
A stage stops at its first hard failure (unless --keep-going) and saves it to failure.json: a
step that differs from the reference in any recorded digest, a residency check that differs,
an index row that differs from the loaded model, or an audit that does not hold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import importlib  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import load_model_without_experts  # noqa: E402
from awpmi.models.moe import ExpertCall, FullLayerOffload, StreamedExperts, find_expert_modules, groups_from_pack, move_except_experts, routed_experts  # noqa: E402
from awpmi.storage.cache import POLICIES, PageCache  # noqa: E402
from awpmi.storage.fileio import process_memory  # noqa: E402
from awpmi.storage.layout import row_bytes_of  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.storage.store import IO_BLOCK_BYTES  # noqa: E402
from awpmi.streaming.streamer import PageStreamer, resolve_device  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl, tensor_digest  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from moe_runtime import load_prompts, logits_sha256  # noqa: E402
from storage_runtime import describe_disk  # noqa: E402

EXCLUDED_FIELDS = ("timings_ms", "system")
LAYER_KINDS = ("router_logits", "top_k_index", "top_k_weights", "expert_outputs", "output")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HardFailure(Exception):
    pass


def short_digest(*tensors: torch.Tensor) -> str:
    return tensor_digest(*tensors)[:32]


def kv_digest(cache) -> str:
    return tensor_digest(*[t for layer in cache.layers for t in (layer.keys, layer.values)])


class StepRecorder:
    """One forward's observations per experts layer: its router's logits and its experts call."""

    def __init__(self, layers: list[str], num_experts: int) -> None:
        self.names = layers
        self.num_experts = num_experts
        self.layers: dict[str, dict] = {}
        self.extra = None  # called with each ExpertCall (memory accounting)

    def clear(self) -> None:
        self.layers = {name: {} for name in self.names}

    def router_hook(self, name: str):
        def hook(module, args, output):
            self.layers[name]["router_logits"] = short_digest(output[0])

        return hook

    def on_call(self, call: ExpertCall) -> None:
        entry = self.layers[call.module]
        entry["routed"] = routed_experts(call.top_k_index, self.num_experts).tolist()
        entry["top_k_index"] = short_digest(call.top_k_index)
        entry["top_k_weights"] = short_digest(call.top_k_weights)
        entry["output"] = short_digest(call.output)
        entry["expert_outputs"] = short_digest(call.per_assignment_outputs())
        if self.extra is not None:
            self.extra(call)

    def summary(self) -> dict:
        missing = [name for name in self.names if set(self.layers[name]) != {"routed", *LAYER_KINDS}]
        if missing:
            raise RuntimeError(f"incomplete observations for {missing[:3]}")
        return {
            "routed": [self.layers[name]["routed"] for name in self.names],
            "layers": {kind: [self.layers[name][kind] for name in self.names] for kind in LAYER_KINDS},
        }


@torch.inference_mode()
def decode(model, token_ids: list[int], steps: int, device: torch.device, before, after) -> None:
    """Greedy decoding with the KV cache; `before(step)` and `after(step, logits, positions, cache)` frame every forward."""
    input_ids = torch.tensor([token_ids], device=device)
    cache = None
    for step in range(steps + 1):
        before(step)
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1]
        cache = output.past_key_values
        after(step, logits, input_ids.shape[1], cache)
        input_ids = logits.argmax().view(1, 1)


def step_record(prompt: dict, step: int, positions: int, logits: torch.Tensor, cache, recorder: StepRecorder) -> dict:
    top2 = torch.topk(logits.float(), 2).values.tolist()
    return {
        "prompt_id": prompt["prompt_id"],
        "step": step,
        "phase": "prefill" if step == 0 else "decode",
        "positions": positions,
        "token": int(logits.argmax()),
        "logits_sha256": logits_sha256(logits),
        "top2_gap": top2[0] - top2[1],
        "kv_sha256": kv_digest(cache),
        **recorder.summary(),
    }


def compare(record: dict, expected: dict) -> dict:
    """Which recorded quantities equal the reference's, and the layers where a per-layer one differs."""
    matches = {
        "token": record["token"] == expected["token"],
        "logits": record["logits_sha256"] == expected["logits_sha256"],
        "kv_cache": record["kv_sha256"] == expected["kv_sha256"],
        "routed": record["routed"] == expected["routed"],
    }
    differing = {}
    for kind in LAYER_KINDS:
        layers = [k for k, (a, b) in enumerate(zip(record["layers"][kind], expected["layers"][kind], strict=True)) if a != b]
        matches[kind] = not layers
        if layers:
            differing[kind] = layers
    return {"matches": matches, "differing_layers": differing}


def load_adapter(model_config: dict):
    return importlib.import_module(f"awpmi.models.{model_config['adapter']}")


# Reference stage


def index_audit(model, pack) -> dict:
    """Every row of every index segment, read from the drive, against the loader's own expert tensors."""
    store = pack.store(direct=True)
    audit = {"segments": 0, "rows": 0, "bytes": 0, "differing": []}
    try:
        for entry in find_expert_modules(model):
            for parameter in entry.parameters:
                name = entry.segment(parameter)
                loaded = row_bytes_of(getattr(entry.module, parameter).detach().contiguous())
                for first in range(0, entry.num_experts, 8):
                    rows = torch.arange(first, min(first + 8, entry.num_experts))
                    if not torch.equal(store.read_rows(name, rows), loaded[rows]):
                        audit["differing"].append([name, first])
                audit["segments"] += 1
                audit["rows"] += entry.num_experts
                audit["bytes"] += loaded.numel()
    finally:
        store.close()
    return audit


def run_reference(raw_config: dict, output: Path, prompts: list[dict], keep_going: bool) -> int:
    model_config, reference_config = raw_config["model"], raw_config["reference"]
    adapter = load_adapter(model_config)
    profile = adapter.REFERENCE_PROFILE
    device = resolve_device("cuda")
    dtype = DTYPES[model_config["dtype"]]
    steps = int(raw_config["prompts"]["decode_steps"])
    report: dict = {"timings_ms": {}}
    failures: list[dict] = []

    def fail(kind: str, entry: dict) -> None:
        failures.append({"kind": kind, **entry})
        if not keep_going:
            raise HardFailure

    started = time.perf_counter()
    pack = open_pack(REPO_ROOT / raw_config["index"]["directory"], verify="files")
    report["timings_ms"]["verify_files"] = (time.perf_counter() - started) * 1e3
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_config["repository"], revision=model_config["revision"], dtype=dtype).eval()
    report["timings_ms"]["load"] = (time.perf_counter() - started) * 1e3
    profile.check_model(model)
    report["memory_after_load"] = process_memory()
    started = time.perf_counter()
    audit = index_audit(model, pack)
    report["timings_ms"]["index_audit"] = (time.perf_counter() - started) * 1e3
    index_info = {"manifest": pack.manifest, "verification": "files against the Hub's sha256", "audit": audit}
    (output / "index.json").write_text(json.dumps(index_info, indent=2), encoding="utf-8")
    try:
        if audit["differing"]:
            fail("index_audit", {"differing": audit["differing"][:10]})
        report["non_expert_device_bytes"] = move_except_experts(model, device)
        modules = find_expert_modules(model)
        recorder = StepRecorder([m.name for m in modules], modules[0].num_experts)
        for name, router in adapter.routers(model).items():
            router.register_forward_hook(recorder.router_hook(name))
        resident = tuple(f"{modules[i].name}" for i in reference_config["resident_layers"])
        check_count = int(reference_config["residency_check_prompts"])
        check: dict = {}
        with JsonlWriter(output / "reference.jsonl.gz") as log:
            for label, layers, pin, chosen in (
                ("all_offloaded", (), False, prompts[:check_count]),
                ("resident", resident, bool(reference_config["pin_host_experts"]), prompts),
            ):
                offload = FullLayerOffload(model, device, resident=layers, pin=pin, on_call=recorder.on_call).install()
                report[f"{label}_device_bytes"] = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for prompt in chosen:
                    timing = {}

                    def before(step):
                        recorder.clear()
                        torch.cuda.synchronize(device)
                        timing["start"] = time.perf_counter()

                    def after(step, logits, positions, cache, prompt=prompt, label=label):
                        record = step_record(prompt, step, positions, logits, cache, recorder)
                        torch.cuda.synchronize(device)
                        key = (prompt["prompt_id"], step)
                        if label == "all_offloaded":
                            check[key] = record
                            return
                        record["timings_ms"] = {"step": (time.perf_counter() - timing["start"]) * 1e3}
                        if key in check:
                            verdict = compare(record, check[key])
                            record["residency_check"] = all(verdict["matches"].values())
                            if not record["residency_check"]:
                                fail("residency", {"prompt_id": prompt["prompt_id"], "step": step, **verdict})
                        log.write(record)

                    decode(model, prompt["token_ids"], steps, device, before, after)
                report["timings_ms"][label] = (time.perf_counter() - started) * 1e3
                report[f"{label}_peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
                report[f"{label}_copied_bytes"] = offload.copied_bytes
                offload.remove()
                del offload
                gc.collect()
                torch.cuda.empty_cache()
                print(f"reference {label}: {len(chosen)} prompts, {report['timings_ms'][label] / 1e3:.0f}s", flush=True)
        report["residency_checked_steps"] = len(check)
    except HardFailure:
        pass
    report["memory_peak"] = process_memory()
    report["profile"] = profile.to_json()
    report["experts_implementation"] = model.config._experts_implementation
    report["attention_implementation"] = model.config._attn_implementation
    (output / "reference_stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 1 if failures else 0


# Stream stage


def audit_step(report: dict, requested: int, cuda: bool) -> list[str]:
    problems = []
    served, storage, transfer = report["materialization"], report["storage"], report.get("transfer", {})
    if served["requested_bytes"] != requested:
        problems.append("requested bytes != served experts")
    if served["cache_hit_bytes"] + served["fetched_bytes"] != served["requested_bytes"]:
        problems.append("cache hits + fetched != requested")
    if storage["logical_bytes"] != served["fetched_bytes"]:
        problems.append("storage logical != fetched")
    if cuda and transfer.get("h2d_bytes") != storage["logical_bytes"]:
        problems.append("h2d != storage logical")
    if not 0 <= storage["blocks_4k"] * IO_BLOCK_BYTES - storage["physical_bytes"] < IO_BLOCK_BYTES * max(1, storage["requests"]):
        problems.append("physical bytes are not the requested rows' 4 KiB blocks")
    if storage["os_read_bytes"] is not None and (
        storage["os_read_bytes"] != storage["physical_bytes"] or storage["os_read_calls"] != storage["read_calls"]
    ):
        problems.append("OS counters differ from the store's reads")
    return problems


def run_stream(raw_config: dict, output: Path, prompts: list[dict], keep_going: bool) -> int:
    model_config, storage = raw_config["model"], raw_config["storage"]
    adapter = load_adapter(model_config)
    profile = adapter.REFERENCE_PROFILE
    device = resolve_device("cuda")
    dtype = DTYPES[model_config["dtype"]]
    steps = int(raw_config["prompts"]["decode_steps"])
    time.sleep(float(raw_config["timing"]["settle_seconds"]))
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(raw_config["gpu_budget_bytes"])
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    report: dict = {"gpu_total_bytes": total, "gpu_budget_bytes": budget, "timings_ms": {}, "configurations": {}}
    failures: list[dict] = []

    def fail(kind: str, entry: dict) -> None:
        failures.append({"kind": kind, **entry})
        if not keep_going:
            raise HardFailure

    pack = open_pack(REPO_ROOT / raw_config["index"]["directory"], verify="size")
    groups = groups_from_pack(pack)
    profile.check_weights({name: segment.dtype for name, segment in pack.segments.items()})
    started = time.perf_counter()
    model, load_report = load_model_without_experts(model_config["repository"], model_config["revision"], dtype, device, pack.files)
    report["timings_ms"]["load"] = (time.perf_counter() - started) * 1e3
    profile.check_model(model)
    report["load"] = {k: v for k, v in load_report.items() if k != "expert_parameters"}
    report["non_expert_device_bytes"] = torch.cuda.memory_allocated(device)
    report["memory_after_load"] = process_memory()
    modules = find_expert_modules(model)
    num_experts = modules[0].num_experts
    expert_bytes = sum(segment.nbytes for segment in pack.segments.values())
    row_bytes = {key: sum(pack.segments[s].row_bytes for s in group.segments.values()) for key, group in groups.items()}
    layer_bytes = max(row_bytes[key] * groups[key].experts for key in groups)
    recorder = StepRecorder([m.name for m in modules], num_experts)
    for name, router in adapter.routers(model).items():
        router.register_forward_hook(recorder.router_hook(name))
    reference = {(r["prompt_id"], r["step"]): r for r in read_jsonl(output / "reference.jsonl.gz")}

    def build(cache: PageCache | None) -> MaterializationBackend:
        store = pack.store(
            direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
            max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
        )
        return MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])), cache)

    # Warm-up (kernels, allocator) with a backend of its own, so that no configuration's cache sees it.
    backend = build(None)
    streamed = StreamedExperts(model, ExpertStore(WeightStore(backend), groups), compact=True, on_call=recorder.on_call).install()
    for prompt in prompts[: int(raw_config["timing"]["warmup_prompts"])]:
        decode(model, prompt["token_ids"], 1, device, lambda step: recorder.clear(), lambda *args: None)
    streamed.remove()
    backend.store.close()
    backend.streamer.close()

    started_all = time.perf_counter()
    with JsonlWriter(output / "records.jsonl.gz") as log, JsonlWriter(output / "ranges.jsonl.gz") as range_log:
        try:
            for index, entry in enumerate(raw_config["configurations"]):
                name = entry["name"]
                capacity = int(float(entry.get("cache_fraction", 0.0)) * expert_bytes)
                cache = None
                if capacity:
                    policy_class = POLICIES[entry["policy"]]
                    policy = policy_class(float(raw_config["hotness_half_life"])) if entry["policy"] == "hotness" else policy_class()
                    cache = PageCache(capacity, policy)
                backend = build(cache)
                all_experts = bool(entry.get("all_experts", False))
                streamed = StreamedExperts(
                    model, ExpertStore(WeightStore(backend), groups), compact=True, all_experts=all_experts, on_call=recorder.on_call
                ).install()
                peaks = {"expert": 0}

                def account(call: ExpertCall, cache=cache, streamed=streamed, peaks=peaks) -> None:
                    peaks["expert"] = max(peaks["expert"], streamed.compact_bytes + (0 if cache is None else cache.resident_bytes))

                recorder.extra = account
                gc.collect()
                torch.cuda.empty_cache()
                report["configurations"][name] = {
                    "cache_capacity_bytes": capacity,
                    "device_bytes_at_start": torch.cuda.memory_allocated(device),
                    "host_bytes": backend.host_resident_bytes,
                }
                count = int(entry.get("num_prompts", len(prompts)))
                config_steps = int(entry.get("decode_steps", steps))
                started = time.perf_counter()
                for position, prompt in enumerate(prompts[:count]):
                    streamed.poison = index == 0 and position < int(raw_config["poison_prompts"])
                    record_ranges = position < int(raw_config["record_ranges_prompts"])
                    timing = {}

                    def before(step, record_ranges=record_ranges):
                        recorder.clear()
                        backend.reset_stats(record_ranges=record_ranges)
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        streamed.reset_peak()
                        peaks["expert"] = 0 if cache is None else cache.resident_bytes
                        timing["start"] = time.perf_counter()

                    def after(step, logits, positions, cache_kv, prompt=prompt, name=name, cache=cache, streamed=streamed):
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - timing["start"]) * 1e3
                        peak_device = torch.cuda.max_memory_allocated(device)
                        peak_reserved = torch.cuda.max_memory_reserved(device)
                        record = step_record(prompt, step, positions, logits, cache_kv, recorder)
                        expected = reference[(prompt["prompt_id"], step)]
                        verdict = compare(record, expected)
                        report_ = backend.report()
                        served_counts = [num_experts if all_experts else len(r) for r in record["routed"]]
                        requested = sum(n * row_bytes[module] for n, module in zip(served_counts, recorder.names))
                        problems = audit_step(report_, requested, True)
                        spare = 2 if streamed.poison else 0
                        expected_compact = max((n + spare) * row_bytes[module] for n, module in zip(served_counts, recorder.names))
                        if streamed.peak_compact_bytes != expected_compact:
                            problems.append("compact buffers != the served experts")
                        storage_stats, served = report_["storage"], report_["materialization"]
                        result = {
                            "configuration": name,
                            "prompt_id": prompt["prompt_id"],
                            "step": step,
                            "phase": record["phase"],
                            "positions": positions,
                            "poisoned": streamed.poison,
                            "token": record["token"],
                            "logits_sha256": record["logits_sha256"],
                            **verdict,
                            "routed_per_layer": [len(r) for r in record["routed"]],
                            "served_per_layer": served_counts,
                            "requested_bytes": served["requested_bytes"],
                            "cache_hit_bytes": served["cache_hit_bytes"],
                            "fetched_bytes": served["fetched_bytes"],
                            "device_copy_bytes": served["device_copy_bytes"],
                            "storage": {k: storage_stats[k] for k in ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k")},
                            "h2d_bytes": report_["transfer"]["h2d_bytes"],
                            "cache": None if cache is None else {k: report_["cache"][k] for k in ("hits", "misses", "hit_bytes", "miss_bytes", "inserts", "evictions", "bypassed")},
                            "cache_resident_bytes": None if cache is None else cache.resident_bytes,
                            "compact_peak_bytes": streamed.peak_compact_bytes,
                            "expert_peak_bytes": peaks["expert"],
                            "requested_fraction": served["requested_bytes"] / expert_bytes,
                            "drive_fraction": storage_stats["physical_bytes"] / expert_bytes,
                            "h2d_fraction": report_["transfer"]["h2d_bytes"] / expert_bytes,
                            "compact_fraction_of_layer": streamed.peak_compact_bytes / layer_bytes,
                            "audit": problems,
                            "timings_ms": {"step": elapsed, "io": storage_stats["io_ms"], "copy": report_["transfer"]["copy_ms"]},
                            "system": {
                                "os_read_calls": storage_stats["os_read_calls"],
                                "os_read_bytes": storage_stats["os_read_bytes"],
                                "peak_device_bytes": peak_device,
                                "peak_reserved_bytes": peak_reserved,
                                "process_resident_bytes": (process_memory() or {}).get("resident_bytes"),
                            },
                        }
                        log.write(result)
                        if record_ranges:
                            range_log.write({"configuration": name, "prompt_id": prompt["prompt_id"], "step": step, "ranges": storage_stats.get("ranges", [])})
                        if not all(verdict["matches"].values()):
                            fail("mismatch", {k: result[k] for k in ("configuration", "prompt_id", "step", "matches", "differing_layers")})
                        if problems:
                            fail("audit", {"configuration": name, "prompt_id": prompt["prompt_id"], "step": step, "problems": problems})

                    decode(model, prompt["token_ids"], config_steps, device, before, after)
                elapsed = time.perf_counter() - started
                report["timings_ms"][name] = elapsed * 1e3
                report["configurations"][name]["peak_cache_resident_bytes"] = None if cache is None else cache.peak_resident_bytes
                print(f"stream {name}: {min(count, len(prompts))} prompts, {elapsed:.0f}s", flush=True)
                streamed.remove()
                backend.store.close()
                backend.streamer.close()
                recorder.extra = None
        except HardFailure:
            pass
    report["timings_ms"]["configurations"] = (time.perf_counter() - started_all) * 1e3
    report["memory_peak"] = process_memory()
    report["expert_bytes_total"] = expert_bytes
    report["layer_bytes"] = layer_bytes
    report["expert_row_bytes"] = sorted(set(row_bytes.values()))
    report["experts_implementation"] = model.config._experts_implementation
    report["attention_implementation"] = model.config._attn_implementation
    (output / "stream_stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        path = output / "failure.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps(existing + failures, indent=2), encoding="utf-8")
    return 1 if failures else 0


# Driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4a-olmoe.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    parser.add_argument("--stage", choices=["all", "reference", "stream"], default="all")
    args = parser.parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    if args.stage != "all":
        prompts = read_jsonl(output / "prompts.jsonl")
        runner = run_reference if args.stage == "reference" else run_stream
        return runner(raw_config, output, prompts, args.keep_going)

    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    model_config, prompt_config = raw_config["model"], dict(raw_config["prompts"])
    if args.num_prompts is not None:
        prompt_config["num_prompts"] = args.num_prompts
    tokenizer = AutoTokenizer.from_pretrained(model_config["repository"], revision=model_config["revision"])
    prompts = load_prompts(prompt_config, tokenizer)
    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    with JsonlWriter(output / "prompts.jsonl") as log:
        for prompt in prompts:
            log.write(prompt)
    environment = environment_metadata(
        REPO_ROOT,
        {"repository": model_config["repository"], "revision": model_config["revision"], "dtype": model_config["dtype"], "device": "cuda"},
        NUMERICS_FLAGS,
    )
    environment["num_prompts"] = len(prompts)
    environment["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    environment["profile"] = load_adapter(model_config).REFERENCE_PROFILE.to_json()
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    index_manifest = json.loads((REPO_ROOT / raw_config["index"]["directory"] / "manifest.json").read_text(encoding="utf-8"))
    first_file = open_pack(REPO_ROOT / raw_config["index"]["directory"], verify="size").files[next(iter(index_manifest["files"]))]
    environment["storage"] = {"disk": describe_disk(first_file)}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    command = [sys.executable, str(Path(__file__).resolve()), "--config", args.config, "--output", str(output)]
    if args.keep_going:
        command.append("--keep-going")
    codes = {}
    for stage in ("reference", "stream"):
        started = time.perf_counter()
        codes[stage] = subprocess.run([*command, "--stage", stage], check=False).returncode
        print(f"stage {stage}: exit {codes[stage]} after {time.perf_counter() - started:.0f}s", flush=True)
        if codes[stage] and not args.keep_going:
            break
    index_info = json.loads((output / "index.json").read_text(encoding="utf-8")) if (output / "index.json").exists() else {}
    digest = {
        "index_sha256": canonical_digest([index_info]),
        "prompts_sha256": canonical_digest(read_jsonl(output / "prompts.jsonl")),
        "reference_sha256": canonical_digest(read_jsonl(output / "reference.jsonl.gz"), EXCLUDED_FIELDS) if (output / "reference.jsonl.gz").exists() else None,
        "records_sha256": canonical_digest(read_jsonl(output / "records.jsonl.gz"), EXCLUDED_FIELDS) if (output / "records.jsonl.gz").exists() else None,
        "ranges_sha256": canonical_digest(read_jsonl(output / "ranges.jsonl.gz")) if (output / "ranges.jsonl.gz").exists() else None,
        "excluded_fields": list(EXCLUDED_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"done: {codes}; output: {output}")
    return 1 if any(codes.values()) or len(codes) < 2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
