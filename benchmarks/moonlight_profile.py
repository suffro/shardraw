"""Phase 4B profile: where a streamed Moonlight step spends its time, prefill and decode (decision 0008).

    uv run python benchmarks/moonlight_profile.py --output experiments/phase4b/<run>/profile-<configuration>.json
        --configuration stream [--prompts 0 7] [--decode-steps 8] [--trace]

One configuration per process (a cache must start empty, and a process ends with all its memory returned). The
correctness benchmark (`moonlight_runtime.py`) hashes every intermediate inside the timed steps; this script runs the
same configuration recording nothing but time. Host wall time per step is split with timers around the generic path
(no model code changes; nested regions are not counted twice):

  route    the experts pre-hook waiting for the router's choice (the GPU finishing attention and router, then the
           device-to-host sync of the routed experts)
  cache    cache lookups, and the device copies of cached rows ("hits first")
  admit    cache admission: copies of fetched rows into the cache's pool, and evictions
  plan     read planning (runs, alignment, extents, pieces)
  io       positioned reads in flight (8 threads, direct I/O; the store's own clock): the drive
  gather   host gathers in pinned staging
  h2d      host time issuing the copies (their device time is reported separately)
  assemble the rest of expert assembly (buffers, bookkeeping)
  chunks   the chunked grouped GEMM's own host work (decision 0008), its reads excluded
  other    everything else: Python and kernel launches of attention, norms, routers, the experts' and shared experts'
           compute and the LM head, and the GPU work the host waits for at the end of the step

With --trace, a torch.profiler trace of the prefill and two decode steps of the first prompt gives device time by
kernel kind (GEMM, attention, other kernels, host-to-device and device-to-device copies) and by module scope
(attention, router, experts, shared experts, LM head), the number of kernel launches and their CPU time, and the GPU's
busy time (union of its activities) against the step's wall time. Timings only: nothing here is digested.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import DTYPES, load_adapter  # noqa: E402  (configures the numerics first)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

import awpmi.materialization.backend as backend_module  # noqa: E402
import awpmi.models.moe as moe_module  # noqa: E402
import awpmi.storage.cache as cache_module  # noqa: E402
import awpmi.storage.store as store_module  # noqa: E402
import awpmi.streaming.streamer as streamer_module  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import checkpoint_sources, load_model_without_experts  # noqa: E402
from awpmi.models.moe import StreamedExperts, groups_from_pack  # noqa: E402
from awpmi.storage.cache import POLICIES, PageCache  # noqa: E402
from awpmi.storage.fileio import process_memory  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402


class Clock:
    """Accumulated host seconds per region, with nesting: an inner region's time is not its outer one's."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self._stack: list[list] = []

    def wrap(self, region: str, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            self._stack.append([region, 0.0])
            try:
                return function(*args, **kwargs)
            finally:
                _, inner = self._stack.pop()
                elapsed = time.perf_counter() - started
                self.totals[region] += elapsed - inner
                if self._stack:
                    self._stack[-1][1] += elapsed

        return timed

    def reset(self) -> dict[str, float]:
        totals = dict(self.totals)
        self.totals.clear()
        return totals


def instrument(clock: Clock) -> None:
    """Timers around the generic path's stages (module functions and methods, wrapped in place)."""
    moe_module.routed_experts = clock.wrap("route", moe_module.routed_experts)
    moe_module._chunked_grouped_mm = clock.wrap("chunks", moe_module._chunked_grouped_mm)
    store_module.plan_reads = clock.wrap("plan", store_module.plan_reads)
    streamer_module.PageStreamer._pieces = clock.wrap("plan", streamer_module.PageStreamer._pieces)
    store_module.FileBackedPageStore.read_extents = clock.wrap("io", store_module.FileBackedPageStore.read_extents)
    streamer_module.gather_runs = clock.wrap("gather", streamer_module.gather_runs)
    streamer_module.PageStreamer._copy = clock.wrap("h2d", streamer_module.PageStreamer._copy)
    backend_module.PageCache.get_many = clock.wrap("cache", backend_module.PageCache.get_many)
    backend_module.MaterializationBackend._cache_copy = clock.wrap("admit", backend_module.MaterializationBackend._cache_copy)
    cache_module.PageCache.put = clock.wrap("admit", cache_module.PageCache.put)
    ExpertStore.assemble = clock.wrap("assemble", ExpertStore.assemble)


def scopes(model, adapter, experts: list[str]) -> list:
    """record_function scopes around the modules whose device time the trace attributes."""
    targets = {"lm_head": model.lm_head}
    for name, module in model.named_modules():
        if name.endswith(".self_attn"):
            targets[name] = module
    for name in experts:
        head = name[: -len(adapter.EXPERTS_SUFFIX)]
        targets[name] = model.get_submodule(name)
        targets[head + adapter.ROUTER_SUFFIX] = model.get_submodule(head + adapter.ROUTER_SUFFIX)
        targets[head + adapter.SHARED_SUFFIX] = model.get_submodule(head + adapter.SHARED_SUFFIX)

    def kind(name: str) -> str:
        if name == "lm_head":
            return "scope:lm_head"
        if name.endswith(".self_attn"):
            return "scope:attention"
        if name.endswith(adapter.EXPERTS_SUFFIX):
            return "scope:experts"
        if name.endswith(adapter.ROUTER_SUFFIX):
            return "scope:router"
        return "scope:shared_experts"

    handles, stack = [], []
    for name, module in targets.items():
        label = kind(name)

        def enter(module_, args, label=label):
            scope = record_function(label)
            scope.__enter__()
            stack.append(scope)

        def leave(module_, args, output):
            stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave, always_call=True))
    return handles


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if "memcpy htod" in lowered or "memcpy h2d" in lowered:
        return "memcpy_h2d"
    if "memcpy dtod" in lowered or "memcpy d2d" in lowered:
        return "memcpy_d2d"
    if "memcpy" in lowered or "memset" in lowered:
        return "memcpy_other"
    if any(k in lowered for k in ("flash", "fmha", "attention", "sdpa")):
        return "attention_kernels"
    if any(k in lowered for k in ("gemm", "cutlass", "gemv", "sm80_xmma", "sm89", "ampere", "cublas", "s16816")):
        return "gemm"
    return "other_kernels"


def summarize_trace(prof, wall_ms: float) -> dict:
    """Device time by kind and by module scope, launches, and the GPU's busy time.

    The scopes' own GPU-side annotations (ranges from a scope's first to last kernel, gaps included) are not activities:
    they only attribute each kernel or copy, by its start time, to the scope whose range holds it.
    """
    import bisect

    kinds: dict[str, float] = defaultdict(float)
    scopes_ms: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    annotations, activities = [], []
    launches = 0
    launch_cpu_ms = 0.0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            if event.name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaLaunchKernelEx"):
                launches += 1
                launch_cpu_ms += event.self_cpu_time_total / 1e3
            continue
        span = (event.time_range.start, event.time_range.end)
        if event.name.startswith("scope:"):
            annotations.append((*span, event.name[6:]))
        else:
            activities.append((*span, kernel_kind(event.name)))
    annotations.sort()
    starts = [a[0] for a in annotations]
    intervals = []
    for start, end, kind in activities:
        kinds[kind] += (end - start) / 1e3
        intervals.append((start, end))
        index = bisect.bisect_right(starts, start) - 1
        label = annotations[index][2] if index >= 0 and start < annotations[index][1] else "outside_scopes"
        scopes_ms[label][kind] += (end - start) / 1e3
    intervals.sort()
    busy, end = 0.0, None
    for start, stop in intervals:
        if end is None or start > end:
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    busy_ms = busy / 1e3
    return {
        "device_ms_by_kind": dict(kinds),
        "device_ms_by_scope": {label: dict(values) for label, values in scopes_ms.items()},
        "kernel_launches": launches,
        "launch_cpu_ms": launch_cpu_ms,
        "gpu_busy_ms": busy_ms,
        "wall_ms": wall_ms,
        "gpu_idle_fraction": max(0.0, 1.0 - busy_ms / wall_ms) if wall_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a moonlight_runtime run directory (its config and prompts are used)")
    parser.add_argument("--output", required=True, help="the profile JSON file to write")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--prompts", type=int, nargs="+", default=[0, 3, 5, 7], help="indices into the run's prompts")
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    raw = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    model_config, storage = raw["model"], raw["storage"]
    adapter = load_adapter(model_config)
    device = torch.device("cuda", torch.cuda.current_device())
    time.sleep(float(raw["timing"]["settle_seconds"]))
    torch.cuda.set_per_process_memory_fraction(int(raw["gpu_budget_bytes"]) / torch.cuda.get_device_properties(device).total_memory, device)
    all_prompts = read_jsonl(run / "prompts.jsonl")
    prompts = [all_prompts[i] for i in args.prompts]
    steps = int(args.decode_steps if args.decode_steps is not None else raw["prompts"]["decode_steps"])
    pack = open_pack(REPO_ROOT / raw["index"]["directory"], verify="size")
    groups = groups_from_pack(pack)
    files = {key: entry.path for key, entry in checkpoint_sources(model_config["repository"], model_config["revision"], declared_sha256=False).items()}
    model, _ = load_model_without_experts(model_config["repository"], model_config["revision"], DTYPES[model_config["dtype"]], device, files)
    expert_row = max(sum(pack.segments[s].row_bytes for s in group.segments.values()) for group in groups.values())
    entry = {e["name"]: e for e in raw["configurations"]}[args.configuration]
    capacity = int(entry.get("cache_experts", 0)) * expert_row
    cache = None
    if capacity:
        policy_class = POLICIES[entry["policy"]]
        cache = PageCache(capacity, policy_class(float(raw["hotness_half_life"])) if entry["policy"] == "hotness" else policy_class())
    call_budget = int(entry["call_budget_experts"]) * expert_row if "call_budget_experts" in entry else entry.get("call_budget_bytes", raw.get("call_budget_bytes"))
    store = pack.store(
        direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
        max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
    )
    backend = MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])), cache)
    clock = Clock()
    instrument(clock)
    experts = ExpertStore(WeightStore(backend), groups)
    streamed = StreamedExperts(model, experts, compact=True, all_experts=bool(entry.get("all_experts", False)), max_call_bytes=call_budget).install()
    names = [m.name for m in streamed.modules]
    result = {"gpu": torch.cuda.get_device_name(device), "configuration": args.configuration, "prompts": [p["prompt_id"] for p in prompts],
              "lengths": [p["length"] for p in prompts], "decode_steps": steps, "call_budget_bytes": call_budget, "cache_capacity_bytes": capacity,
              "memory_after_load": process_memory()}
    records: list[dict] = []
    traces: list[dict] = []
    # Warm-up: one short prefill and a decode step (kernels, allocator), not recorded.
    with torch.inference_mode():
        warm = model(input_ids=torch.tensor([all_prompts[0]["token_ids"][:16]], device=device), use_cache=True, logits_to_keep=1)
        model(input_ids=warm.logits[0, -1].argmax().view(1, 1), past_key_values=warm.past_key_values, use_cache=True, logits_to_keep=1)
    del warm
    if cache is not None:  # the warm-up must not leave anything in the measured cache
        cache.clear()
        cache.stats.reset()
    gc.collect()
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            input_ids = torch.tensor([prompt["token_ids"]], device=device)
            cache_kv = None
            for step in range(steps + 1):
                trace = args.trace and index == 0 and step <= 2
                handles = scopes(model, adapter, names) if trace else []
                torch.cuda.synchronize(device)
                backend.reset_stats()
                clock.reset()
                calls = (streamed.calls, streamed.chunked_calls)
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if trace else None
                if profiler is not None:
                    profiler.__enter__()
                started = time.perf_counter()
                output = model(input_ids=input_ids, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
                token = output.logits[0, -1].argmax().view(1, 1)
                torch.cuda.synchronize(device)
                wall = (time.perf_counter() - started) * 1e3
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    traces.append({"step": step, "phase": "prefill" if step == 0 else "decode", **summarize_trace(profiler, wall)})
                for handle in handles:
                    handle.remove()
                regions = {k: v * 1e3 for k, v in clock.reset().items()}
                report = backend.report()
                records.append({
                    "prompt": prompt["prompt_id"], "length": prompt["length"], "step": step, "phase": "prefill" if step == 0 else "decode",
                    "wall_ms": wall, "regions_ms": regions, "store_io_ms": report["storage"]["io_ms"], "h2d_device_ms": report["transfer"]["copy_ms"],
                    "physical_bytes": report["storage"]["physical_bytes"], "h2d_bytes": report["transfer"]["h2d_bytes"],
                    "read_calls": report["storage"]["read_calls"], "pieces": report["transfer"]["pieces"], "h2d_copies": report["transfer"]["h2d_copies"],
                    "requests": report["materialization"]["requests"], "chunked_calls": streamed.chunked_calls - calls[1],
                    "cache_hits": None if cache is None else report["cache"]["hits"], "traced": trace,
                })
                cache_kv = output.past_key_values
                input_ids = token
    streamed.remove()
    store.close()
    backend.streamer.close()
    result["summary"] = summarize(records)
    result["traces"] = traces
    result["steps"] = records
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "traces": traces}, indent=1), flush=True)
    return 0


def summarize(steps: list[dict]) -> dict:
    out = {}
    for phase in ("prefill", "decode"):
        chosen = [s for s in steps if s["phase"] == phase and not s["traced"]]
        if not chosen:
            continue
        regions = sorted({k for s in chosen for k in s["regions_ms"]})
        mean = lambda key: statistics.mean(s[key] for s in chosen)  # noqa: E731
        entry = {
            "steps": len(chosen),
            "wall_ms": mean("wall_ms"),
            "regions_ms": {k: statistics.mean(s["regions_ms"].get(k, 0.0) for s in chosen) for k in regions},
            "h2d_device_ms": mean("h2d_device_ms"),
            "physical_mb": mean("physical_bytes") / 1e6,
            "read_calls": mean("read_calls"),
            "pieces": mean("pieces"),
            "h2d_copies": mean("h2d_copies"),
            "requests": mean("requests"),
            "chunked_calls": mean("chunked_calls"),
        }
        accounted = sum(entry["regions_ms"].values())
        entry["regions_ms"]["other"] = entry["wall_ms"] - accounted
        entry["share"] = {k: v / entry["wall_ms"] for k, v in entry["regions_ms"].items()}
        entry["drive_gb_per_s"] = entry["physical_mb"] / 1e3 / max(1e-9, entry["regions_ms"].get("io", 0.0) / 1e3)
        if phase == "prefill":
            entry["by_length"] = {
                int(length): statistics.mean(s["wall_ms"] for s in chosen if s["length"] == length) for length in sorted({s["length"] for s in chosen})
            }
        out[phase] = entry
    return out


if __name__ == "__main__":
    raise SystemExit(main())
