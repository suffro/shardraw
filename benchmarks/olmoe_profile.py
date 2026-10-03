"""Phase 4A profile: where a streamed OLMoE decode step spends its time (decision 0007).

    uv run python benchmarks/olmoe_profile.py --output experiments/phase4a/<name>/profile.json
        [--configurations stream lru-25] [--prompts 4] [--trace-steps 2]

The correctness benchmark (`olmoe_runtime.py`) hashes every intermediate inside the timed steps;
this script runs the same configurations with nothing recorded but time. For every decode step
it splits the host's wall time with timers around the generic path (no model code is changed):

  route      the experts pre-hook waiting for the router's choice (the GPU finishing the layer's
             attention and router, then the device-to-host sync of the routed experts)
  cache      cache lookups and the device copies of cached experts ("hits first")
  plan       read planning (runs, alignment, extents, pieces)
  io         positioned reads in flight (8 threads, direct I/O; the store's own clock)
  gather     host gathers of requested bytes in pinned staging
  h2d        host time issuing copies; the copies' device time is reported separately
  admit      copies of fetched experts into the cache
  other      everything else: Python and kernel launches of attention, norms, the experts call
             and the LM head, and the GPU work the host waits for at the end of the step

and, from CUDA events, the device time of the host-to-device copies and of the experts calls
(from the moment their weights are complete to their output). A torch.profiler trace of the
first `--trace-steps` decode steps gives kernel time by kind and the number of launches.
Timings only: nothing here is digested.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import awpmi.materialization.backend as backend_module  # noqa: E402
import awpmi.models.moe as moe_module  # noqa: E402
import awpmi.storage.store as store_module  # noqa: E402
import awpmi.streaming.streamer as streamer_module  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import load_model_without_experts  # noqa: E402
from awpmi.models.moe import StreamedExperts, groups_from_pack  # noqa: E402
from awpmi.storage.cache import POLICIES, PageCache  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer, resolve_device  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from moe_runtime import load_prompts  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


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
    moe_module.routed_experts = clock.wrap("route", moe_module.routed_experts)  # torch.unique: waits for the router
    store_module.plan_reads = clock.wrap("plan", store_module.plan_reads)
    streamer_module.PageStreamer._pieces = clock.wrap("plan", streamer_module.PageStreamer._pieces)
    store_module.FileBackedPageStore.read_extents = clock.wrap("io", store_module.FileBackedPageStore.read_extents)
    streamer_module.gather_runs = clock.wrap("gather", streamer_module.gather_runs)
    streamer_module.PageStreamer._copy = clock.wrap("h2d", streamer_module.PageStreamer._copy)
    backend_module.PageCache.get_many = clock.wrap("cache", backend_module.PageCache.get_many)
    backend_module.MaterializationBackend._cache_copy = clock.wrap("admit", backend_module.MaterializationBackend._cache_copy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4a-olmoe.yaml"))
    parser.add_argument("--output", required=True, help="the profile JSON file to write")
    parser.add_argument("--configurations", nargs="+", default=["stream", "lru-25"])
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--trace-steps", type=int, default=2)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config, storage = raw["model"], raw["storage"]
    device = resolve_device("cuda")
    time.sleep(float(raw["timing"]["settle_seconds"]))
    torch.cuda.set_per_process_memory_fraction(int(raw["gpu_budget_bytes"]) / torch.cuda.get_device_properties(device).total_memory, device)
    tokenizer = AutoTokenizer.from_pretrained(model_config["repository"], revision=model_config["revision"])
    prompts = load_prompts({**raw["prompts"], "num_prompts": args.prompts}, tokenizer)
    pack = open_pack(REPO_ROOT / raw["index"]["directory"], verify="size")
    groups = groups_from_pack(pack)
    model, _ = load_model_without_experts(model_config["repository"], model_config["revision"], DTYPES[model_config["dtype"]], device, pack.files)
    expert_bytes = sum(segment.nbytes for segment in pack.segments.values())
    clock = Clock()
    instrument(clock)
    entries = {entry["name"]: entry for entry in raw["configurations"]}
    result = {"gpu": torch.cuda.get_device_name(device), "prompts": args.prompts, "configurations": {}}
    for name in args.configurations:
        entry = entries[name]
        capacity = int(float(entry.get("cache_fraction", 0.0)) * expert_bytes)
        cache = None
        if capacity:
            policy_class = POLICIES[entry["policy"]]
            cache = PageCache(capacity, policy_class(float(raw["hotness_half_life"])) if entry["policy"] == "hotness" else policy_class())
        store = pack.store(
            direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
            max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
        )
        backend = MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])), cache)
        experts = ExpertStore(WeightStore(backend), groups)
        experts.assemble = clock.wrap("assemble", experts.assemble)
        streamed = StreamedExperts(model, experts, compact=True, all_experts=bool(entry.get("all_experts", False))).install()
        events: list = []
        for module in streamed.modules:
            # Device time of each experts call: from its weights being complete (pre-hook end) to its output.
            def pre(module_, args, kwargs, holder=None):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                events.append([start, None])

            def post(module_, args, kwargs, output):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                events[-1][1] = end

            module.module.register_forward_pre_hook(pre, with_kwargs=True)  # after StreamedExperts' own pre-hook
            module.module.register_forward_hook(post, with_kwargs=True)
        steps: list[dict] = []
        with torch.inference_mode():
            for index, prompt in enumerate(prompts):
                input_ids = torch.tensor([prompt["token_ids"]], device=device)
                cache_kv = None
                for step in range(int(raw["prompts"]["decode_steps"]) + 1):
                    trace = index == 0 and 1 <= step <= args.trace_steps
                    torch.cuda.synchronize(device)
                    backend.reset_stats()
                    clock.reset()
                    events.clear()
                    started = time.perf_counter()
                    profiler = None
                    if trace:
                        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                        profiler.__enter__()
                    output = model(input_ids=input_ids, past_key_values=cache_kv, use_cache=True)
                    logits = output.logits[0, -1]
                    token = logits.argmax().view(1, 1)
                    torch.cuda.synchronize(device)
                    wall = (time.perf_counter() - started) * 1e3
                    if profiler is not None:
                        profiler.__exit__(None, None, None)
                    regions = {k: v * 1e3 for k, v in clock.reset().items()}
                    report = backend.report()
                    record = {
                        "prompt": index,
                        "step": step,
                        "phase": "prefill" if step == 0 else "decode",
                        "wall_ms": wall,
                        "regions_ms": regions,
                        "store_io_ms": report["storage"]["io_ms"],
                        "h2d_device_ms": report["transfer"]["copy_ms"],
                        "experts_device_ms": sum(s.elapsed_time(e) for s, e in events if e is not None),
                        "physical_bytes": report["storage"]["physical_bytes"],
                        "h2d_bytes": report["transfer"]["h2d_bytes"],
                        "read_calls": report["storage"]["read_calls"],
                        "pieces": report["transfer"]["pieces"],
                        "h2d_copies": report["transfer"]["h2d_copies"],
                        "traced": trace,
                    }
                    if profiler is not None:
                        record["trace"] = summarize_trace(profiler)
                    steps.append(record)
                    cache_kv = output.past_key_values
                    input_ids = token
        streamed.remove()
        store.close()
        backend.streamer.close()
        result["configurations"][name] = summarize(steps)
        result["configurations"][name]["steps"] = steps
        print(json.dumps({name: {k: v for k, v in result["configurations"][name].items() if k != "steps"}}, indent=1), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def summarize_trace(profiler) -> dict:
    kinds: dict[str, float] = defaultdict(float)
    launches = 0
    cpu_total = 0.0
    for event in profiler.key_averages():
        cuda_us = getattr(event, "device_time_total", 0.0) or 0.0
        name = event.key.lower()
        if event.device_type == torch.autograd.DeviceType.CUDA or cuda_us:
            kind = (
                "memcpy_h2d" if "memcpy htod" in name
                else "memcpy_d2d" if "memcpy dtod" in name
                else "gemm" if any(k in name for k in ("gemm", "cutlass", "sm80_xmma", "sm89", "ampere", "mm_", "cublas"))
                else "other"
            )
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kinds[kind] += event.self_device_time_total / 1e3
        if event.key in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaLaunchKernelEx"):
            launches += event.count
            cpu_total += event.self_cpu_time_total / 1e3
    return {"device_ms_by_kind": dict(kinds), "kernel_launches": launches, "launch_cpu_ms": cpu_total}


def summarize(steps: list[dict]) -> dict:
    out = {}
    for phase in ("prefill", "decode"):
        chosen = [s for s in steps if s["phase"] == phase and not s["traced"] and (phase == "prefill" or s["prompt"] > 0 or s["step"] > 3)]
        if not chosen:
            continue
        regions = sorted({k for s in chosen for k in s["regions_ms"]})
        mean = lambda key: statistics.mean(s[key] for s in chosen)  # noqa: E731
        entry = {
            "steps": len(chosen),
            "wall_ms": mean("wall_ms"),
            "regions_ms": {k: statistics.mean(s["regions_ms"].get(k, 0.0) for s in chosen) for k in regions},
            "h2d_device_ms": mean("h2d_device_ms"),
            "experts_device_ms": mean("experts_device_ms"),
            "physical_mb": mean("physical_bytes") / 1e6,
            "read_calls": mean("read_calls"),
            "pieces": mean("pieces"),
            "h2d_copies": mean("h2d_copies"),
        }
        accounted = sum(entry["regions_ms"].values())
        entry["regions_ms"]["other"] = entry["wall_ms"] - accounted
        entry["drive_gb_per_s"] = entry["physical_mb"] / 1e3 / max(1e-9, entry["regions_ms"].get("io", 0.0) / 1e3)
        out[phase] = entry
    traces = [s["trace"] for s in steps if "trace" in s]
    if traces:
        out["trace"] = traces
    return out


if __name__ == "__main__":
    raise SystemExit(main())
