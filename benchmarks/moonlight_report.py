"""Aggregate a Phase 4B Moonlight run and evaluate its gates (decision 0008).

    uv run python benchmarks/moonlight_report.py experiments/phase4b/<run> [--compare experiments/phase4b/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff every gate of configs/phase4b-moonlight.yaml
holds (and, with --compare, the two runs used the same source tree and have equal digests):

  correctness  every step of every configuration equal to the reference in every recorded digest; every audit clean;
               the source files' sha256 equal the Hub's, every index row equals the reference's, the residency check
               agreed on every step
  A            routed expert bytes at least the configured multiple of the device-memory cap; every configuration completed
  B            both processes' peak working set and private bytes below the configured fraction of the routed expert bytes
  C            with a call budget, every step's expert buffers within it and within the configured fraction of a layer,
               prefill included; the expert working set within the cache capacity plus the budget
  D            decode drive bytes without a cache within the configured fraction of all routed expert bytes, and of what the
               all-experts stream reads per step
  E            the compact (one buffer per call) and chunk-1 (one expert per chunk) configurations equal the reference
  F            tests/test_layering.py passes

Also: the prefill table, host and device memory, routing statistics, and a replay of the run's routing through the cache
(the backend's request order and policies, checked against the measured hits, then at larger capacities).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from awpmi.storage.cache import POLICIES, PageCache
from awpmi.tracing import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]


def quantiles(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def summarize(records: list[dict]) -> dict:
    requested = sum(r["requested_bytes"] for r in records)
    hits = sum(r["cache_hit_bytes"] for r in records)
    lookups = [(r["cache"]["hits"], r["cache"]["misses"]) for r in records if r["cache"]]
    tokens = [r["positions"] if r["phase"] == "prefill" else 1 for r in records]
    kinds = sorted(records[0]["matches"]) if records else []
    return {
        "steps": len(records),
        "matches": {kind: sum(r["matches"][kind] for r in records) for kind in kinds},
        "all_equal": sum(all(r["matches"].values()) for r in records),
        "expert_outputs_checked_layers": sum(r["expert_outputs_checked"] for r in records),
        "expert_outputs_layers": sum(len(r["routed_per_layer"]) for r in records),
        "audit_failures": sum(bool(r["audit"]) for r in records),
        "poisoned_steps": sum(r["poisoned"] for r in records),
        "chunked_calls": sum(r["chunked_calls"] for r in records),
        "calls": sum(len(r["chunks_per_layer"]) for r in records),
        "experts_per_layer": quantiles(n for r in records for n in r["served_per_layer"]),
        "experts_requested_per_token": quantiles(sum(r["served_per_layer"]) / t for r, t in zip(records, tokens)),
        "requested_fraction": quantiles(r["requested_fraction"] for r in records),
        "drive_fraction": quantiles(r["drive_fraction"] for r in records),
        "drive_fraction_per_token": quantiles(r["drive_fraction"] / t for r, t in zip(records, tokens)),
        "h2d_fraction": quantiles(r["h2d_fraction"] for r in records),
        "amplification": quantiles(r["storage"]["physical_bytes"] / r["storage"]["logical_bytes"] for r in records if r["storage"]["logical_bytes"]),
        "cache_hit_rate_bytes": hits / requested if requested else 0.0,
        "cache_hit_rate_lookups": sum(h for h, _ in lookups) / max(1, sum(h + m for h, m in lookups)) if lookups else None,
        "physical_gb": quantiles(r["storage"]["physical_bytes"] / 1e9 for r in records),
        "read_calls": quantiles(r["storage"]["read_calls"] for r in records),
        "extents": quantiles(r["storage"]["extents"] for r in records),
        "h2d_gb": quantiles(r["h2d_bytes"] / 1e9 for r in records),
        "h2d_copies": quantiles(r["h2d_copies"] for r in records),
        "largest_request_mb": quantiles(r["largest_request_bytes"] / 1e6 for r in records),
        "compact_peak_mb": quantiles(r["compact_peak_bytes"] / 1e6 for r in records),
        "compact_fraction_of_layer": quantiles(r["compact_fraction_of_layer"] for r in records),
        "expert_peak_mb": quantiles(r["expert_peak_bytes"] / 1e6 for r in records),
        "peak_device_mb": quantiles(r["system"]["peak_device_bytes"] / 1e6 for r in records),
        "peak_reserved_mb": quantiles(r["system"]["peak_reserved_bytes"] / 1e6 for r in records),
        "process_resident_mb": quantiles(r["system"]["process"]["resident_bytes"] / 1e6 for r in records),
        "process_private_mb": quantiles(r["system"]["process"]["private_bytes"] / 1e6 for r in records),
        "step_ms_instrumented": quantiles(r["timings_ms"]["step"] for r in records),
        "io_ms": quantiles(r["timings_ms"]["io"] for r in records),
        "copy_ms": quantiles(r["timings_ms"]["copy"] for r in records),
        "os_counters_match": sum(
            r["system"]["os_read_bytes"] == r["storage"]["physical_bytes"] and r["system"]["os_read_calls"] == r["storage"]["read_calls"]
            for r in records
            if r["system"]["os_read_bytes"] is not None
        ),
    }


def routing_statistics(reference: list[dict], experts: int) -> dict:
    decode = [r for r in reference if r["step"] > 0]
    popularity = Counter()
    for r in decode:
        for layer, chosen in enumerate(r["routed"]):
            popularity.update((layer, e) for e in chosen)
    counts = sorted(popularity.values(), reverse=True)
    total = sum(counts)
    layers = len(decode[0]["routed"]) if decode else 0
    share = {f"top_{int(p * 100)}pct_slots": sum(counts[: int(round(p * experts * layers))]) / total if total else 0.0 for p in (0.1, 0.25, 0.5)}
    by_prompt = defaultdict(dict)
    for r in reference:
        by_prompt[r["prompt_id"]][r["step"]] = r
    reuse = []
    for steps in by_prompt.values():
        for step in sorted(steps):
            if step >= 2:
                for now, before in zip(steps[step]["routed"], steps[step - 1]["routed"]):
                    reuse.append(len(set(now) & set(before)) / len(now))
    prefill = defaultdict(list)
    for r in reference:
        if r["step"] == 0:
            prefill[r["positions"]].append(np.mean([len(c) for c in r["routed"]]))
    return {
        "decode_experts_per_layer": quantiles(len(c) for r in decode for c in r["routed"]),
        "prefill_experts_per_layer": quantiles(len(c) for r in reference if r["step"] == 0 for c in r["routed"]),
        "prefill_experts_per_layer_by_length": {int(k): float(np.mean(v)) for k, v in sorted(prefill.items())},
        "routing_share_of_most_used_slots": share,
        "previous_step_reuse": quantiles(reuse),
    }


class _Row:
    """A cache entry of a row's size, without its bytes (the replay needs only sizes)."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


def replay_cache(reference: list[dict], prompts: list[dict], segments: dict[str, int], capacity_bytes: int, policy: str,
                 half_life: float, call_budget: int | None, steps: int, num_prompts: int) -> dict:
    """The run's routing through a `PageCache` as the backend drives it (entries of each segment's row size).

    Per experts call: for each expert-sliced parameter in the group's order, one request per chunk (the whole call when
    its experts fit the call budget): every row looked up, misses inserted in order. Prompts in order, the cache
    persisting across them.
    """
    by_key = {(r["prompt_id"], r["step"]): r for r in reference}
    cache = PageCache(capacity_bytes, POLICIES[policy](half_life) if policy == "hotness" else POLICIES[policy]())
    expert_row = sum(segments.values())
    rows = {segment: _Row(nbytes) for segment, nbytes in segments.items()}
    per_step = []
    for prompt in prompts[:num_prompts]:
        for step in range(steps + 1):
            record = by_key[(prompt["prompt_id"], step)]
            before = (cache.stats.hits, cache.stats.misses)
            for layer, experts in enumerate(record["routed"]):
                chunked = call_budget is not None and len(experts) * expert_row > call_budget
                size = call_budget // expert_row if chunked else len(experts)
                for segment, nbytes in segments.items():
                    for first in range(0, len(experts), size):
                        keys = [((layer, segment), e) for e in experts[first : first + size]]
                        found = cache.get_many(keys, nbytes)
                        for key, entry in zip(keys, found):
                            if entry is None:
                                cache.put(key, rows[segment])
            per_step.append({"prompt_id": prompt["prompt_id"], "step": step, "hits": cache.stats.hits - before[0], "misses": cache.stats.misses - before[1]})
    return {"steps": per_step, "hits": cache.stats.hits, "misses": cache.stats.misses}


def cache_study(config: dict, reference: list[dict], records: list[dict], prompts: list[dict], segments: dict[str, int],
                experts_per_layer: int, call_budget: int | None) -> dict:
    expert_row = sum(segments.values())
    steps = int(config["prompts"]["decode_steps"])
    half_life = float(config["hotness_half_life"])
    measured = {}
    for entry in config["configurations"]:
        if not entry.get("cache_experts"):
            continue
        name = entry["name"]
        count = min(int(entry.get("num_prompts", len(prompts))), len(prompts))
        budget = entry["call_budget_experts"] * expert_row if "call_budget_experts" in entry else entry.get("call_budget_bytes", call_budget)
        replay = replay_cache(reference, prompts, segments, int(entry["cache_experts"]) * expert_row, entry["policy"], half_life, budget, steps, count)
        mine = [r for r in records if r["configuration"] == name]
        agree = len(mine) == len(replay["steps"]) and all(
            (r["cache"]["hits"], r["cache"]["misses"]) == (s["hits"], s["misses"]) for r, s in zip(mine, replay["steps"])
        )
        measured[name] = {"replay_equals_measured_hits_every_step": agree, "steps": len(mine)}
    study = {}
    total_experts = len(reference[0]["routed"]) * experts_per_layer
    for policy in ("lru", "hotness"):
        for capacity in (40, 80, 160, 320, 640, 960, 1248):
            replay = replay_cache(reference, prompts, segments, capacity * expert_row, policy, half_life, call_budget, steps, len(prompts))
            decode = [s for s in replay["steps"] if s["step"] > 0]
            prefill = [s for s in replay["steps"] if s["step"] == 0]
            rate = lambda rows: sum(s["hits"] for s in rows) / max(1, sum(s["hits"] + s["misses"] for s in rows))  # noqa: E731
            study[f"{policy}-{capacity}"] = {
                "capacity_experts": capacity, "capacity_gb": capacity * expert_row / 1e9, "fraction_of_experts": capacity / total_experts,
                "decode_hit_rate": rate(decode), "prefill_hit_rate": rate(prefill),
            }
    return {"replay_checked_against_measured": measured, "capacities": study}


def profile_summary(run: Path) -> dict | None:
    """The un-instrumented profiles (`profile-<configuration>.json`, benchmarks/moonlight_profile.py), without their steps."""
    found = {}
    for path in sorted(run.glob("profile-*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        found[profile["configuration"]] = {key: value for key, value in profile.items() if key != "steps"}
    return found or None


def render(summary: dict) -> str:
    gate = summary["gates"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    stream = summary["stream_stage"]
    reference = summary["reference_stage"]
    lines = [
        f"# Phase 4B Moonlight run: {summary['run']}",
        "",
        " · ".join(f"{name}: **{mark(g['passed'])}**" for name, g in gate.items()),
        "",
        f"Model: {summary['model']} · {summary['experts']['layers']} MoE layers × {summary['experts']['per_layer']} experts · "
        f"{summary['experts']['bytes'] / 1e9:.2f} GB of routed experts ({summary['experts']['row_bytes'] / 2**20:.1f} MiB each) · "
        f"GPU {stream['gpu_total_bytes'] / 1e9:.2f} GB, cap {stream['gpu_budget_bytes'] / 1e9:.2f} GB · non-expert weights "
        f"{stream['non_expert_device_bytes'] / 1e9:.2f} GB on the device (shared experts {stream['shared_experts_device_bytes'] / 1e9:.2f} GB) · "
        f"prompts {summary['num_prompts']}",
        "",
        "## Per step (fractions of all routed expert bytes: mean / median / p95)",
        "",
        "| Configuration | Phase | Steps | All equal | Expert checks | Experts / layer | Drive | Drive / token | H2D | Hit rate | Reads | Chunked calls | Compact MB (max) | Expert MB (max) | Peak device MB | Step ms* | I/O ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    fmt = lambda q: f"{q['mean']:.4f} / {q['median']:.4f} / {q['p95']:.4f}" if q else "–"  # noqa: E731
    for key, s in summary["groups"].items():
        name, phase = key.split("/")
        hit = "–" if s["cache_hit_rate_lookups"] is None else f"{s['cache_hit_rate_lookups']:.3f}"
        lines.append(
            f"| {name} | {phase} | {s['steps']} | {s['all_equal']} | {s['expert_outputs_checked_layers']}/{s['expert_outputs_layers']} | "
            f"{s['experts_per_layer']['mean']:.1f} | {fmt(s['drive_fraction'])} | {s['drive_fraction_per_token']['mean']:.4f} | "
            f"{s['h2d_fraction']['mean']:.4f} | {hit} | {s['read_calls']['mean']:.0f} | {s['chunked_calls']}/{s['calls']} | "
            f"{s['compact_peak_mb']['max']:.0f} | {s['expert_peak_mb']['max']:.0f} | {s['peak_device_mb']['max']:.0f} | "
            f"{s['step_ms_instrumented']['mean']:.0f} | {s['io_ms']['mean']:.0f} |"
        )
    lines += ["", "\\* Step times include the benchmark's digests and checks; see the profile for un-instrumented times."]
    if summary.get("profile"):
        lines += ["", "## Profile (no digests): host milliseconds per step by region, and their shares", ""]
        for name, profile in summary["profile"].items():
            for phase, entry in profile["summary"].items():
                regions = ", ".join(f"{k} {v:.0f} ({entry['share'][k]:.0%})" for k, v in sorted(entry["regions_ms"].items(), key=lambda kv: -kv[1]))
                lines.append(f"- {name} {phase}: {entry['wall_ms']:.0f} ms over {entry['steps']} steps: {regions}; drive {entry['drive_gb_per_s']:.2f} GB/s")
            for trace in profile.get("traces", []):
                kinds = ", ".join(f"{k} {v:.0f}" for k, v in sorted(trace["device_ms_by_kind"].items(), key=lambda kv: -kv[1]))
                lines.append(
                    f"  - trace {trace['phase']} step {trace['step']}: wall {trace['wall_ms']:.0f} ms, GPU busy {trace['gpu_busy_ms']:.0f} ms "
                    f"(idle {trace['gpu_idle_fraction']:.0%}), {trace['kernel_launches']} launches ({trace['launch_cpu_ms']:.0f} ms CPU); device ms: {kinds}"
                )
    memory = summary["memory"]
    lines += ["", "## Host and device memory", "", "```json", json.dumps(memory, indent=1), "```"]
    lines += [
        "",
        "## Reference stage",
        "",
        f"- Load: {reference['load']['load_seconds']:.1f} s ({reference['load']['non_expert_checkpoint_bytes'] / 1e9:.2f} GB of non-expert weights).",
        f"- Streamed run: {reference['streamed']['layer_loads']} experts-layer loads, {reference['streamed']['loaded_bytes'] / 1e9:.1f} GB converted in "
        f"{reference['streamed']['load_seconds']:.0f} s ({reference['streamed']['loaded_bytes'] / 1e9 / max(1e-9, reference['streamed']['load_seconds']):.2f} GB/s); "
        f"peak device {reference['streamed']['peak_device_bytes'] / 1e9:.2f} GB, experts at most {reference['streamed']['peak_expert_device_bytes'] / 1e9:.2f} GB.",
    ]
    routing = summary["routing"]
    lines += [
        "",
        "## Routing (reference)",
        "",
        f"- Experts per layer: decode {routing['decode_experts_per_layer']['mean']:.2f}, prefill {routing['prefill_experts_per_layer']['mean']:.2f} (of {summary['experts']['per_layer']}); by prompt length: {json.dumps({k: round(v, 1) for k, v in routing['prefill_experts_per_layer_by_length'].items()})}",
        f"- Share of decode routings received by the most used (layer, expert) slots: {json.dumps({k: round(v, 3) for k, v in routing['routing_share_of_most_used_slots'].items()})}",
        f"- Of a decode step's experts, routed at the previous step in the same layer: {routing['previous_step_reuse'].get('mean', float('nan')):.3f}",
        "",
        "## Cache replay",
        "",
        "```json",
        json.dumps(summary["cache_study"], indent=1),
        "```",
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(gate, indent=2),
        "```",
    ]
    if summary.get("compare"):
        lines += ["", "## Reproducibility", "", "```json", json.dumps(summary["compare"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    index = json.loads((run / "index.json").read_text(encoding="utf-8"))
    reference_stage = json.loads((run / "reference_stage.json").read_text(encoding="utf-8"))
    stream_stage = json.loads((run / "stream_stage.json").read_text(encoding="utf-8"))
    reference = read_jsonl(run / "reference.jsonl.gz")
    records = read_jsonl(run / "records.jsonl.gz")
    prompts = read_jsonl(run / "prompts.jsonl")
    failures = json.loads((run / "failure.json").read_text(encoding="utf-8")) if (run / "failure.json").exists() else []
    layers = len(reference[0]["routed"])
    per_layer = next(iter(index["manifest"]["metadata"]["groups"].values()))["experts"]
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record['configuration']}/{record['phase']}"].append(record)
    groups = {key: summarize(rows) for key, rows in grouped.items()}
    num_prompts = len(prompts)
    steps = int(config["prompts"]["decode_steps"])
    expected = {
        e["name"]: min(int(e.get("num_prompts", num_prompts)), num_prompts) * (min(int(e.get("decode_steps", steps)), steps) + 1)
        for e in config["configurations"]
    }
    completed = {name: sum(r["configuration"] == name for r in records) for name in expected}
    expected_reference = num_prompts * (steps + 1)
    gates_config = config["gate"]
    expert_bytes = stream_stage["expert_bytes_total"]
    expert_row = stream_stage["expert_row_bytes"][0]
    layer_bytes = stream_stage["layer_bytes"]

    rows_expected = sum(int(s["rows"]) for s in index["manifest"]["segments"].values())
    correctness = {
        "hard_failures": len(failures),
        "reference_steps": len(reference),
        "reference_steps_expected": expected_reference,
        "source_files_verified_against_hub_sha256": index.get("verification") == "files against the Hub's sha256" and all(
            entry.get("sha256_direct_read") == entry.get("sha256_declared") is not None for entry in index.get("other_files", {}).values()
        ),
        "source_files_verified": len(index["manifest"]["files"]) + len(index.get("other_files", {})),
        "index_rows_audited": index["audit"].get("rows", 0),
        "index_rows_expected": rows_expected,
        "index_rows_differing": len(index["audit"]["differing"]),
        "residency_checked_steps": sum("residency_check" in r for r in reference),
        "residency_check_all_equal": all(r.get("residency_check", True) for r in reference),
        "steps_all_equal": {name: sum(all(r["matches"].values()) for r in records if r["configuration"] == name) for name in expected},
        "steps_expected": expected,
        "quantities_compared": sorted(records[0]["matches"]) if records else [],
        "expert_outputs_checked_layers": sum(r["expert_outputs_checked"] for r in records),
        "expert_outputs_layers": sum(len(r["routed_per_layer"]) for r in records),
        "audit_failures": sum(s["audit_failures"] for s in groups.values()),
        "os_counters_match_all_steps": all(s["os_counters_match"] == s["steps"] for s in groups.values()),
        "poisoned_steps": sum(s["poisoned_steps"] for s in groups.values()),
    }
    correctness["passed"] = (
        not failures
        and correctness["reference_steps"] == expected_reference
        and correctness["source_files_verified_against_hub_sha256"]
        and correctness["index_rows_audited"] == rows_expected
        and correctness["index_rows_differing"] == 0
        and correctness["residency_checked_steps"] > 0
        and correctness["residency_check_all_equal"]
        and all(correctness["steps_all_equal"][name] == expected[name] == completed[name] for name in expected)
        and correctness["audit_failures"] == 0
        and correctness["os_counters_match_all_steps"]
    )

    gate_a = {
        "expert_bytes": expert_bytes,
        "gpu_budget_bytes": stream_stage["gpu_budget_bytes"],
        "gpu_total_bytes": stream_stage["gpu_total_bytes"],
        "experts_over_budget": expert_bytes / stream_stage["gpu_budget_bytes"],
        "experts_over_gpu": expert_bytes / stream_stage["gpu_total_bytes"],
        "min_experts_over_budget": gates_config["min_expert_bytes_over_cap"],
        "completed_steps": completed,
        "max_peak_device_bytes": max(r["system"]["peak_device_bytes"] for r in records),
        "max_peak_reserved_bytes": max(r["system"]["peak_reserved_bytes"] for r in records),
    }
    gate_a["passed"] = (
        gate_a["experts_over_budget"] >= gate_a["min_experts_over_budget"]
        and completed == expected
        and gate_a["max_peak_reserved_bytes"] <= stream_stage["gpu_budget_bytes"]
    )

    def peak(memory: dict | None, key: str) -> int:
        return int((memory or {}).get(key) or 0)

    stream_phases = stream_stage["memory"]["phases"]
    keys = ("resident_bytes", "private_bytes", "host_commit_bytes")
    host = {
        "stream": {
            "before_setup": stream_stage["memory"]["start"],
            "after_non_expert_load": stream_stage["memory"]["after_load"],
            "peak_prefill": {k: max(p.get("prefill", {}).get(k, 0) for p in stream_phases.values()) for k in keys},
            "peak_decode": {k: max(p.get("decode", {}).get(k, 0) for p in stream_phases.values()) for k in keys},
            "peak_by_configuration": stream_phases,
            "lifetime_peak_resident_bytes": peak(stream_stage["memory"]["peak"], "peak_resident_bytes"),
            "pinned_staging_bytes_per_backend": stream_stage["host_staging_bytes"],
            "pinned_host_memory": stream_stage.get("pinned_host_memory"),
            "system": stream_stage["system"],
        },
        "reference": {
            "before_setup": reference_stage["memory"]["start"],
            "after_non_expert_load": reference_stage["memory"]["after_load"],
            "peak_prefill": reference_stage["memory"]["phases"]["prefill"],
            "peak_decode": reference_stage["memory"]["phases"]["decode"],
            "lifetime_peak_resident_bytes": peak(reference_stage["memory"]["peak"], "peak_resident_bytes"),
            "system": reference_stage["system"],
        },
    }
    # Physical host memory (working set) and host commit (private bytes less the device memory WDDM charges to them).
    worst = {
        process: {
            "working_set": max(entry["lifetime_peak_resident_bytes"], *(entry[p]["resident_bytes"] for p in ("peak_prefill", "peak_decode"))),
            "host_commit": max(entry[p]["host_commit_bytes"] for p in ("peak_prefill", "peak_decode")),
            "private_bytes_with_device_memory": max(entry[p]["private_bytes"] for p in ("peak_prefill", "peak_decode")),
        }
        for process, entry in host.items()
    }
    gate_b = {
        "worst_process_bytes": worst,
        "fraction_of_experts": {
            process: {k: v / expert_bytes for k, v in values.items() if k != "private_bytes_with_device_memory"} for process, values in worst.items()
        },
        "max_fraction_of_experts": gates_config["max_host_fraction_of_experts"],
    }
    gate_b["passed"] = all(v <= gate_b["max_fraction_of_experts"] for values in gate_b["fraction_of_experts"].values() for v in values.values())

    budgets = {name: entry["call_budget_bytes"] for name, entry in stream_stage["configurations"].items()}
    capacities = {name: entry["cache_capacity_bytes"] for name, entry in stream_stage["configurations"].items()}
    budgeted = [r for r in records if budgets.get(r["configuration"]) is not None]
    prefill = [r for r in budgeted if r["phase"] == "prefill"]
    gate_c = {
        "steps_with_a_budget": len(budgeted),
        "steps_over_budget": sum(r["compact_peak_bytes"] > budgets[r["configuration"]] for r in budgeted),
        "max_compact_fraction_of_layer": max((r["compact_fraction_of_layer"] for r in budgeted), default=None),
        "max_compact_fraction_of_layer_allowed": gates_config["max_compact_fraction_of_layer"],
        "prefill_steps_completed": len(prefill),
        "prefill_max_compact_bytes": max((r["compact_peak_bytes"] for r in prefill), default=None),
        "prefill_max_routed_experts_per_layer": max((max(r["routed_per_layer"]) for r in prefill), default=None),
        "layer_bytes": layer_bytes,
        "working_set_within_capacity_plus_budget": all(
            r["expert_peak_bytes"] <= capacities[r["configuration"]] + max(budgets[r["configuration"]], r["compact_peak_bytes"]) for r in budgeted
        ),
        "unbounded_compact_max_bytes": max((r["compact_peak_bytes"] for r in records if budgets.get(r["configuration"]) is None), default=None),
    }
    gate_c["passed"] = (
        bool(prefill)
        and gate_c["steps_over_budget"] == 0
        and gate_c["max_compact_fraction_of_layer"] <= gate_c["max_compact_fraction_of_layer_allowed"]
        and gate_c["working_set_within_capacity_plus_budget"]
    )

    no_cache = groups.get("stream/decode")
    dense = groups.get("all-experts/decode")
    gate_d = {
        "decode_drive_fraction_without_cache": None if no_cache is None else no_cache["drive_fraction"]["mean"],
        "max_decode_drive_fraction_without_cache": gates_config["max_decode_drive_fraction_without_cache"],
        "all_experts_decode_drive_fraction": None if dense is None else dense["drive_fraction"]["mean"],
        "decode_drive_ratio_to_all_experts": None if no_cache is None or dense is None else no_cache["drive_fraction"]["mean"] / dense["drive_fraction"]["mean"],
        "max_decode_drive_ratio_to_all_experts": gates_config["max_decode_drive_ratio_to_all_experts"],
        "decode_drive_fraction_by_configuration": {k.split("/")[0]: v["drive_fraction"]["mean"] for k, v in groups.items() if k.endswith("/decode")},
        "prefill_drive_fraction_by_configuration": {k.split("/")[0]: v["drive_fraction"]["mean"] for k, v in groups.items() if k.endswith("/prefill")},
    }
    gate_d["passed"] = (
        no_cache is not None
        and dense is not None
        and gate_d["decode_drive_fraction_without_cache"] <= gate_d["max_decode_drive_fraction_without_cache"]
        and gate_d["decode_drive_ratio_to_all_experts"] <= gate_d["max_decode_drive_ratio_to_all_experts"]
    )

    gate_e = {
        name: {"steps": completed.get(name, 0), "equal": correctness["steps_all_equal"].get(name, 0), "chunked_calls": groups.get(f"{name}/prefill", {}).get("chunked_calls", 0) + groups.get(f"{name}/decode", {}).get("chunked_calls", 0)}
        for name in ("compact", "chunk-1", "stream")
    }
    gate_e = {
        "configurations": gate_e,
        "passed": all(e["steps"] > 0 and e["equal"] == e["steps"] == expected[name] for name, e in gate_e.items())
        and gate_e["compact"]["chunked_calls"] == 0 and gate_e["chunk-1"]["chunked_calls"] > 0,
    }

    layering = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests" / "test_layering.py")], capture_output=True, text=True, cwd=REPO_ROOT)
    gate_f = {"test_layering": layering.stdout.strip().splitlines()[-1] if layering.stdout.strip() else layering.stderr[-200:], "passed": layering.returncode == 0}

    summary = {
        "run": str(run),
        "model": f"{config['model']['repository']}@{config['model']['revision'][:8]}",
        "num_prompts": num_prompts,
        "source_tree_sha256": environment["source_tree_sha256"],
        "python_hash_seed": environment.get("python_hash_seed"),
        "profile_declared": environment.get("profile"),
        "experts": {"layers": layers, "per_layer": per_layer, "bytes": expert_bytes, "row_bytes": expert_row},
        "reference_stage": reference_stage,
        "stream_stage": stream_stage,
        "groups": groups,
        "memory": host,
        "routing": routing_statistics(reference, per_layer),
        "cache_study": cache_study(
            config, reference, records, prompts,
            {p: index["manifest"]["segments"][s]["row_bytes"] for p, s in next(iter(index["manifest"]["metadata"]["groups"].values()))["segments"].items()},
            per_layer, config.get("call_budget_bytes"),
        ),
        "profile": profile_summary(run),
        "gates": {"correctness": correctness, "A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e, "F": gate_f},
    }
    if args.compare:
        mine = json.loads((run / "digest.json").read_text(encoding="utf-8"))
        theirs = json.loads((Path(args.compare) / "digest.json").read_text(encoding="utf-8"))
        other_env = json.loads((Path(args.compare) / "environment.json").read_text(encoding="utf-8"))
        summary["compare"] = {
            "other": args.compare,
            "same_source_tree": other_env["source_tree_sha256"] == environment["source_tree_sha256"],
            "python_hash_seeds": [environment.get("python_hash_seed"), other_env.get("python_hash_seed")],
            "digests_equal": {key: mine[key] == theirs[key] for key in mine if key.endswith("_sha256")},
        }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run / "summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    passed = all(g["passed"] for g in summary["gates"].values())
    if summary.get("compare"):
        passed = passed and summary["compare"]["same_source_tree"] and all(summary["compare"]["digests_equal"].values())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
