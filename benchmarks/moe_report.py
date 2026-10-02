"""Aggregate a Phase 3 MoE run and evaluate gate C.

    uv run python benchmarks/moe_report.py experiments/phase3/<run> [--compare experiments/phase3/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff gate C holds
(configs/phase3-moe.yaml): every step of every configuration bit for bit the resident model's,
every audit clean, and a decode step without a cache reading at most the configured fraction of
the expert bytes. Fractions are of all expert bytes of the model.

Routing statistics come from the resident reference: experts routed per layer, how skewed the
routing is, and how many of a decode step's experts the previous step had already routed in
the same layer (what a prefetch of the previous step's experts would have caught, and wasted).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from awpmi.tracing import read_jsonl


def quantiles(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def phase(record: dict) -> str:
    return "prefill" if record["step"] == 0 else "decode"


def summarize(records: list[dict]) -> dict:
    requested = sum(r["requested_bytes"] for r in records)
    hits = sum(r["cache_hit_bytes"] for r in records)
    return {
        "steps": len(records),
        "bitwise": sum(r["matches_reference"] for r in records),
        "routing_matches": sum(r["routing_matches_reference"] for r in records),
        "audit_failures": sum(bool(r["audit"]) for r in records),
        "poisoned_steps": sum(r["poisoned"] for r in records),
        "requested_fraction": quantiles(r["requested_fraction"] for r in records),
        "drive_fraction": quantiles(r["drive_fraction"] for r in records),
        "cache_hit_rate": hits / requested if requested else 0.0,
        "h2d_mb": quantiles(r["h2d_bytes"] / 1e6 for r in records),
        "read_calls": quantiles(r["storage"]["read_calls"] for r in records),
        "step_ms": quantiles(r["timings_ms"]["step"] for r in records),
        "io_ms": quantiles(r["timings_ms"]["io"] for r in records),
        "copy_ms": quantiles(r["timings_ms"]["copy"] for r in records),
        "peak_device_mb": quantiles(r["system"]["peak_device_bytes"] / 1e6 for r in records),
        "os_counters_match": sum(
            r["system"]["os_read_bytes"] == r["storage"]["physical_bytes"] for r in records if r["system"]["os_read_bytes"] is not None
        ),
    }


def routing_statistics(reference: list[dict]) -> dict:
    decode = [r for r in reference if r["step"] > 0]
    per_layer = [n for r in decode for n in r["routed_per_layer"]]
    prefill = [n for r in reference if r["step"] == 0 for n in r["routed_per_layer"]]
    popularity = Counter()
    for r in decode:
        for layer, experts in enumerate(r["routed"]):
            popularity.update((layer, e) for e in experts)
    counts = sorted(popularity.values(), reverse=True)
    total = sum(counts)
    keys = len(popularity)
    captured = {}
    for share in (0.1, 0.25, 0.5):
        top = int(round(share * 32 * len(decode[0]["routed"]))) if decode else 0
        captured[f"top_{int(share * 100)}pct_expert_slots"] = sum(counts[:top]) / total if total else 0.0
    by_prompt = defaultdict(dict)
    for r in reference:
        by_prompt[r["prompt_id"]][r["step"]] = r
    reuse, waste = [], []
    for steps in by_prompt.values():
        for step in sorted(steps):
            if step < 2:
                continue  # the step after prefill is compared with a single-token step only from step 2 on
            current, previous = steps[step]["routed"], steps[step - 1]["routed"]
            for now, before in zip(current, previous):
                if now:
                    reuse.append(len(set(now) & set(before)) / len(now))
                if before:
                    waste.append(len(set(before) - set(now)) / len(before))
    return {
        "decode_experts_per_layer": quantiles(per_layer),
        "prefill_experts_per_layer": quantiles(prefill),
        "distinct_layer_expert_pairs_in_decode": keys,
        "routing_share_of_most_used_slots": captured,
        "previous_step_reuse": quantiles(reuse),
        "previous_step_waste": quantiles(waste),
    }


def render(summary: dict) -> str:
    lines = [
        f"# Phase 3 MoE run: {summary['run']}",
        "",
        f"Gate C (general backend, MoE): **{'PASS' if summary['gate']['passed'] else 'FAIL'}**",
        "",
        f"Model: {summary['model']} · experts: {summary['pack']['expert_modules']} layers × {summary['pack']['experts_per_module']} · "
        f"{summary['pack']['expert_bytes_total'] / 1e9:.2f} GB of experts, {summary['pack']['expert_bytes_per_expert'][0] / 2**20:.0f} MiB each · "
        f"copied into the pack: {len(summary['pack']['copied_segments'])} segments · experts implementation: {summary['pack']['experts_implementation']}",
        "",
        f"Device memory: resident model {summary['pack']['resident_device_bytes'] / 1e9:.2f} GB; streamed, before any cache: "
        f"{summary['device_bytes_streamed'] / 1e9:.2f} GB (slot buffers {summary['slot_buffer_bytes'] / 1e6:.0f} MB).",
        "",
        "## Per step (fractions of all expert bytes: mean / median / p95)",
        "",
        "| Configuration | Phase | Steps | Bitwise | Requested | Drive | Cache hit rate | H2D MB (mean) | Step ms (mean / p95) | Drive I/O ms | Copy ms | Peak device MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, s in summary["groups"].items():
        name, kind = key.split("/")
        fmt = lambda q: f"{q['mean']:.3f} / {q['median']:.3f} / {q['p95']:.3f}"  # noqa: E731
        lines.append(
            f"| {name} | {kind} | {s['steps']} | {s['bitwise']} | {fmt(s['requested_fraction'])} | {fmt(s['drive_fraction'])} | "
            f"{s['cache_hit_rate']:.2f} | {s['h2d_mb']['mean']:.0f} | {s['step_ms']['mean']:.0f} / {s['step_ms']['p95']:.0f} | "
            f"{s['io_ms']['mean']:.0f} | {s['copy_ms']['mean']:.0f} | {s['peak_device_mb']['mean']:.0f} |"
        )
    r = summary["routing"]
    lines += [
        "",
        "## Routing (resident reference)",
        "",
        f"- Experts per layer: decode {r['decode_experts_per_layer']['mean']:.2f}, prefill {r['prefill_experts_per_layer']['mean']:.2f} (of 32)",
        f"- Share of decode routings that the most used (layer, expert) slots receive: {json.dumps({k: round(v, 3) for k, v in r['routing_share_of_most_used_slots'].items()})}",
        f"- Of a decode step's experts, routed at the previous step in the same layer: mean {r['previous_step_reuse']['mean']:.3f} "
        f"(a previous-step prefetch would waste {r['previous_step_waste']['mean']:.3f} of what it fetched)",
        "",
        "## Gate C",
        "",
        f"{json.dumps(summary['gate'], indent=2)}",
    ]
    if summary.get("compare"):
        lines += ["", "## Reproducibility", "", json.dumps(summary["compare"], indent=2)]
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
    pack = json.loads((run / "pack.json").read_text(encoding="utf-8"))
    reference = read_jsonl(run / "reference.jsonl")
    records = read_jsonl(run / "records.jsonl.gz")
    failures = json.loads((run / "failure.json").read_text(encoding="utf-8")) if (run / "failure.json").exists() else []
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record['configuration']}/{phase(record)}"].append(record)
    groups = {key: summarize(rows) for key, rows in grouped.items()}
    expected_steps = {}
    prompts = environment["num_prompts"]
    steps = int(config["prompts"]["decode_steps"])
    for entry in config["configurations"]:
        count = min(int(entry.get("num_prompts", prompts)), prompts)
        expected_steps[entry["name"]] = count * (int(entry.get("decode_steps", steps)) + 1)
    completed = {name: sum(r["configuration"] == name for r in records) for name in expected_steps}
    no_cache = groups.get("stream/decode")
    threshold = config["gate"]["max_decode_drive_fraction_without_cache"]
    gate = {
        "hard_failures": len(failures),
        "all_steps_completed": completed == expected_steps,
        "all_steps_bitwise": all(s["bitwise"] == s["steps"] for s in groups.values()),
        "all_routing_matches": all(s["routing_matches"] == s["steps"] for s in groups.values()),
        "audit_failures": sum(s["audit_failures"] for s in groups.values()),
        "os_counters_match_all_steps": all(s["os_counters_match"] == s["steps"] for s in groups.values()),
        "poisoned_steps": sum(s["poisoned_steps"] for s in groups.values()),
        "decode_drive_fraction_without_cache": None if no_cache is None else no_cache["drive_fraction"]["mean"],
        "max_decode_drive_fraction_without_cache": threshold,
    }
    gate["passed"] = (
        gate["hard_failures"] == 0
        and gate["all_steps_completed"]
        and gate["all_steps_bitwise"]
        and gate["all_routing_matches"]
        and gate["audit_failures"] == 0
        and gate["os_counters_match_all_steps"]
        and no_cache is not None
        and no_cache["drive_fraction"]["mean"] <= threshold
    )
    streamed = next(iter(pack["configurations"].values()))
    summary = {
        "run": str(run),
        "model": f"{config['model']['repository']}@{config['model']['revision'][:8]}",
        "num_prompts": prompts,
        "source_tree_sha256": environment["source_tree_sha256"],
        "pack": {k: pack[k] for k in ("expert_modules", "experts_per_module", "expert_bytes_total", "expert_bytes_per_expert", "copied_segments", "experts_implementation", "resident_device_bytes")},
        "device_bytes_streamed": streamed["device_bytes_at_start"],
        "slot_buffer_bytes": streamed["slot_buffer_bytes"],
        "configurations": pack["configurations"],
        "groups": groups,
        "routing": routing_statistics(reference),
        "completed_steps": completed,
        "gate": gate,
    }
    if args.compare:
        mine = json.loads((run / "digest.json").read_text(encoding="utf-8"))
        theirs = json.loads((Path(args.compare) / "digest.json").read_text(encoding="utf-8"))
        other_env = json.loads((Path(args.compare) / "environment.json").read_text(encoding="utf-8"))
        summary["compare"] = {
            "other": args.compare,
            "same_source_tree": other_env["source_tree_sha256"] == environment["source_tree_sha256"],
            "digests_equal": {key: mine[key] == theirs[key] for key in mine if key.endswith("_sha256")},
        }
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run / "summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    passed = gate["passed"]
    if summary.get("compare"):
        passed = passed and summary["compare"]["same_source_tree"] and all(summary["compare"]["digests_equal"].values())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
