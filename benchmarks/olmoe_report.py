"""Aggregate a Phase 4A OLMoE run and evaluate its gates (decision 0007).

    uv run python benchmarks/olmoe_report.py experiments/phase4a/<run> [--compare experiments/phase4a/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff every gate of
configs/phase4a-olmoe.yaml holds (and, with --compare, the two runs used the same source tree
and have equal digests):

  correctness  every step of every configuration equal to the reference in every recorded
               digest (token, logits, KV cache, routed experts; per layer router logits, top-k
               indices and weights, every expert output, the experts output), every audit clean;
               the source files' sha256 equal the Hub's (the reference stage verified them), the
               index rows equal the loader's tensors, and the residency check agreed on every step
  A            the expert bytes exceed the device-memory cap of the streamed process and the
               device, and every configuration completed
  B            decode drive bytes without a cache within the configured fraction of all expert
               bytes, and of the bytes streaming every expert reads
  C            compact buffers audited on every step, within the configured fraction of a layer on
               decode steps; the expert working set within the cache capacity plus one call's
               compact buffers
  D            tests/test_layering.py passes

Fractions are of all expert bytes. Step times in records include the benchmark's own digests;
`profile.json` (benchmarks/olmoe_profile.py), if present, gives un-instrumented times.
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

from awpmi.tracing import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCH_KINDS = ("token", "logits", "kv_cache", "routed", "router_logits", "top_k_index", "top_k_weights", "expert_outputs", "output")


def quantiles(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def summarize(records: list[dict], layers: int) -> dict:
    requested = sum(r["requested_bytes"] for r in records)
    hits = sum(r["cache_hit_bytes"] for r in records)
    lookups = [(r["cache"]["hits"], r["cache"]["misses"]) for r in records if r["cache"]]
    tokens = [r["positions"] if r["phase"] == "prefill" else 1 for r in records]
    return {
        "steps": len(records),
        "matches": {kind: sum(r["matches"][kind] for r in records) for kind in MATCH_KINDS},
        "audit_failures": sum(bool(r["audit"]) for r in records),
        "poisoned_steps": sum(r["poisoned"] for r in records),
        "experts_per_layer": quantiles(n for r in records for n in r["served_per_layer"]),
        "experts_requested_per_token": quantiles(sum(r["served_per_layer"]) / t for r, t in zip(records, tokens)),
        "requested_fraction": quantiles(r["requested_fraction"] for r in records),
        "drive_fraction": quantiles(r["drive_fraction"] for r in records),
        "drive_fraction_per_token": quantiles(r["drive_fraction"] / t for r, t in zip(records, tokens)),
        "h2d_fraction": quantiles(r["h2d_fraction"] for r in records),
        "amplification": quantiles(r["storage"]["physical_bytes"] / r["storage"]["logical_bytes"] for r in records if r["storage"]["logical_bytes"]),
        "cache_hit_rate_bytes": hits / requested if requested else 0.0,
        "cache_hit_rate_lookups": sum(h for h, _ in lookups) / max(1, sum(h + m for h, m in lookups)) if lookups else None,
        "cache_miss_rate_lookups": sum(m for _, m in lookups) / max(1, sum(h + m for h, m in lookups)) if lookups else None,
        "physical_mb": quantiles(r["storage"]["physical_bytes"] / 1e6 for r in records),
        "read_calls": quantiles(r["storage"]["read_calls"] for r in records),
        "extents": quantiles(r["storage"]["extents"] for r in records),
        "h2d_mb": quantiles(r["h2d_bytes"] / 1e6 for r in records),
        "compact_peak_mb": quantiles(r["compact_peak_bytes"] / 1e6 for r in records),
        "compact_fraction_of_layer": quantiles(r["compact_fraction_of_layer"] for r in records),
        "expert_peak_mb": quantiles(r["expert_peak_bytes"] / 1e6 for r in records),
        "peak_device_mb": quantiles(r["system"]["peak_device_bytes"] / 1e6 for r in records),
        "peak_reserved_mb": quantiles(r["system"]["peak_reserved_bytes"] / 1e6 for r in records),
        "process_resident_mb": quantiles(r["system"]["process_resident_bytes"] / 1e6 for r in records if r["system"]["process_resident_bytes"]),
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
    return {
        "decode_experts_per_layer": quantiles(len(c) for r in decode for c in r["routed"]),
        "prefill_experts_per_layer": quantiles(len(c) for r in reference if r["step"] == 0 for c in r["routed"]),
        "routing_share_of_most_used_slots": share,
        "previous_step_reuse": quantiles(reuse),
    }


def profile_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, entry in profile["configurations"].items():
        out[name] = {
            phase: {k: entry[phase][k] for k in ("steps", "wall_ms", "regions_ms", "h2d_device_ms", "experts_device_ms", "physical_mb", "pieces", "h2d_copies", "drive_gb_per_s")}
            for phase in ("prefill", "decode")
            if phase in entry
        }
        if "trace" in entry:
            out[name]["trace"] = entry["trace"]
    return out


def render(summary: dict) -> str:
    gate = summary["gates"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    stream = summary["stream_stage"]
    lines = [
        f"# Phase 4A OLMoE run: {summary['run']}",
        "",
        f"Correctness: **{mark(gate['correctness']['passed'])}** · Gate A (out of VRAM): **{mark(gate['A']['passed'])}** · "
        f"Gate B (selective I/O): **{mark(gate['B']['passed'])}** · Gate C (compact residency): **{mark(gate['C']['passed'])}** · "
        f"Gate D (model-general core): **{mark(gate['D']['passed'])}**",
        "",
        f"Model: {summary['model']} · {summary['experts']['layers']} layers × {summary['experts']['per_layer']} experts · "
        f"{summary['experts']['bytes'] / 1e9:.2f} GB of experts ({summary['experts']['row_bytes'] / 2**20:.0f} MiB each) · "
        f"GPU {stream['gpu_total_bytes'] / 1e9:.2f} GB, budget {stream['gpu_budget_bytes'] / 1e9:.2f} GB · "
        f"non-expert weights on the device {stream['non_expert_device_bytes'] / 1e9:.2f} GB · prompts {summary['num_prompts']}",
        "",
        "## Per step (fractions of all expert bytes: mean / median / p95)",
        "",
        "| Configuration | Phase | Steps | All equal | Experts / token | Drive | Drive / token | H2D | Hit rate (lookups) | Reads | Compact MB (max) | Expert MB (max) | Peak device MB (max) | Step ms* | I/O ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    fmt = lambda q: f"{q['mean']:.4f} / {q['median']:.4f} / {q['p95']:.4f}" if q else "–"  # noqa: E731
    for key, s in summary["groups"].items():
        name, phase = key.split("/")
        equal = min(s["matches"].values())
        hit = "–" if s["cache_hit_rate_lookups"] is None else f"{s['cache_hit_rate_lookups']:.3f}"
        lines.append(
            f"| {name} | {phase} | {s['steps']} | {equal} | {s['experts_requested_per_token']['mean']:.1f} | {fmt(s['drive_fraction'])} | "
            f"{s['drive_fraction_per_token']['mean']:.4f} | {s['h2d_fraction']['mean']:.4f} | {hit} | {s['read_calls']['mean']:.0f} | "
            f"{s['compact_peak_mb']['max']:.0f} | {s['expert_peak_mb']['max']:.0f} | {s['peak_device_mb']['max']:.0f} | "
            f"{s['step_ms_instrumented']['mean']:.0f} | {s['io_ms']['mean']:.0f} |"
        )
    lines += ["", "\\* Step times include the benchmark's digests (hashing, the direct expert-output check); see the profile."]
    if summary.get("profile"):
        lines += ["", "## Profile (no digests)", ""]
        for name, entry in summary["profile"].items():
            for phase in ("prefill", "decode"):
                if phase in entry:
                    e = entry[phase]
                    regions = ", ".join(f"{k} {v:.0f}" for k, v in sorted(e["regions_ms"].items(), key=lambda kv: -kv[1]))
                    lines.append(f"- {name} {phase}: {e['wall_ms']:.0f} ms ({regions}); H2D device {e['h2d_device_ms']:.0f} ms; experts device {e['experts_device_ms']:.0f} ms; drive {e['drive_gb_per_s']:.2f} GB/s")
    routing = summary["routing"]
    lines += [
        "",
        "## Routing (reference)",
        "",
        f"- Experts per layer: decode {routing['decode_experts_per_layer']['mean']:.2f}, prefill {routing['prefill_experts_per_layer']['mean']:.2f} (of {summary['experts']['per_layer']})",
        f"- Share of decode routings received by the most used (layer, expert) slots: {json.dumps({k: round(v, 3) for k, v in routing['routing_share_of_most_used_slots'].items()})}",
        f"- Of a decode step's experts, routed at the previous step in the same layer: {routing['previous_step_reuse']['mean']:.3f}",
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
    failures = json.loads((run / "failure.json").read_text(encoding="utf-8")) if (run / "failure.json").exists() else []
    layers = len(reference[0]["routed"])
    groups_meta = index["manifest"]["metadata"]["groups"]
    per_layer = next(iter(groups_meta.values()))["experts"]
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record['configuration']}/{record['phase']}"].append(record)
    groups = {key: summarize(rows, layers) for key, rows in grouped.items()}
    prompts = environment["num_prompts"]
    steps = int(config["prompts"]["decode_steps"])
    expected = {e["name"]: min(int(e.get("num_prompts", prompts)), prompts) * (int(e.get("decode_steps", steps)) + 1) for e in config["configurations"]}
    completed = {name: sum(r["configuration"] == name for r in records) for name in expected}
    expected_reference = prompts * (steps + 1)
    gates_config = config["gate"]

    correctness = {
        "hard_failures": len(failures),
        "reference_steps": len(reference),
        "reference_steps_expected": expected_reference,
        "index_rows_audited": index["audit"]["rows"],
        "index_rows_differing": len(index["audit"]["differing"]),
        "source_files_verified_against_hub_sha256": "verify_files" in reference_stage["timings_ms"],
        "residency_checked_steps": sum("residency_check" in r for r in reference),
        "residency_check_all_equal": all(r.get("residency_check", True) for r in reference),
        "steps_all_equal": {name: sum(all(r["matches"].values()) for r in records if r["configuration"] == name) for name in expected},
        "audit_failures": sum(s["audit_failures"] for s in groups.values()),
        "os_counters_match_all_steps": all(s["os_counters_match"] == s["steps"] for s in groups.values()),
        "poisoned_steps": sum(s["poisoned_steps"] for s in groups.values()),
    }
    correctness["passed"] = (
        not failures
        and correctness["reference_steps"] == expected_reference
        and correctness["index_rows_audited"] == sum(int(s["rows"]) for s in index["manifest"]["segments"].values())
        and correctness["index_rows_differing"] == 0
        and correctness["source_files_verified_against_hub_sha256"]
        and correctness["residency_checked_steps"] > 0
        and correctness["residency_check_all_equal"]
        and all(correctness["steps_all_equal"][name] == expected[name] == completed[name] for name in expected)
        and correctness["audit_failures"] == 0
        and correctness["os_counters_match_all_steps"]
    )

    expert_bytes = stream_stage["expert_bytes_total"]
    gate_a = {
        "expert_bytes": expert_bytes,
        "gpu_budget_bytes": stream_stage["gpu_budget_bytes"],
        "gpu_total_bytes": stream_stage["gpu_total_bytes"],
        "experts_over_budget": expert_bytes / stream_stage["gpu_budget_bytes"],
        "experts_over_gpu": expert_bytes / stream_stage["gpu_total_bytes"],
        "completed_steps": completed,
        "expected_steps": expected,
        "max_peak_device_bytes": max(r["system"]["peak_device_bytes"] for r in records),
        "max_peak_reserved_bytes": max(r["system"]["peak_reserved_bytes"] for r in records),
    }
    gate_a["passed"] = expert_bytes > gate_a["gpu_budget_bytes"] and expert_bytes > gate_a["gpu_total_bytes"] and completed == expected

    no_cache = groups.get("stream/decode")
    dense = groups.get("all-experts/decode")
    gate_b = {
        "decode_drive_fraction_without_cache": None if no_cache is None else no_cache["drive_fraction"]["mean"],
        "max_decode_drive_fraction_without_cache": gates_config["max_decode_drive_fraction_without_cache"],
        "all_experts_decode_drive_fraction": None if dense is None else dense["drive_fraction"]["mean"],
        "decode_drive_ratio_to_all_experts": None if no_cache is None or dense is None else no_cache["drive_fraction"]["mean"] / dense["drive_fraction"]["mean"],
        "max_decode_drive_ratio_to_all_experts": gates_config["max_decode_drive_ratio_to_all_experts"],
        "decode_drive_fraction_by_configuration": {k.split("/")[0]: v["drive_fraction"]["mean"] for k, v in groups.items() if k.endswith("/decode")},
    }
    gate_b["passed"] = (
        no_cache is not None
        and dense is not None
        and gate_b["decode_drive_fraction_without_cache"] <= gate_b["max_decode_drive_fraction_without_cache"]
        and gate_b["decode_drive_ratio_to_all_experts"] <= gate_b["max_decode_drive_ratio_to_all_experts"]
    )

    layer_bytes = stream_stage["layer_bytes"]
    capacities = {name: entry["cache_capacity_bytes"] for name, entry in stream_stage["configurations"].items()}
    compact_audits = sum("compact buffers != the served experts" in r["audit"] for r in records)
    decode_compact = max((r["compact_fraction_of_layer"] for r in records if r["phase"] == "decode" and r["configuration"] != "all-experts"), default=None)
    working_set_ok = all(r["expert_peak_bytes"] <= capacities[r["configuration"]] + r["compact_peak_bytes"] for r in records)
    gate_c = {
        "compact_audit_failures": compact_audits,
        "max_decode_compact_fraction_of_layer": decode_compact,
        "max_decode_compact_fraction_of_layer_allowed": gates_config["max_decode_compact_fraction_of_layer"],
        "layer_bytes": layer_bytes,
        "expert_working_set_within_capacity_plus_compact": working_set_ok,
        "max_expert_peak_bytes": {name: max(r["expert_peak_bytes"] for r in records if r["configuration"] == name) for name in expected},
        "cache_capacity_bytes": capacities,
    }
    gate_c["passed"] = compact_audits == 0 and decode_compact is not None and decode_compact <= gate_c["max_decode_compact_fraction_of_layer_allowed"] and working_set_ok

    layering = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests" / "test_layering.py")], capture_output=True, text=True, cwd=REPO_ROOT)
    gate_d = {"test_layering": layering.stdout.strip().splitlines()[-1] if layering.stdout.strip() else layering.stderr[-200:], "passed": layering.returncode == 0}

    summary = {
        "run": str(run),
        "model": f"{config['model']['repository']}@{config['model']['revision'][:8]}",
        "num_prompts": prompts,
        "source_tree_sha256": environment["source_tree_sha256"],
        "python_hash_seed": environment.get("python_hash_seed"),
        "profile_declared": environment.get("profile"),
        "experts": {"layers": layers, "per_layer": per_layer, "bytes": expert_bytes, "row_bytes": stream_stage["expert_row_bytes"][0]},
        "reference_stage": reference_stage,
        "stream_stage": stream_stage,
        "groups": groups,
        "routing": routing_statistics(reference, per_layer),
        "profile": profile_summary(run / "profile.json"),
        "gates": {"correctness": correctness, "A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d},
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
