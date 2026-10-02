"""Aggregate a Phase 3 storage run and evaluate its gates.

    uv run python benchmarks/storage_report.py experiments/phase3/<run> [--compare experiments/phase3/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff gate A (physical
selective materialization) and gate B (real byte reduction) pass and the Phase 1C assumption
is not invalidated (configs/phase3-storage.yaml). Fractions are bytes per token over the BF16
LM-head bytes: `logical` is the store's log as in Phase 1C (resident metadata included),
`drive` the bytes read from the drive, `host` the bytes gathered from host memory, `h2d` the
bytes copied host-to-device.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def stage_groups(timings: dict[str, float]) -> dict[str, float]:
    """Stage times of one run, grouped: reads (storage + transfer, waited for), decode, bounds, certificate."""
    groups = defaultdict(float)
    for stage, value in timings.items():
        if stage == "total":
            continue
        if stage.endswith("read"):
            groups["read"] += value
        elif stage == "0:matvec":
            groups["decode_and_matvec"] += value
        elif stage.endswith(":certify"):
            groups["certificate"] += value
        elif stage in ("0:bounds",) or stage.endswith(":refine") or stage.endswith(":exact"):
            groups["bounds"] += value
        elif stage == "linear":
            groups["gemv"] += value
        else:
            groups[stage] += value
    groups["total"] = timings.get("total", sum(groups.values()))
    return dict(groups)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    head = records[0]["fallback_mode"] is not None
    summary = {"runs": n, "tier": records[0]["tier"]}
    if head:
        certified = [r for r in records if r["certified"]]
        summary.update(
            certified_mismatches=sum(r["token"] != r["reference_token"] for r in certified),
            fallback_mismatches=sum(r["token"] != r["reference_token"] for r in records if not r["certified"]),
            coverage=len(certified) / n,
            masked_guard_trips=sum(r["masked_guard_tripped"] for r in records),
        )
    else:
        summary.update(
            token_mismatches=sum(r["token"] != r["reference_token"] for r in records),
            logits_bitwise=sum(r["logits_bitwise_equal"] for r in records),
        )
    served = sum(r["storage"]["logical_bytes"] for r in records)
    drive = sum(r["physical_fraction"] for r in records)
    summary["fraction"] = {
        "logical": quantiles(r["fraction"] for r in records),
        "drive": quantiles(r["physical_fraction"] for r in records),
        "host": quantiles(r["host_memory_fraction"] for r in records),
        "h2d": quantiles(r["h2d_fraction"] for r in records),
        "blocks_4k": quantiles(r["blocks_4k_fraction"] for r in records),
    }
    weight_bytes = None
    for r in records:
        if r["fraction"] and r["storage"]["logical_bytes"]:
            weight_bytes = r["bytes"]["total"] / r["fraction"] if "bytes" in r else None
            break
    drive_bytes = sum(r["storage"]["physical_bytes"] for r in records) if records[0]["tier"] in ("direct", "buffered") else 0
    summary["amplification"] = drive_bytes / served if drive_bytes and served else None
    summary["reads"] = {
        "requests": quantiles(r["storage"]["requests"] for r in records),
        "read_calls": quantiles(r["storage"]["read_calls"] for r in records),
        "extents": quantiles(r["storage"]["extents"] for r in records),
        "h2d_copies": quantiles(r["transfer"]["h2d_copies"] for r in records),
    }
    by_segment: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in records:
        for segment, entry in r["storage"]["by_segment"].items():
            for key, value in entry.items():
                by_segment[segment][key] += value / n
    summary["by_segment_mean"] = {segment: dict(values) for segment, values in sorted(by_segment.items())}
    if any(r.get("cache") for r in records):
        summary["cache"] = {
            key: float(np.mean([r["cache"][key] for r in records]))
            for key in ("hits", "misses", "hit_bytes", "miss_bytes", "bypassed")
        }
    groups = [stage_groups(r["timings_ms"]) for r in records]
    keys = sorted({key for group in groups for key in group})
    summary["time_ms"] = {key: quantiles(group.get(key, 0.0) for group in groups) for key in keys}
    summary["system"] = {
        "io_ms": quantiles(r["system"]["io_ms"] for r in records),
        "copy_ms": quantiles(r["system"]["copy_ms"] for r in records),
        "peak_vram_bytes": quantiles(r["system"]["peak_vram_bytes"] for r in records),
        "process_resident_bytes": quantiles(r["system"]["process_resident_bytes"] or 0 for r in records),
        "process_peak_resident_bytes_max": max(r["system"]["process_peak_resident_bytes"] or 0 for r in records),
    }
    del drive, weight_bytes
    return summary


def correctness(run: Path, config: dict, records: list[dict], expected_prompts: int) -> dict:
    references = read_jsonl(run / "reference.jsonl")
    validation = read_jsonl(run / "validation.jsonl")
    pack = json.loads((run / "pack.json").read_text(encoding="utf-8"))
    failures = json.loads((run / "failure.json").read_text(encoding="utf-8")) if (run / "failure.json").exists() else []
    head = [r for r in records if r["fallback_mode"] is not None]
    full_head = [r for r in records if r["fallback_mode"] is None]
    checks = {
        "prompts_completed": len(references) == expected_prompts and len(validation) == expected_prompts,
        "hard_failures": len(failures),
        "certified_mismatches": sum(r["token"] != r["reference_token"] for r in head if r["certified"]),
        "fallback_mismatches": sum(r["token"] != r["reference_token"] for r in head if not r["certified"]),
        "reference_prefix_and_gemv_bitwise": sum(r["prefix_bitwise_equal"] and r["fallback_bitwise_equal"] for r in references),
        "full_head_gemv_bitwise": sum(r["logits_bitwise_equal"] for r in full_head),
        "full_head_runs": len(full_head),
        "envelope_violations": sum(sum(v["envelope_violations"].values()) for v in validation),
        "coarse_arithmetic_violations": sum(v["coarse_arithmetic_violations"] for v in validation),
        "coarse_arithmetic_max_ratio": max(v["coarse_arithmetic_max_ratio"] for v in validation),
        "masked_fallbacks_checked": sum(bool(v.get("masked_fallback_checked")) for v in validation),
        "parity_failures": {},
        "storage_audit_failures": {},
        "pack": {key: pack[key] for key in ("levels_match_phase1c", "store_matches_resident", "store_matches_phase1c")},
        "masked_self_tests": pack["masked_fallback_self_test"],
    }
    for entry in validation:
        for key, fields in entry["parity"].items():
            checks["parity_failures"].setdefault(key, 0)
            checks["parity_failures"][key] += bool(fields)
        for key, problems in entry["storage_audit"].items():
            checks["storage_audit_failures"].setdefault(key, 0)
            checks["storage_audit_failures"][key] += bool(problems)
    checks["passed"] = (
        checks["prompts_completed"]
        and checks["hard_failures"] == 0
        and checks["certified_mismatches"] == 0
        and checks["fallback_mismatches"] == 0
        and checks["reference_prefix_and_gemv_bitwise"] == len(references)
        and checks["full_head_gemv_bitwise"] == len(full_head)
        and checks["envelope_violations"] == 0
        and checks["coarse_arithmetic_violations"] == 0
        and not any(checks["parity_failures"].values())
        and not any(checks["storage_audit_failures"].values())
        and all(checks["pack"].values())
    )
    return checks


def format_fraction(q: dict) -> str:
    return "–" if not q else f"{q['mean']:.3f} / {q['median']:.3f} / {q['p95']:.3f}"


def render(summary: dict) -> str:
    gates = summary["gates"]
    lines = [
        f"# Phase 3 storage run: {summary['run']}",
        "",
        f"Gate A (physical selective materialization): **{'PASS' if gates['A']['passed'] else 'FAIL'}** · "
        f"Gate B (real byte reduction): **{'PASS' if gates['B']['passed'] else 'FAIL'}** · "
        f"Phase 1C assumption: **{'INVALIDATED' if gates['invalidated'] else 'holds'}**",
        "",
        "## Bytes per token (fraction of the BF16 LM head: mean / median / p95)",
        "",
        "| Configuration | Logical | Drive | Host memory | Host→device | 4 KiB blocks | Amplification | Read calls (mean) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, group in summary["groups"].items():
        f = group["fraction"]
        amplification = "–" if group["amplification"] is None else f"{group['amplification']:.2f}×"
        lines.append(
            f"| {key} | {format_fraction(f['logical'])} | {format_fraction(f['drive'])} | {format_fraction(f['host'])} | "
            f"{format_fraction(f['h2d'])} | {format_fraction(f['blocks_4k'])} | {amplification} | {group['reads']['read_calls']['mean']:.1f} |"
        )
    lines += ["", "## Time per token (ms: mean / median / p95)", "", "| Configuration | Total | Read (I/O + copy, waited) | Drive I/O | Copy (device) | Decode + matvec | Bounds | Certificate |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, group in summary["groups"].items():
        t = group["time_ms"]

        def cell(name):
            q = t.get(name)
            return "–" if not q else f"{q['mean']:.1f} / {q['median']:.1f} / {q['p95']:.1f}"

        s = group["system"]
        lines.append(
            f"| {key} | {cell('total')} | {cell('read')} | {s['io_ms']['mean']:.1f} | {s['copy_ms']['mean']:.2f} | "
            f"{cell('decode_and_matvec') if 'decode_and_matvec' in t else cell('gemv')} | {cell('bounds')} | {cell('certificate')} |"
        )
    lines += ["", "## Memory", "", "| Configuration | Device resident (MB) | Host resident (MB) | Peak device memory per token (MB, mean / max) |", "| --- | --- | --- | --- |"]
    for key, group in summary["groups"].items():
        name = key.split("/")[0]
        resident = summary["resident_bytes"].get(name, {})
        peak = group["system"]["peak_vram_bytes"]
        lines.append(f"| {key} | {resident.get('device', 0) / 1e6:.1f} | {resident.get('host', 0) / 1e6:.1f} | {peak['mean'] / 1e6:.1f} / {peak['max'] / 1e6:.1f} |")
    c = summary["correctness"]
    lines += [
        "",
        "## Correctness (gate A)",
        "",
        f"- Prompts completed: {c['prompts_completed']}; hard failures: {c['hard_failures']}",
        f"- Certified mismatches: {c['certified_mismatches']}; fallback mismatches: {c['fallback_mismatches']}",
        f"- Reference prefix and GEMV bitwise: {c['reference_prefix_and_gemv_bitwise']}; full-head GEMV from storage bitwise: {c['full_head_gemv_bitwise']} of {c['full_head_runs']}",
        f"- Envelope violations: {c['envelope_violations']}; coarse-arithmetic violations: {c['coarse_arithmetic_violations']} (largest ratio {c['coarse_arithmetic_max_ratio']:.2e})",
        f"- Parity failures (B against Phase 1C, C against B): {c['parity_failures']}",
        f"- Storage audit failures: {c['storage_audit_failures']}",
        f"- Pack: {c['pack']}",
        "",
        "## Gates",
        "",
        f"- A: {gates['A']}",
        f"- B: {gates['B']}",
        "",
        "## Storage micro-benchmarks (profile.json)",
        "",
        "```json",
        json.dumps(summary["profile"], indent=2),
        "```",
    ]
    if summary.get("compare"):
        lines += ["", "## Reproducibility", "", f"{summary['compare']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None, help="a second run of the same source tree: its digests must match")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    pack = json.loads((run / "pack.json").read_text(encoding="utf-8"))
    records = read_jsonl(run / "records.jsonl.gz")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = record["configuration"] if record["fallback_mode"] is None else f"{record['configuration']}/{record['fallback_mode']}"
        grouped[key].append(record)
    groups = {key: summarize(rows) for key, rows in grouped.items()}
    checks = correctness(run, config, records, environment["num_prompts"])
    gate = config["gate"]
    primary_key = f"{config['primary']['configuration']}/{config['primary']['fallback']}"
    primary, baseline = groups[primary_key], groups[config["baseline"]]
    drive, h2d = primary["fraction"]["drive"]["mean"], primary["fraction"]["h2d"]["mean"]
    gate_b = {
        "primary": primary_key,
        "mean_drive_fraction": drive,
        "mean_h2d_fraction": h2d,
        "baseline_mean_drive_fraction": baseline["fraction"]["drive"]["mean"],
        "baseline_mean_h2d_fraction": baseline["fraction"]["h2d"]["mean"],
        "max_mean_physical_fraction": gate["max_mean_physical_fraction"],
        "max_mean_h2d_fraction": gate["max_mean_h2d_fraction"],
        "passed": drive <= gate["max_mean_physical_fraction"] and h2d <= gate["max_mean_h2d_fraction"],
    }
    summary = {
        "run": str(run),
        "num_prompts": environment["num_prompts"],
        "source_tree_sha256": environment["source_tree_sha256"],
        "disk": environment.get("storage", {}).get("disk"),
        "resident_bytes": pack["resident_bytes"],
        "groups": groups,
        "correctness": checks,
        "profile": json.loads((run / "profile.json").read_text(encoding="utf-8")) if (run / "profile.json").exists() else None,
        "gates": {
            "A": {"passed": checks["passed"]},
            "B": gate_b,
            "invalidated": drive > gate["invalidation_physical_fraction"],
        },
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
    passed = summary["gates"]["A"]["passed"] and gate_b["passed"] and not summary["gates"]["invalidated"]
    if summary.get("compare"):
        passed = passed and all(summary["compare"]["digests_equal"].values()) and summary["compare"]["same_source_tree"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
