"""Aggregate a Phase 1 run and evaluate the Phase 1 gate.

    uv run python benchmarks/report.py experiments/phase1/<run> [--compare experiments/phase1/<other run>]

Writes summary.json and summary.md into the run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from awpmi.tracing import read_jsonl

GAP_BUCKETS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def gap_bucket(gap: float) -> str:
    for low, high in zip(GAP_BUCKETS, GAP_BUCKETS[1:]):
        if low <= gap < high:
            return f"[{low:g}, {high:g})"
    raise ValueError(gap)


def summarize_config(records: list[dict], gaps: dict[int, float]) -> dict:
    n = len(records)
    certified = [r for r in records if r["certified"]]
    early = [r for r in certified if r["pages_materialized"] < r["pages_total"]]
    by_gap: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_gap[gap_bucket(gaps[record["prompt_id"]])].append(record)
    return {
        "inputs": n,
        "pages_total": records[0]["pages_total"],
        "certified_mismatches": sum(r["awpmi_token"] != r["reference_token"] for r in certified),
        "fallback_mismatches": sum(r["awpmi_token"] != r["reference_token"] for r in records if r["fallback"]),
        "certificate_coverage": len(certified) / n,
        "certified_early": len(early),
        "certified_early_rate": len(early) / n,
        "certified_at_full_materialization": len(certified) - len(early),
        "fallback_rate": sum(r["fallback"] for r in records) / n,
        "materialized_fraction": quantiles([r["materialized_fraction"] for r in records]),
        "materialized_fraction_when_certified_early": quantiles([r["materialized_fraction"] for r in early]),
        "bytes_fraction_mean": float(np.mean([r["bytes_materialized"] / r["bytes_total"] for r in records])),
        "certification_points": {
            str(k): v for k, v in sorted(Counter(r["pages_materialized"] for r in certified).items())
        },
        "fallbacks": sum(r["fallback"] for r in records),
        "early_rate_by_reference_gap": {
            bucket: {
                "inputs": len(rows),
                "certified_early_rate": sum(r["certified"] and r["pages_materialized"] < r["pages_total"] for r in rows)
                / len(rows),
                "mean_materialized_fraction": float(np.mean([r["materialized_fraction"] for r in rows])),
            }
            for bucket, rows in sorted(by_gap.items(), key=lambda item: float(item[0][1:].split(",")[0]))
        },
    }


def summarize_validation(rows: list[dict]) -> dict:
    return {
        "inputs": len(rows),
        "prefix_mismatches": sum(not r["prefix_bitwise_equal"] for r in rows),
        "fallback_not_bitwise_equal": sum(not r["fallback_bitwise_equal"] for r in rows),
        "forced_fallback_token_mismatches": sum(r["fallback_token"] != r["reference_token"] for r in rows),
        "envelope_violations": sum(r["envelope_violations"] for r in rows),
        "max_abs_partial_minus_reference": max(r["max_abs_partial_minus_reference"] for r in rows),
        "max_envelope_halfwidth": max(r["max_envelope_halfwidth"] for r in rows),
    }


def evaluate(run: Path, compare: Path | None) -> dict:
    records = read_jsonl(run / "records.jsonl")
    validation = read_jsonl(run / "validation.jsonl")
    prompts = read_jsonl(run / "prompts.jsonl")
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    digest = json.loads((run / "digest.json").read_text(encoding="utf-8"))

    gaps = {row["prompt_id"]: row["reference_top2_gap"] for row in validation}
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["page_width"], record["scheduler"])].append(record)
    configs = {
        f"width={width}/{scheduler}": summarize_config(rows, gaps) for (width, scheduler), rows in sorted(groups.items())
    }
    validation_by_width: dict[int, list[dict]] = defaultdict(list)
    for row in validation:
        validation_by_width[row["page_width"]].append(row)
    validations = {f"width={w}": summarize_validation(rows) for w, rows in sorted(validation_by_width.items())}

    unique_gaps = [gaps[p["prompt_id"]] for p in prompts]
    reference = {
        "top2_gap": quantiles(unique_gaps),
        "bf16_top1_ties": sum(g == 0.0 for g in unique_gaps),
    }

    expected = len(prompts) * len(environment["page_layout"]) * len({r["scheduler"] for r in records})
    certified_mismatches = sum(c["certified_mismatches"] for c in configs.values())
    fallback_mismatches = sum(c["fallback_mismatches"] for c in configs.values()) + sum(
        v["forced_fallback_token_mismatches"] for v in validations.values()
    )
    reproduces = all(
        v["prefix_mismatches"] == 0 and v["fallback_not_bitwise_equal"] == 0 and v["envelope_violations"] == 0
        for v in validations.values()
    )
    early_total = sum(c["certified_early"] for c in configs.values())

    if compare is None:
        reproducible = None
    else:
        other = json.loads((compare / "digest.json").read_text(encoding="utf-8"))
        reproducible = other["records_sha256"] == digest["records_sha256"] and (
            other["validation_sha256"] == digest["validation_sha256"]
        )
    gate = {
        "run_complete": len(records) == expected and not (run / "failure.json").exists(),
        "certified_mismatches_zero": certified_mismatches == 0,
        "fallback_mismatches_zero": fallback_mismatches == 0,
        "full_materialization_reproduces_reference": reproduces,
        "certificate_triggers_before_final_page_on_some_inputs": early_total > 0,
        "results_reproducible": reproducible,
    }
    gate["passed"] = all(value is True for value in gate.values())

    best_name, best = min(configs.items(), key=lambda item: item[1]["materialized_fraction"]["mean"])
    diagnosis = {
        "best_config_by_mean_materialized_fraction": best_name,
        "best_mean_materialized_fraction": best["materialized_fraction"]["mean"],
        "best_certified_early_rate": best["certified_early_rate"],
        "certification_mostly_near_full_materialization": best["materialized_fraction"]["mean"] > 0.9,
    }
    return {
        "run": str(run),
        "compared_with": str(compare) if compare else None,
        "inputs": len(prompts),
        "records": len(records),
        "digest": digest,
        "gate": gate,
        "diagnosis": diagnosis,
        "reference": reference,
        "validation": validations,
        "page_layout": environment["page_layout"],
        "configs": configs,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(summary: dict) -> str:
    lines = [f"# Phase 1 summary — `{Path(summary['run']).name}`", ""]
    lines += [f"Inputs: {summary['inputs']} prompts, {summary['records']} AWPMI runs.", ""]
    lines += ["## Gate", "", "| criterion | result |", "|---|---|"]
    for key, value in summary["gate"].items():
        lines.append(f"| {key} | {'NOT EVALUATED' if value is None else ('PASS' if value else 'FAIL')} |")
    if summary["diagnosis"]["certification_mostly_near_full_materialization"]:
        lines += [
            "",
            "**Roadmap check: certification occurs almost exclusively near 100% materialization "
            f"(best mean materialized fraction {fmt(summary['diagnosis']['best_mean_materialized_fraction'])}). "
            "Do not implement streaming; improve bounds, page decomposition and ordering first.**",
        ]
    lines += ["", "## Configurations", ""]
    lines += [
        "| config | coverage | early | fallback | mean frac | median | p90 | p95 | cert. mism. | fb. mism. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary["configs"].items():
        f = c["materialized_fraction"]
        lines.append(
            f"| {name} | {fmt(c['certificate_coverage'])} | {fmt(c['certified_early_rate'])} | {fmt(c['fallback_rate'])} "
            f"| {fmt(f['mean'])} | {fmt(f['median'])} | {fmt(f['p90'])} | {fmt(f['p95'])} "
            f"| {c['certified_mismatches']} | {c['fallback_mismatches']} |"
        )
    lines += ["", "## Certification points (pages materialized when certified)", ""]
    for name, c in summary["configs"].items():
        points = ", ".join(f"{k}: {v}" for k, v in c["certification_points"].items()) or "none"
        lines.append(f"- {name} (of {c['pages_total']}): {points}; fallbacks: {c['fallbacks']}")
    lines += ["", "## Early certification by reference top-2 gap", ""]
    for name, c in summary["configs"].items():
        cells = ", ".join(
            f"{bucket} n={b['inputs']} early={fmt(b['certified_early_rate'])}"
            for bucket, b in c["early_rate_by_reference_gap"].items()
        )
        lines.append(f"- {name}: {cells}")
    lines += ["", "## Full-materialization validation", "", "```json", json.dumps(summary["validation"], indent=2), "```"]
    lines += ["", "## Page layout", "", "```json", json.dumps(summary["page_layout"], indent=2), "```"]
    lines += ["", "## Reference", "", "```json", json.dumps(summary["reference"], indent=2), "```"]
    lines += ["", "## Diagnosis", "", "```json", json.dumps(summary["diagnosis"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="second run of the same config, for reproducibility")
    args = parser.parse_args()
    summary = evaluate(args.run, args.compare)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.run / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
