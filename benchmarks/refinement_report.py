"""Aggregate a Phase 1B run, compare it with Phase 1A and classify every decomposition.

    uv run python benchmarks/refinement_report.py experiments/phase1b/<run> [--compare experiments/phase1b/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff the
correctness gate passes.

Primary configuration (decision 0003): realistic bound, lowest_index ties,
row_selective mode, conservative fallback (a fallback reads every BF16 row not yet
read). Fractions are effective bytes to a decision, including all resident
metadata, divided by the original BF16 LM-head bytes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from awpmi.tracing import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ("row_selective", "realistic", "lowest_index")
IDEAL = ("row_selective", "ideal", "lowest_index")
GAP_BUCKETS = (0.0, 0.5, 1.0, 2.0, 4.0, float("inf"))
BYTE_PARTS = ("metadata", "base_payload", "base_scales", "refinement_payload", "refinement_scales", "exact_rows", "fallback_rows")


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


def gap_bucket(gap: float) -> str:
    for low, high in zip(GAP_BUCKETS, GAP_BUCKETS[1:]):
        if low <= gap < high:
            return f"[{low:g}, {high:g})"
    raise ValueError(gap)


def summarize(records: list[dict], states: list[str], weight_bytes: int, gaps: dict[int, float]) -> dict:
    n = len(records)
    certified = [r for r in records if r["certified"]]
    masked_known = [r for r in records if r["fallback"] and r["masked_fallback_exact"] is not None]
    # Level names can repeat (q4+q4), so states are labelled by position.
    labels = [f"{index}:{name}" for index, name in enumerate(states)]
    by_state = {label: 0 for label in labels}
    for record in certified:
        by_state[labels[record["decision_state_index"]]] += 1
    cumulative, running = {}, 0
    for label in labels:
        running += by_state[label]
        cumulative[label] = running / n
    by_gap: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_gap[gap_bucket(gaps[record["prompt_id"]])].append(record)
    return {
        "inputs": n,
        "certified_mismatches": sum(r["token"] != r["reference_token"] for r in certified),
        "fallback_mismatches": sum(r["token"] != r["reference_token"] for r in records if r["fallback"]),
        "certificate_coverage": len(certified) / n,
        "fallback_rate": 1 - len(certified) / n,
        "certified_by_state": {name: count / n for name, count in by_state.items()},
        "certified_by_state_cumulative": cumulative,
        "fraction": quantiles(r["fraction"] for r in records),
        "fraction_when_certified": quantiles(r["fraction"] for r in certified),
        "fraction_masked_fallback": quantiles(r["fraction_masked_fallback"] for r in records),
        "masked_fallback_checks": len(masked_known),
        "masked_fallback_mismatches": sum(not r["masked_fallback_exact"] for r in masked_known),
        "io_4k_fraction_mean": float(np.mean([r["io_bytes_4k"] / weight_bytes for r in records])),
        "bytes_mean_fraction": {
            part: float(np.mean([r["bytes"][part] / weight_bytes for r in records])) for part in BYTE_PARTS
        },
        "contenders_after_base": quantiles(r["contenders"][0] for r in records),
        "rows_reaching_exact": quantiles(r["rows_loaded"][-1] for r in records),
        "by_reference_gap": {
            bucket: {
                "inputs": len(rows),
                "coverage": sum(r["certified"] for r in rows) / len(rows),
                "mean_fraction": float(np.mean([r["fraction"] for r in rows])),
            }
            for bucket, rows in sorted(by_gap.items(), key=lambda item: float(item[0][1:].split(",")[0]))
        },
    }


def phase1a_baseline(source_run: Path) -> dict:
    summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    weight_bytes = next(iter(summary["page_layout"].values()))["bytes_total"]
    effective = {}
    for name, config in summary["configs"].items():
        width = name.split("/")[0].split("=")[1]
        metadata = summary["page_layout"][width]["metadata_bytes"]
        effective[name] = config["bytes_fraction_mean"] + metadata / weight_bytes
    best_effective = min(effective, key=effective.get)
    return {
        "source_run": source_run.relative_to(REPO_ROOT).as_posix(),
        "best_config": summary["diagnosis"]["best_config_by_mean_materialized_fraction"],
        "best_mean_fraction_pages_only": summary["diagnosis"]["best_mean_materialized_fraction"],
        "best_mean_fraction_with_metadata": effective[summary["diagnosis"]["best_config_by_mean_materialized_fraction"]],
        "best_effective_config": best_effective,
        "best_effective_mean_fraction": effective[best_effective],
        "certificate_coverage": summary["configs"][summary["diagnosis"]["best_config_by_mean_materialized_fraction"]][
            "certificate_coverage"
        ],
    }


def classify(primary: dict, ideal: dict, gate: dict) -> dict:
    real, best = primary["fraction"]["mean"], ideal["fraction"]["mean"]
    real_saving, ideal_saving = 1 - real, 1 - best
    share = real_saving / ideal_saving if ideal_saving > 0 else 0.0
    if best > gate["decomposition_limited_ideal_fraction"]:
        verdict, reason = "NOT PROMISING", "decomposition-limited: even the ideal bound needs most of the BF16 bytes"
    elif real <= gate["promising_max_fraction"] and share >= gate["promising_min_ideal_saving_share"]:
        verdict, reason = "PROMISING", "large realistic saving, within reach of the ideal bound"
    else:
        verdict, reason = "NOT PROMISING", "bound-limited: the ideal bound saves bytes but the realistic bound does not"
    return {
        "verdict": verdict,
        "reason": reason,
        "realistic_mean_fraction": real,
        "ideal_mean_fraction": best,
        "realistic_share_of_ideal_saving": share,
    }


def evaluate(run: Path, compare: Path | None) -> dict:
    settings = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    decompositions = json.loads((run / "decompositions.json").read_text(encoding="utf-8"))
    digest = json.loads((run / "digest.json").read_text(encoding="utf-8"))
    reference = read_jsonl(run / "reference.jsonl")
    validation = read_jsonl(run / "validation.jsonl")
    records = read_jsonl(run / "records.jsonl.gz")
    gaps = {row["prompt_id"]: row["reference_top2_gap"] for row in reference}

    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["decomposition"], record["mode"], record["bound"], record["tie_break"])].append(record)
    configs = {}
    for (spec, mode, bound, tie), rows in groups.items():
        info = decompositions[spec]
        configs[f"{spec}/{mode}/{bound}/{tie}"] = {
            "decomposition": spec,
            "mode": mode,
            "bound": bound,
            "tie_break": tie,
            **summarize(rows, info["states"], info["weight_bytes"], gaps),
        }

    def lookup(spec: str, mode: str, bound: str, tie: str) -> dict:
        return configs[f"{spec}/{mode}/{bound}/{tie}"]

    specs = list(decompositions)
    row_selective_savings = {}
    for spec in specs:
        for bound, tie in settings["configurations"]:
            whole = lookup(spec, "global", bound, tie)["fraction"]["mean"]
            selective = lookup(spec, "row_selective", bound, tie)["fraction"]["mean"]
            row_selective_savings[f"{spec}/{bound}/{tie}"] = {
                "global_mean_fraction": whole,
                "row_selective_mean_fraction": selective,
                "relative_saving": 1 - selective / whole,
            }
    classification = {spec: classify(lookup(spec, *PRIMARY), lookup(spec, *IDEAL), settings["gate"]) for spec in specs}
    promising = sorted(
        (spec for spec, c in classification.items() if c["verdict"] == "PROMISING"),
        key=lambda spec: (classification[spec]["realistic_mean_fraction"], len(decompositions[spec]["states"])),
    )

    expected = len(reference) * len(decompositions) * len(settings["modes"]) * len(settings["configurations"])
    envelope = sum(sum(row["envelope_violations"].values()) for row in validation)
    reproducible = None
    if compare is not None:
        other = json.loads((compare / "digest.json").read_text(encoding="utf-8"))
        reproducible = all(other[key] == digest[key] for key in digest)
    gate = {
        "run_complete": len(records) == expected and not (run / "failure.json").exists(),
        "certified_mismatches_zero": sum(c["certified_mismatches"] for c in configs.values()) == 0,
        "fallback_mismatches_zero": sum(c["fallback_mismatches"] for c in configs.values()) == 0,
        "reference_prefix_and_fallback_bitwise": all(
            row["prefix_bitwise_equal"] and row["fallback_bitwise_equal"] and row["fallback_token"] == row["reference_token"]
            for row in reference
        ),
        "envelope_violations_zero": envelope == 0,
        "reconstruction_exact": all(info["reconstruction_exact"] for info in decompositions.values()),
        "results_reproducible": reproducible,
    }
    gate["passed"] = all(value is True for value in gate.values())

    unique_gaps = list(gaps.values())
    return {
        "run": run.as_posix(),
        "compared_with": compare.as_posix() if compare else None,
        "inputs": len(reference),
        "records": len(records),
        "digest": digest,
        "environment": {key: environment.get(key) for key in ("gpu", "packages", "source_tree_sha256", "git_commit")},
        "gate": gate,
        "decision_gate_thresholds": settings["gate"],
        "baseline_phase1a": phase1a_baseline(REPO_ROOT / settings["source_run"]),
        "reference": {
            "top2_gap": quantiles(unique_gaps),
            "bf16_top1_ties": sum(g == 0.0 for g in unique_gaps),
        },
        "masked_fallback": {
            "checks": sum(row["masked_fallback_checks"] for row in validation),
            "mismatches": sum(row["masked_fallback_mismatches"] for row in validation),
        },
        "decompositions": decompositions,
        "classification": classification,
        "recommended": promising[0] if promising else None,
        "row_selective_savings": row_selective_savings,
        "configs": configs,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f} %"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(summary: dict) -> str:
    base = summary["baseline_phase1a"]
    lines = [f"# Phase 1B summary — `{Path(summary['run']).name}`", ""]
    lines += [f"Inputs: {summary['inputs']} prompts, {summary['records']} simulated runs.", ""]
    lines += ["## Correctness gate", "", "| criterion | result |", "|---|---|"]
    for key, value in summary["gate"].items():
        lines.append(f"| {key} | {'NOT EVALUATED' if value is None else ('PASS' if value else 'FAIL')} |")
    lines += [
        "",
        f"Masked-fallback diagnostic: {summary['masked_fallback']['checks']} checks, "
        f"{summary['masked_fallback']['mismatches']} mismatches.",
        "",
        "## Phase 1A baseline",
        "",
        f"- Best Phase 1A configuration `{base['best_config']}`: mean fraction "
        f"{fmt(base['best_mean_fraction_pages_only'])} (pages only), "
        f"{fmt(base['best_mean_fraction_with_metadata'])} with its resident metadata.",
        f"- Lowest effective Phase 1A fraction: `{base['best_effective_config']}` at "
        f"{fmt(base['best_effective_mean_fraction'])}.",
        "",
        "## Classification (primary configuration: row_selective, realistic, lowest_index)",
        "",
        "| decomposition | realistic mean | ideal mean | share of ideal saving | verdict |",
        "|---|---|---|---|---|",
    ]
    for spec, c in summary["classification"].items():
        lines.append(
            f"| {spec} | {fmt(c['realistic_mean_fraction'])} | {fmt(c['ideal_mean_fraction'])} "
            f"| {fmt(c['realistic_share_of_ideal_saving'])} | {c['verdict']} ({c['reason'].split(':')[0]}) |"
        )
    lines += ["", f"Recommended: **{summary['recommended']}**", ""]
    lines += ["## All configurations", ""]
    lines += [
        "| config | coverage | mean | median | p90 | p95 | masked-fb mean | io 4K mean | contenders after base (median) | cert. mism. | fb. mism. |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary["configs"].items():
        f = c["fraction"]
        lines.append(
            f"| {name} | {pct(c['certificate_coverage'])} | {fmt(f['mean'])} | {fmt(f['median'])} | {fmt(f['p90'])} "
            f"| {fmt(f['p95'])} | {fmt(c['fraction_masked_fallback']['mean'])} | {fmt(c['io_4k_fraction_mean'])} "
            f"| {c['contenders_after_base']['median']:.0f} | {c['certified_mismatches']} | {c['fallback_mismatches']} |"
        )
    lines += ["", "## Byte breakdown (mean fraction of the BF16 LM head)", ""]
    lines += ["| config | " + " | ".join(BYTE_PARTS) + " |", "|---|" + "---|" * len(BYTE_PARTS)]
    for name, c in summary["configs"].items():
        if c["mode"] == "row_selective":
            lines.append(f"| {name} | " + " | ".join(fmt(c["bytes_mean_fraction"][p]) for p in BYTE_PARTS) + " |")
    lines += ["", "## Certified after each state (cumulative)", ""]
    for name, c in summary["configs"].items():
        if c["mode"] == "row_selective":
            cells = ", ".join(f"{state}: {pct(v)}" for state, v in c["certified_by_state_cumulative"].items())
            lines.append(f"- {name}: {cells}")
    lines += ["", "## Primary configuration by reference top-2 gap (coverage, mean fraction)", ""]
    for name, c in summary["configs"].items():
        if (c["mode"], c["bound"], c["tie_break"]) == PRIMARY:
            cells = ", ".join(
                f"{bucket} n={b['inputs']}: {pct(b['coverage'])}, {fmt(b['mean_fraction'])}"
                for bucket, b in c["by_reference_gap"].items()
            )
            lines.append(f"- {c['decomposition']}: {cells}")
    lines += ["", "## Row-selective savings over global refinement", ""]
    lines += ["| decomposition / bound / tie | global mean | row-selective mean | relative saving |", "|---|---|---|---|"]
    for name, s in summary["row_selective_savings"].items():
        lines.append(
            f"| {name} | {fmt(s['global_mean_fraction'])} | {fmt(s['row_selective_mean_fraction'])} | {pct(s['relative_saving'])} |"
        )
    lines += ["", "## Reference", "", "```json", json.dumps(summary["reference"], indent=2), "```", ""]
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
