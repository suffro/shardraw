"""Aggregate a Phase 2 run: correctness, materialization curves, bound provenance, gate and decision.

    uv run python benchmarks/suffix_report.py experiments/phase2/<run> [--compare experiments/phase2/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff the gate passes.
Fractions are effective bytes to a decision (all resident metadata, the final norm's weight, the
suffix pages and the LM-head reads, including every fallback) divided by the BF16 bytes of the
adaptive region (the stage's adaptive MLP roles plus the LM head). The baseline of a stage reads
its whole suffix and runs Phase 1C on the exact h.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from awpmi.tracing import read_jsonl

TERMS = (
    "unread_neurons",
    "down_accumulation",
    "o_rounding",
    "y_rounding",
    "norm_rounding",
    "lm_accumulation",
)


def quantiles(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def cell_summary(records: list[dict], certified_model: bool) -> dict:
    n = len(records)
    paths = defaultdict(int)
    for record in records:
        paths[record["path"]] += 1
    suffix_certified = sum(record["would_certify"] for record in records)
    summary = {
        "runs": n,
        "suffix_coverage": suffix_certified / n,
        "paths": dict(paths),
        "wrong_would_certify": sum(r["would_certify"] and r["token"] != r["reference_token"] for r in records),
        "h_relative_width_median": float(np.median([r["widths"]["h"] for r in records if "widths" in r] or [np.nan])),
        "y_relative_width_median": float(np.median([r["widths"]["y"] for r in records if "widths" in r] or [np.nan])),
    }
    reasons = defaultdict(int)
    for record in records:
        if not record["would_certify"]:
            reasons[record["reason"] or "none"] += 1
    summary["uncertified_reasons"] = dict(reasons)
    if certified_model:
        region = records[0]["region_bytes"]
        totals = [r["bytes_total"] for r in records]
        baselines = [r.get("baseline_bytes", r["bytes_total"]) for r in records]  # depth 0 is its own baseline
        summary.update(
            coverage=sum(r["certified"] for r in records) / n,
            fraction=quantiles(r["fraction"] for r in records),
            mean_bytes=float(np.mean(totals)),
            baseline_mean_bytes=float(np.mean(baselines)),
            mean_saving_fraction=float((np.mean(baselines) - np.mean(totals)) / region),
            certified_with_unread_suffix=sum(
                r["path"] in ("interval", "pairwise") and r["certified"] and r["budget"] < 1.0 for r in records
            ),
            head_fallbacks=sum(1 for r in records if r.get("head_fallback")),
            head_guard_trips=sum(1 for r in records if r.get("head_guard_tripped")),
            mismatches=sum(r["token"] != r["reference_token"] for r in records),
        )
    pairwise = [r["pairwise"] for r in records if "pairwise" in r and r["pairwise"]["terms"]]
    if pairwise:
        summary["pairwise"] = {
            "runs": len(pairwise),
            "certified": sum(p["certified"] for p in pairwise),
            "contenders_median": float(np.median([p["contenders"] for p in pairwise])),
            # Mean share of each uncertainty term in the tightest pair's total uncertainty.
            "term_shares": term_shares(pairwise),
        }
    return summary


def term_shares(pairwise: list[dict]) -> dict[str, float]:
    shares = defaultdict(list)
    for entry in pairwise:
        terms = entry["terms"]
        total = sum(abs(terms.get(name, 0.0)) for name in TERMS)
        if total > 0:
            for name in TERMS:
                shares[name].append(abs(terms.get(name, 0.0)) / total)
    return {name: float(np.mean(values)) for name, values in shares.items()}


def evaluate(run: Path, compare: Path | None) -> dict:
    settings = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    records = read_jsonl(run / "records.jsonl.gz")
    validation = read_jsonl(run / "validation.jsonl.gz")
    reference = read_jsonl(run / "reference.jsonl")
    store = json.loads((run / "store.json").read_text(encoding="utf-8"))
    certified_name = settings["certified_assumptions"]
    prompts = len(reference)

    cells: dict[tuple[str, float, str, str], list[dict]] = defaultdict(list)
    for record in records:
        cells[(record["stage"], record["budget"], record["assumptions"], record["policy"])].append(record)
    order = lambda item: (item[0][0], item[0][3], -item[0][1], item[0][2])
    curves = {
        f"{stage}|{budget:g}|{name}|{policy}": cell_summary(cell, name == certified_name)
        for (stage, budget, name, policy), cell in sorted(cells.items(), key=order)
    }

    certified_validation = [v for v in validation if v["assumptions"] == certified_name]
    experimental_validation = [v for v in validation if v["assumptions"] != certified_name]

    def violations(entries, key) -> dict[str, int]:
        totals = defaultdict(int)
        for entry in entries:
            for name, count in entry.get(key, {}).items():
                totals[name] += count
        return dict(totals)

    certified_records = [r for r in records if r["assumptions"] == certified_name]
    correctness = {
        "prompts": prompts,
        "prefix_bitwise": sum(all(r["prefix_bitwise_equal"].values()) for r in reference),
        "reference_fallback_bitwise": sum(r["fallback_bitwise_equal"] for r in reference),
        "certified_mismatches": sum(r["certified"] and r["token"] != r["reference_token"] for r in certified_records),
        "fallback_mismatches": sum(not r["certified"] and r["token"] != r["reference_token"] for r in certified_records),
        "enclosure_violations": violations(certified_validation, "enclosure_violations"),
        "pairwise_violations": violations(certified_validation, "pairwise_violations"),
        "logit_violations": sum(v["logit_violations"] for v in certified_validation),
        "exact_suffix_runs": sum("exact_suffix_bitwise" in v for v in certified_validation),
        "exact_suffix_bitwise": sum(v.get("exact_suffix_bitwise", False) for v in certified_validation),
        "fallbacks_checked": sum("fallback_bitwise" in v for v in validation),
        "fallbacks_bitwise": sum(v.get("fallback_bitwise", False) for v in validation),
        "reads_twice": sum(v["head_rows_read_twice"] + v.get("suffix_pages_read_twice", 0) for v in validation),
        "byte_audits_failed": sum(
            not (v["head_byte_audit"] and v.get("suffix_byte_audit", True) and v.get("suffix_budget_respected", True))
            for v in validation
        ),
        "masked_self_test": store["lm_head"]["masked_fallback_self_test"],
        "masked_enabled": store["lm_head"]["masked_enabled"],
    }
    experimental = {
        "enclosure_violations": violations(experimental_validation, "enclosure_violations"),
        "pairwise_violations": violations(experimental_validation, "pairwise_violations"),
        "logit_violations": sum(v["logit_violations"] for v in experimental_validation),
        "wrong_would_certify": sum(
            r["would_certify"] and r["token"] != r["reference_token"] for r in records if r["assumptions"] != certified_name
        ),
    }

    decision_settings = settings["decision"]
    decisions = {}
    for stage in settings["stages"]:
        if stage == "lm_head":
            continue
        options = {
            budget: curves[f"{stage}|{budget:g}|{certified_name}|limited"] for budget in settings["budgets"]
        }
        best = min(options, key=lambda budget: options[budget]["mean_bytes"])
        saving = options[best]["mean_saving_fraction"]
        decisions[stage] = {
            "best_budget": best,
            "mean_bytes": options[best]["mean_bytes"],
            "baseline_mean_bytes": options[best]["baseline_mean_bytes"],
            "mean_saving_fraction": saving,
            "verdict": "BENEFICIAL" if saving >= decision_settings["min_mean_saving_fraction"] else "NOT BENEFICIAL",
        }

    unread_certified = sum(
        cell.get("certified_with_unread_suffix", 0) for key, cell in curves.items() if key.split("|")[2] == certified_name
    )
    budget_cells = len(settings["budgets"]) + len(settings["runtime"]["ceiling_budgets"])
    expected_cells = 1 + (len(settings["stages"]) - 1) * budget_cells * (1 + len(settings["experimental_assumptions"]))
    gate = {
        "certified_mismatches_zero": correctness["certified_mismatches"] == 0,
        "fallback_mismatches_zero": correctness["fallback_mismatches"] == 0,
        "adaptive_region_extends_beyond_lm_head": unread_certified >= settings["gate"]["min_certified_with_unread_suffix"],
        "bound_propagation_conservative": not any(correctness["enclosure_violations"].values())
        and not any(correctness["pairwise_violations"].values())
        and correctness["logit_violations"] == 0,
        "exact_paths_bitwise": correctness["prefix_bitwise"] == prompts
        and correctness["reference_fallback_bitwise"] == prompts
        and correctness["exact_suffix_bitwise"] == correctness["exact_suffix_runs"]
        and correctness["fallbacks_bitwise"] == correctness["fallbacks_checked"],
        "reads_and_bytes_audited": correctness["reads_twice"] == 0 and correctness["byte_audits_failed"] == 0,
        "materialization_curves_recorded": len(curves) == expected_cells
        and all(cell["runs"] == prompts for cell in curves.values()),
        "no_hard_failure": not (run / "failure.json").exists(),
    }
    digests = json.loads((run / "digest.json").read_text(encoding="utf-8"))
    reproducibility = None
    if compare is not None:
        other = json.loads((compare / "digest.json").read_text(encoding="utf-8"))
        keys = ("store_sha256", "reference_sha256", "validation_sha256", "records_sha256")
        reproducibility = {key: digests[key] == other[key] for key in keys}
        gate["reproducible"] = all(reproducibility.values())
    gate["passed"] = all(value is True for value in gate.values())
    walls = [r["wall_ms"] for r in certified_records]
    return {
        "run": run.as_posix(),
        "compared_with": compare.as_posix() if compare else None,
        "prompts": prompts,
        "regions": {stage: info["region_bytes"] for stage, info in store["suffix"].items()},
        "lm_head_weight_bytes": store["lm_head"]["weight_bytes"],
        "correctness": correctness,
        "experimental": experimental,
        "curves": curves,
        "decisions": decisions,
        "certified_with_unread_suffix": unread_certified,
        "gate": gate,
        "reproducibility": reproducibility,
        "digests": digests,
        "wall_ms": quantiles(walls),
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_markdown(summary: dict, certified_name: str) -> str:
    lines = [f"# Phase 2 summary — {summary['run']}", ""]
    lines += ["## Gate", "", "| Criterion | Result |", "| --- | --- |"]
    for key, value in summary["gate"].items():
        lines.append(f"| {key} | {'PASS' if value else 'FAIL'} |")
    lines.append("")
    c = summary["correctness"]
    lines += ["## Correctness (certified model)", ""]
    for key, value in c.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Experimental models (what-ifs, never certified)", ""]
    for key, value in summary["experimental"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Curves",
        "",
        "| stage | policy | budget | model | suffix coverage | coverage | mean fraction | mean MB | baseline MB | saving / region | h width (median) | paths |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, cell in summary["curves"].items():
        stage, budget, name, policy = key.split("|")
        if name == certified_name:
            lines.append(
                f"| {stage} | {policy} | {budget} | {name} | {pct(cell['suffix_coverage'])} | {pct(cell['coverage'])} | "
                f"{cell['fraction']['mean']:.3f} | {cell['mean_bytes'] / 1e6:.2f} | {cell['baseline_mean_bytes'] / 1e6:.2f} | "
                f"{cell['mean_saving_fraction']:+.4f} | {cell['h_relative_width_median']:.4f} | {cell['paths']} |"
            )
        else:
            lines.append(
                f"| {stage} | {policy} | {budget} | {name} | {pct(cell['suffix_coverage'])} | – | – | – | – | – | "
                f"{cell['h_relative_width_median']:.4f} | {cell['paths']} |"
            )
    lines += ["", "## Pairwise certificate: share of each uncertainty term (tightest pair, mean)", ""]
    lines += ["| stage | policy | budget | model | runs | certified | " + " | ".join(TERMS) + " |"]
    lines += ["| --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in TERMS) + " |"]
    for key, cell in summary["curves"].items():
        if "pairwise" in cell:
            stage, budget, name, policy = key.split("|")
            shares = cell["pairwise"]["term_shares"]
            lines.append(
                f"| {stage} | {policy} | {budget} | {name} | {cell['pairwise']['runs']} | {cell['pairwise']['certified']} | "
                + " | ".join(pct(shares.get(term, 0.0)) for term in TERMS)
                + " |"
            )
    lines += ["", "## Decision (per stage, best budget)", ""]
    for stage, decision in summary["decisions"].items():
        lines.append(
            f"- **{stage}**: {decision['verdict']} — best budget {decision['best_budget']}, mean "
            f"{decision['mean_bytes'] / 1e6:.3f} MB against {decision['baseline_mean_bytes'] / 1e6:.3f} MB, "
            f"saving {decision['mean_saving_fraction']:+.4f} of the region"
        )
    lines += ["", f"Wall time per certified run: {summary['wall_ms']}", ""]
    if summary["reproducibility"] is not None:
        lines += ["## Reproducibility", "", f"Compared with {summary['compared_with']}: {summary['reproducibility']}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="second run of the same config, for reproducibility")
    args = parser.parse_args()
    summary = evaluate(args.run, args.compare)
    settings = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.run / "summary.md").write_text(render_markdown(summary, settings["certified_assumptions"]), encoding="utf-8")
    print(render_markdown(summary, settings["certified_assumptions"]))
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
