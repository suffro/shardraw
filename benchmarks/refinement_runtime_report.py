"""Aggregate a Phase 1C run, compare it with the Phase 1B oracle and evaluate the gate.

    uv run python benchmarks/refinement_runtime_report.py experiments/phase1c/<run> [--compare experiments/phase1c/<other run>]

Writes summary.json and summary.md into the run directory. Exit code 0 iff the gate
passes. The primary configuration is the full (conservative) fallback; fractions are
effective bytes to a decision, all resident metadata included, divided by the
original BF16 LM-head bytes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from awpmi.bounds.floating import FLOAT64_BOUND_SLACK, next_down, next_up, round_down_to_grid, round_up_to_grid
from awpmi.stores.refinement import BYTE_PARTS
from awpmi.tracing import read_jsonl

PRIMARY = "full"
GAP_BUCKETS = (0.0, 0.5, 1.0, 2.0, 4.0, float("inf"))


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
    labels = [f"{index}:{name}" for index, name in enumerate(states)]
    by_state = dict.fromkeys(labels, 0)
    for record in certified:
        by_state[labels[record["decision_state_index"]]] += 1
    cumulative, running = {}, 0
    for label in labels:
        running += by_state[label]
        cumulative[label] = running / n
    stages: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for stage, value in record["timings_ms"].items():
            stages[stage].append(value)
    by_gap: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_gap[gap_bucket(gaps[record["prompt_id"]])].append(record)
    total_ms = [r["timings_ms"]["total"] for r in records]
    reference_ms = [r["reference_ms"] for r in records]
    return {
        "inputs": n,
        "certified_mismatches": sum(r["token"] != r["reference_token"] for r in certified),
        "fallback_mismatches": sum(r["token"] != r["reference_token"] for r in records if not r["certified"]),
        "certificate_coverage": len(certified) / n,
        "fallback_rate": 1 - len(certified) / n,
        "masked_guard_trips": sum(r["masked_guard_tripped"] for r in records),
        "certified_by_state": {label: count / n for label, count in by_state.items()},
        "certified_by_state_cumulative": cumulative,
        "fraction": quantiles(r["fraction"] for r in records),
        "fraction_when_certified": quantiles(r["fraction"] for r in certified),
        "io_4k_fraction": quantiles(r["io_bytes_4k"] / weight_bytes for r in records),
        "bytes_mean_fraction": {
            part: float(np.mean([r["bytes"][part] / weight_bytes for r in records])) for part in BYTE_PARTS
        },
        "contenders_after_base": quantiles(r["contenders"][0] for r in records),
        "rows_reaching_exact": quantiles(r["rows_loaded"][-1] for r in records),
        "time_ms": quantiles(total_ms),
        "stage_ms_mean": {stage: float(np.mean(values)) for stage, values in stages.items()},
        "stage_runs": {stage: len(values) for stage, values in stages.items()},
        "reference_ms": quantiles(reference_ms),
        "time_ratio_to_reference": float(np.mean(total_ms) / np.mean(reference_ms)),
        "by_reference_gap": {
            bucket: {
                "inputs": len(rows),
                "coverage": sum(r["certified"] for r in rows) / len(rows),
                "mean_fraction": float(np.mean([r["fraction"] for r in rows])),
            }
            for bucket, rows in sorted(by_gap.items(), key=lambda item: float(item[0][1:].split(",")[0]))
        },
    }


def compare_with_oracle(records: list[dict]) -> dict:
    """Primary records against the Phase 1B oracle's primary configuration, prompt by prompt."""
    n = len(records)
    oracle = [r["oracle"] for r in records]
    ratio = [r["contenders"][0] / o["contenders"][0] for r, o in zip(records, oracle)]
    same = {
        "token": sum(r["token"] == o["token"] for r, o in zip(records, oracle)),
        "certified": sum(r["certified"] == o["certified"] for r, o in zip(records, oracle)),
        "decision_state": sum(r["decision_state_index"] == o["decision_state_index"] for r, o in zip(records, oracle)),
        "contenders_after_base": sum(r["contenders"][0] == o["contenders"][0] for r, o in zip(records, oracle)),
    }
    for state in range(1, len(records[0]["contenders"])):
        same[f"contenders_after_state_{state}"] = sum(
            r["contenders"][state] == o["contenders"][state] for r, o in zip(records, oracle)
        )
        same[f"rows_loaded_state_{state}"] = sum(
            r["rows_loaded"][state] == o["rows_loaded"][state] for r, o in zip(records, oracle)
        )
    runtime_mean = float(np.mean([r["fraction"] for r in records]))
    oracle_mean = float(np.mean([o["fraction"] for o in oracle]))
    return {
        "inputs": n,
        "same": same,
        "contenders_after_base_ratio": quantiles(ratio),
        "mean_fraction_runtime": runtime_mean,
        "mean_fraction_oracle": oracle_mean,
        "mean_fraction_excess": runtime_mean - oracle_mean,
        "mean_fraction_oracle_masked_fallback": float(np.mean([o["fraction_masked_fallback"] for o in oracle])),
        "io_4k_mean_runtime": float(np.mean([r["io_bytes_4k"] for r in records])),
        "io_4k_mean_oracle": float(np.mean([o["io_bytes_4k"] for o in oracle])),
    }


def nearest_even(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round-to-nearest-even of float64 values onto the `dtype` grid, exactly (no double rounding)."""
    down, up = round_down_to_grid(values, dtype), round_up_to_grid(values, dtype)
    below, above = values - down, up - values  # exact: Sterbenz, the neighbours bracket the value
    even = (down.to(dtype).view(torch.int16) & 1) == 0
    return torch.where(below < above, down, torch.where(above < below, up, torch.where(even, down, up)))


def nearest_even_what_if(validation: list[dict], tokens: dict[int, int]) -> dict:
    """Would a round-to-nearest-even output model (instead of faithful rounding) certify the fallbacks?

    Diagnostic only (decision 0004 does not adopt it). For every fallback, the exact-state
    centre and radius of each final contender give an accumulator interval; RN-even is
    monotone, so the reference logit lies in [RN(lower), RN(upper)], intersected with the
    faithful interval. Eliminated rows cannot block a certificate, so the tie-aware
    certificate is checked among the final contenders only.
    """
    fallbacks = certified = wrong = 0
    for entry in validation:
        contenders = entry.get("final_contenders")
        if not contenders:
            continue
        fallbacks += 1
        rows = [c["row"] for c in contenders]
        center = torch.tensor([c["center"] for c in contenders], dtype=torch.float64)
        radius = torch.tensor([c["radius"] for c in contenders], dtype=torch.float64) * (1.0 + FLOAT64_BOUND_SLACK)
        lower = torch.maximum(
            torch.tensor([c["lower"] for c in contenders], dtype=torch.float64),
            nearest_even(next_down(center - radius), torch.bfloat16),
        )
        upper = torch.minimum(
            torch.tensor([c["upper"] for c in contenders], dtype=torch.float64),
            nearest_even(next_up(center + radius), torch.bfloat16),
        )
        best = float(lower.max())
        winner = min(row for row, value in zip(rows, lower.tolist()) if value == best)
        ok = all(
            best > up if row < winner else best >= up
            for row, up in zip(rows, upper.tolist())
            if row != winner
        )
        if ok:
            certified += 1
            wrong += winner != tokens[entry["prompt_id"]]
    return {"fallbacks": fallbacks, "would_certify": certified, "would_certify_wrong_token": wrong}


def evaluate(run: Path, compare: Path | None) -> dict:
    settings = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    store = json.loads((run / "store.json").read_text(encoding="utf-8"))
    digest = json.loads((run / "digest.json").read_text(encoding="utf-8"))
    profile_path = run / "profile.json"
    profile_info = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else None
    reference = read_jsonl(run / "reference.jsonl")
    validation = read_jsonl(run / "validation.jsonl")
    records = read_jsonl(run / "records.jsonl")
    gaps = {row["prompt_id"]: row["reference_top2_gap"] for row in reference}
    tokens = {row["prompt_id"]: row["reference_token"] for row in reference}
    weight_bytes = store["weight_bytes"]

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_mode[record["fallback_mode"]].append(record)
    modes = {mode: summarize(rows, store["states"], weight_bytes, gaps) for mode, rows in by_mode.items()}
    oracle = compare_with_oracle(by_mode[PRIMARY])

    masked_checks = [row for row in validation if row.get("masked_fallback_checked")]
    masked = {
        "self_test": store["masked_fallback_self_test"].get("masked"),
        "fallbacks": len(masked_checks),
        "bitwise": sum(bool(row["masked_fallback_bitwise"]) for row in masked_checks),
        "guard_trips": sum(bool(row.get("masked_guard_tripped")) for row in validation),
    }
    envelope = sum(sum(row["envelope_violations"].values()) for row in validation)
    arithmetic = sum(row["coarse_arithmetic_violations"] for row in validation)

    reproducible = None
    if compare is not None:
        other = json.loads((compare / "digest.json").read_text(encoding="utf-8"))
        reproducible = all(other[key] == digest[key] for key in digest)
    all_records = records
    correctness = {
        "run_complete": len(records) == len(reference) * len(settings["fallbacks"])
        and len(validation) == len(reference)
        and not (run / "failure.json").exists(),
        "certified_mismatches_zero": sum(m["certified_mismatches"] for m in modes.values()) == 0,
        "fallback_mismatches_zero": sum(m["fallback_mismatches"] for m in modes.values()) == 0,
        "reference_prefix_and_full_fallback_bitwise": all(
            row["prefix_bitwise_equal"] and row["fallback_bitwise_equal"] and row["fallback_token"] == row["reference_token"]
            for row in reference
        ),
        "envelope_violations_zero": envelope == 0,
        "coarse_arithmetic_violations_zero": arithmetic == 0,
        "store_consistent": store["packing_lossless"]
        and store["accounting_matches_decomposition"]
        and store["levels_match_oracle_run"],
        "masked_fallback_bitwise": masked["bitwise"] == masked["fallbacks"]
        and masked["guard_trips"] == 0
        and (masked["self_test"] is None or masked["self_test"]["mismatches"] == 0),
        "results_reproducible": reproducible,
    }
    thresholds = settings["gate"]
    primary_mean = modes[PRIMARY]["fraction"]["mean"]
    efficiency = {
        "mean_fraction_within_limit": primary_mean <= thresholds["max_mean_fraction"],
        "excess_over_oracle_within_limit": oracle["mean_fraction_excess"] <= thresholds["max_mean_fraction_excess_over_oracle"],
    }
    gate = {**correctness, **efficiency}
    gate["passed"] = all(value is True for value in gate.values())

    unique_gaps = list(gaps.values())
    return {
        "run": run.as_posix(),
        "compared_with": compare.as_posix() if compare else None,
        "inputs": len(reference),
        "records": len(all_records),
        "digest": digest,
        "environment": {key: environment.get(key) for key in ("gpu", "packages", "source_tree_sha256", "git_commit", "os")},
        "store": store,
        "gate": gate,
        "gate_thresholds": thresholds,
        "reference": {"top2_gap": quantiles(unique_gaps), "bf16_top1_ties": sum(g == 0.0 for g in unique_gaps)},
        "coarse_arithmetic_max_ratio": max(row["coarse_arithmetic_max_ratio"] for row in validation),
        "masked_fallback": masked,
        "nearest_even_what_if": nearest_even_what_if(validation, tokens),
        "oracle_comparison": oracle,
        "modes": modes,
        "profile": profile_info,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f} %"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(summary: dict) -> str:
    lines = [f"# Phase 1C summary — `{Path(summary['run']).name}`", ""]
    lines += [f"Inputs: {summary['inputs']} prompts, {summary['records']} runtime runs.", ""]
    lines += ["## Gate", "", "| criterion | result |", "|---|---|"]
    for key, value in summary["gate"].items():
        lines.append(f"| {key} | {'NOT EVALUATED' if value is None else ('PASS' if value else 'FAIL')} |")
    masked, what_if = summary["masked_fallback"], summary["nearest_even_what_if"]
    lines += [
        "",
        f"Coarse binary32 error: largest observed share of its bound {summary['coarse_arithmetic_max_ratio']:.2e}.",
        f"Masked fallback: self-test {masked['self_test']}, {masked['fallbacks']} fallbacks, "
        f"{masked['bitwise']} bitwise on the surviving rows, {masked['guard_trips']} guard trips.",
        f"Round-to-nearest-even what-if: {what_if['would_certify']} of {what_if['fallbacks']} fallbacks would certify "
        f"({what_if['would_certify_wrong_token']} with a wrong token).",
        "",
        "## Runtime by fallback mode",
        "",
        "| mode | coverage | mean | median | p90 | p95 | certified mean | 4 KiB I/O mean | contenders after base (median / p90) | time ms (mean / median / p95) | reference ms (mean) | cert. mism. | fb. mism. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, m in summary["modes"].items():
        f, t = m["fraction"], m["time_ms"]
        lines.append(
            f"| {mode} | {pct(m['certificate_coverage'])} | {fmt(f['mean'])} | {fmt(f['median'])} | {fmt(f['p90'])} "
            f"| {fmt(f['p95'])} | {fmt(m['fraction_when_certified']['mean'])} | {fmt(m['io_4k_fraction']['mean'])} "
            f"| {m['contenders_after_base']['median']:.0f} / {m['contenders_after_base']['p90']:.0f} "
            f"| {t['mean']:.2f} / {t['median']:.2f} / {t['p95']:.2f} | {m['reference_ms']['mean']:.3f} "
            f"| {m['certified_mismatches']} | {m['fallback_mismatches']} |"
        )
    lines += ["", "## Byte breakdown (mean fraction of the BF16 LM head)", ""]
    lines += ["| mode | " + " | ".join(BYTE_PARTS) + " |", "|---|" + "---|" * len(BYTE_PARTS)]
    for mode, m in summary["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(fmt(m["bytes_mean_fraction"][p]) for p in BYTE_PARTS) + " |")
    lines += ["", "## Certified after each state (cumulative)", ""]
    for mode, m in summary["modes"].items():
        cells = ", ".join(f"{state}: {pct(v)}" for state, v in m["certified_by_state_cumulative"].items())
        lines.append(f"- {mode}: {cells}")
    o = summary["oracle_comparison"]
    lines += ["", "## Against the Phase 1B oracle (primary configuration)", ""]
    lines += [f"- same {key}: {value} of {o['inputs']}" for key, value in o["same"].items()]
    lines += [
        f"- contenders after the base, runtime over oracle: mean {o['contenders_after_base_ratio']['mean']:.3f}, "
        f"median {o['contenders_after_base_ratio']['median']:.3f}, max {o['contenders_after_base_ratio']['max']:.3f}",
        f"- mean fraction: runtime {fmt(o['mean_fraction_runtime'])}, oracle {fmt(o['mean_fraction_oracle'])}, "
        f"excess {o['mean_fraction_excess']:+.4f}; oracle with masked fallback {fmt(o['mean_fraction_oracle_masked_fallback'])}",
        f"- 4 KiB I/O mean bytes: runtime {o['io_4k_mean_runtime']:.0f}, oracle {o['io_4k_mean_oracle']:.0f}",
        "",
        "## Time per stage (mean ms over the runs that reach it)",
        "",
    ]
    for mode, m in summary["modes"].items():
        cells = ", ".join(
            f"{stage} {value:.2f} (n={m['stage_runs'][stage]})" for stage, value in m["stage_ms_mean"].items()
        )
        lines.append(f"- {mode}: {cells}; total / reference = {m['time_ratio_to_reference']:.1f}×")
    if summary["profile"]:
        lines += ["", "## Profile", "", "```json", json.dumps(summary["profile"], indent=2), "```"]
    lines += ["", "## Primary mode by reference top-2 gap (coverage, mean fraction)", ""]
    for bucket, b in summary["modes"][PRIMARY]["by_reference_gap"].items():
        lines.append(f"- {bucket} n={b['inputs']}: {pct(b['coverage'])}, {fmt(b['mean_fraction'])}")
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
