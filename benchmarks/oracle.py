"""Scheduler-independent ceilings on early certification for a completed Phase 1 run.

    uv run python benchmarks/oracle.py experiments/phase1/<run> --output experiments/phase1/<run>-oracle

Diagnostic only: nothing here participates in a certificate.

* **Cauchy–Schwarz oracle.** Intervals are nested as pages are added (a page's true
  contribution lies inside the radius it removes), so certification is monotone in
  the materialized set: an input can certify before the final page under *some*
  ordering iff it certifies with exactly one page missing. Every singleton is checked
  with the real certificate (the ceiling for any scheduler); the missing set is then
  grown greedily (a lower bound on what an oracle scheduler could skip).
* **Ideal-bound oracle.** The same, but each missing page p is bounded by its actual
  magnitude |W[j,p]·h_p| — the tightest sign-agnostic per-page bound, unattainable in
  practice. If even this rarely certifies early, no better per-page magnitude bound
  can rescue this page decomposition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from awpmi.bounds.linear import page_contribution  # noqa: E402
from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.certificate import Certificate  # noqa: E402
from awpmi.config import load_config  # noqa: E402
from awpmi.executor import AdaptiveLMHead  # noqa: E402
from awpmi.models.smollm2 import (  # noqa: E402
    LM_HEAD_PARAMETER,
    ModelSpec,
    final_hidden_state,
    lm_head_weight,
    load_model,
    resolve_dtype,
)
from awpmi.paging.index import build_column_page_index  # noqa: E402
from awpmi.paging.source import InMemoryPageSource  # noqa: E402
from awpmi.state import ResidualState  # noqa: E402
from awpmi.tracing import JsonlWriter, environment_metadata, read_jsonl  # noqa: E402


def cs_certificate(bounder, full_partial, contributions, missing: frozenset[int]):
    """The real certificate with `missing` pages unmaterialized."""
    page_count = contributions.shape[0]
    mask = torch.zeros(page_count, dtype=torch.bool, device=full_partial.device)
    if missing:
        mask[list(missing)] = True
    partial = full_partial - contributions[mask].sum(dim=0)
    lower, upper = bounder.logit_bounds(partial, mask)
    materialized = tuple(p for p in range(page_count) if p not in missing)
    return Certificate.check(ResidualState(partial, lower, upper, materialized, tuple(sorted(missing))))


def ideal_margins(partial: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Row-wise margin lower[w] − max_{j≠w} upper[j] for stacked candidate states [C, V]."""
    winners = partial.argmax(dim=1, keepdim=True)
    lower_w = (partial - radius).gather(1, winners).squeeze(1)
    upper = (partial + radius).scatter(1, winners, -torch.inf)
    return lower_w - upper.max(dim=1).values


def greedy_ideal_skip(full_partial: torch.Tensor, contributions: torch.Tensor) -> tuple[bool, int]:
    """(some single page skippable, pages skipped greedily) under the ideal magnitude bound."""
    magnitudes = contributions.abs()
    partial, radius = full_partial.clone(), torch.zeros_like(full_partial)
    candidates = list(range(contributions.shape[0]))
    skipped = 0
    single = None
    while candidates:
        index = torch.tensor(candidates, device=full_partial.device)
        margins = ideal_margins(partial - contributions[index], radius + magnitudes[index])
        if single is None:
            single = bool((margins > 0).any())
        best = int(margins.argmax())
        if float(margins[best]) <= 0:
            break
        p = candidates.pop(best)
        partial, radius = partial - contributions[p], radius + magnitudes[p]
        skipped += 1
    return bool(single), skipped


def analyse(head: AdaptiveLMHead, hidden: torch.Tensor, reference_token: int) -> dict:
    hidden_vector = hidden.reshape(-1).to(torch.float64)
    contributions = torch.stack(
        [page_contribution(head.source.get(page), hidden_vector[page.column_slice]) for page in head.index.pages]
    )
    full_partial = contributions.sum(dim=0)
    _, bounder = head.full_partial_logits(hidden)
    pages_total = len(head.index)

    full = cs_certificate(bounder, full_partial, contributions, frozenset())
    singles = (
        [p for p in range(pages_total) if cs_certificate(bounder, full_partial, contributions, frozenset({p})).certified]
        if full.certified
        else []
    )
    missing: frozenset[int] = frozenset()
    candidates = set(singles)
    while candidates:
        best = None
        for p in sorted(candidates):
            result = cs_certificate(bounder, full_partial, contributions, missing | {p})
            if result.certified and (best is None or result.certificate_margin > best[1]):
                best = (p, result.certificate_margin)
        if best is None:
            break
        missing = missing | {best[0]}
        candidates.discard(best[0])

    ideal_single, ideal_skipped = greedy_ideal_skip(full_partial, contributions)

    w = reference_token
    cs_radius = float(bounder.row_page_norms[w] @ bounder.hidden_page_norms)
    absolute_mass = float(contributions[:, w].abs().sum())
    top2 = torch.topk(full_partial, 2).values
    return {
        "certified_with_all_pages": full.certified,
        "cs_single_page_skippable": len(singles),
        "cs_early_possible": bool(singles),
        "cs_greedy_pages_skipped": len(missing),
        "cs_greedy_materialized_fraction": (pages_total - len(missing)) / pages_total,
        "ideal_early_possible": ideal_single,
        "ideal_greedy_pages_skipped": ideal_skipped,
        "ideal_greedy_materialized_fraction": (pages_total - ideal_skipped) / pages_total,
        "exact_top2_gap": float(top2[0] - top2[1]),
        "winner_cs_radius": cs_radius,
        "winner_absolute_page_mass": absolute_mass,
    }


def summarize(rows: list[dict]) -> dict:
    gaps = [r["exact_top2_gap"] for r in rows if r["exact_top2_gap"] > 0]
    radius_over_gap = [r["winner_cs_radius"] / r["exact_top2_gap"] for r in rows if r["exact_top2_gap"] > 0]
    mass_over_gap = [r["winner_absolute_page_mass"] / r["exact_top2_gap"] for r in rows if r["exact_top2_gap"] > 0]
    looseness = [r["winner_cs_radius"] / r["winner_absolute_page_mass"] for r in rows if r["winner_absolute_page_mass"] > 0]
    return {
        "inputs": len(rows),
        "pages_total": rows[0]["pages_total"],
        "cs_early_certification_ceiling": float(np.mean([r["cs_early_possible"] for r in rows])),
        "cs_greedy_mean_materialized_fraction": float(np.mean([r["cs_greedy_materialized_fraction"] for r in rows])),
        "cs_greedy_max_pages_skipped": int(max(r["cs_greedy_pages_skipped"] for r in rows)),
        "ideal_early_certification_ceiling": float(np.mean([r["ideal_early_possible"] for r in rows])),
        "ideal_greedy_mean_materialized_fraction": float(
            np.mean([r["ideal_greedy_materialized_fraction"] for r in rows])
        ),
        "exact_top2_gap_median": float(np.median(gaps)),
        "winner_cs_radius_over_gap_median": float(np.median(radius_over_gap)),
        "winner_absolute_page_mass_over_gap_median": float(np.median(mass_over_gap)),
        "winner_cs_looseness_median": float(np.median(looseness)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="completed benchmark run (its config and prompts are reused)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config, _ = load_config(args.run / "config.yaml")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(config.model.repository, config.model.revision, resolve_dtype(config.model.dtype, device), device)
    model, _ = load_model(spec)
    weight = lm_head_weight(model)
    numerics = ReferenceNumerics(weight.dtype, config.numerics.accumulation_unit_roundoff, weight.shape[1])
    heads = {
        w: AdaptiveLMHead(build_column_page_index(weight, LM_HEAD_PARAMETER, w), InMemoryPageSource(weight), numerics)
        for w in config.paging.page_widths
    }
    reference_tokens = {
        (r["prompt_id"], r["page_width"]): r["reference_token"] for r in read_jsonl(args.run / "validation.jsonl")
    }
    model_info = {"repository": spec.repository, "revision": spec.revision, "dtype": str(spec.dtype)}
    environment = environment_metadata(REPO_ROOT, model_info, NUMERICS_FLAGS)
    environment["source_run"] = args.run.as_posix()
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    rows: list[dict] = []
    started = time.perf_counter()
    with JsonlWriter(args.output / "oracle.jsonl") as log:
        for prompt in read_jsonl(args.run / "prompts.jsonl"):
            input_ids = torch.tensor([prompt["token_ids"]], device=device)
            hidden = final_hidden_state(model, input_ids)
            for width, head in heads.items():
                row = {
                    "prompt_id": prompt["prompt_id"],
                    "page_width": width,
                    "pages_total": len(head.index),
                    **analyse(head, hidden, reference_tokens[(prompt["prompt_id"], width)]),
                }
                log.write(row)
                rows.append(row)
    summary = {f"width={w}": summarize([r for r in rows if r["page_width"] == w]) for w in heads}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"done in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
