# AWPMI — Adaptive Weight-Page Materialization for Inference

Research prototype. AWPMI progressively materializes independently fetchable weight
pages. It stops only when a conservative certificate proves that the discrete
next-token decision is identical to that of the fully materialized reference model.
Confidence is never a certificate. If certification fails, AWPMI materializes
everything and reproduces the reference computation exactly.

The full plan is in [`.context/state/AWPMI_Implementation_Roadmap.md`](.context/state/AWPMI_Implementation_Roadmap.md).
The current status and results are in [`.context/state/current.md`](.context/state/current.md).

## Status

**Phase 1A — minimal certified materialization** is implemented. The adaptive region
is the LM head of `HuggingFaceTB/SmolLM2-135M-Instruct`, split into input-column
pages with Cauchy–Schwarz residual bounds. The transformer body runs exactly, and
all weights stay resident in memory.

**Phase 1B — precision-refinement decomposition search** is an oracle, not a
runtime. It decomposes the LM head exactly as W = coarse + refinements + exact
remainder, adds row-selective refinement and a tie-aware certificate, and measures
how many bytes a certified decision would need.

**Phase 1C — refinement runtime** implements the decomposition Phase 1B chose: an int6
coarse pass over every row, int4 refinement of the rows that can still win, then the
original BF16 rows of the last few. It reads a bit-packed store that counts every byte,
and certifies with on average 0.50 of the BF16 LM-head bytes, or 0.40 with the masked
fallback (the default since Phase 2).

**Phase 2 — certification into the transformer** extends the adaptive region into the
last layer's MLP: the final projection (`down_proj`) or the whole MLP is read only in
part, its output is bounded through the reference's own BF16 operations and the final
RMSNorm, and a scale-free pairwise certificate decides through the norm. Only the faithful
rounding model certifies; round-to-nearest-even is recorded as a what-if. It is sound
(zero mismatches and zero bound violations over 1000 prompts), but on this model it does
not save bytes: the reference's own BF16 roundings of activations cap the final
projection's certificate at 27% of tokens even with every weight read.

## Setup

```bash
python -m pip install uv     # if uv is not installed
uv sync                      # Python 3.11+, torch 2.14.1 (CUDA 13.0 wheels), transformers 5.18.0
```

## Usage

```bash
uv run pytest                                                   # unit + model tests
uv run python benchmarks/run.py --output experiments/phase1/my-run [--num-prompts 50]
uv run python benchmarks/report.py experiments/phase1/my-run [--compare experiments/phase1/other-run]
uv run python benchmarks/oracle.py experiments/phase1/my-run --output experiments/phase1/my-run-oracle
uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/my-run [--num-prompts 50]
uv run python benchmarks/refinement_report.py experiments/phase1b/my-run [--compare experiments/phase1b/other-run]
uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/my-run [--num-prompts 50]
uv run python benchmarks/refinement_runtime_report.py experiments/phase1c/my-run [--compare experiments/phase1c/other-run]
uv run python benchmarks/fallback_study.py --output experiments/phase1c/my-study
uv run python benchmarks/suffix_runtime.py --output experiments/phase2/my-run [--num-prompts 50]
uv run python benchmarks/suffix_report.py experiments/phase2/my-run [--compare experiments/phase2/other-run]
```

`run.py` writes raw per-input records, validation records, the prompts, the
config and the environment metadata. It stops at the first hard failure.
`report.py` aggregates the results and evaluates the Phase 1 gate. `oracle.py`
computes ceilings that do not depend on the scheduler, to tell whether the
ordering or the bound is the bottleneck. `refinement_oracle.py` and
`refinement_report.py` do the same for Phase 1B decompositions
(`configs/phase1b-refinement.yaml`), and classify each one against Phase 1A.
`refinement_runtime.py` runs the Phase 1C runtime on every prompt with both fallbacks,
validates it against the reference, and compares its bytes with the Phase 1B oracle;
`refinement_runtime_report.py` evaluates the gate (`configs/phase1c-runtime.yaml`).
`fallback_study.py` gathers the evidence behind the fallback decisions.
`suffix_runtime.py` runs the Phase 2 adaptive suffix for every stage, budget and rounding
model, checks every intermediate against the reference, and `suffix_report.py` draws the
materialization curves and evaluates the gate (`configs/phase2-suffix.yaml`).

## Layout

```text
src/awpmi/      reference, paging, bounds, state, certificate, schedulers, executor, tracing,
                decomposition (Phase 1B, packing), oracle (Phase 1B simulator),
                stores (Phase 1C packed store, Phase 2 MLP store), refinement_head (Phase 1C runtime),
                bounds/{rounding,enclosure,operators,pairwise} and suffix_runtime (Phase 2)
tests/          bound soundness, pages, certificate and ties, reference parity, fallback parity,
                decomposition exactness, refinement oracle, packing, coarse bounds, runtime,
                rounding models, operator bounds, enclosure LM head, adaptive suffix
benchmarks/     run.py, report.py, prompts.py, oracle.py, refinement_oracle.py, refinement_report.py,
                refinement_runtime.py, refinement_runtime_report.py, fallback_study.py,
                suffix_runtime.py, suffix_report.py
configs/        smollm2-135m.yaml (pinned model and dataset revisions), phase1b-refinement.yaml,
                phase1c-runtime.yaml, phase2-suffix.yaml
experiments/    raw results per phase and run
```
