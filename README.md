# AWPMI — Adaptive Weight-Page Materialization for Inference

Research prototype. AWPMI progressively materializes independently fetchable weight
pages. It stops only when a conservative certificate proves that the discrete
next-token decision is identical to that of the fully materialized reference model.
Confidence is never a certificate. If certification fails, AWPMI materializes
everything and reproduces the reference computation exactly.

The full plan is in [`.context/state/AWPMI_Implementation_Roadmap.md`](.context/state/AWPMI_Implementation_Roadmap.md).
The current status and results are in [`.context/state/current.md`](.context/state/current.md).

## Status

**Phase 1 — minimal certified materialization** is implemented. The adaptive region
is the LM head of `HuggingFaceTB/SmolLM2-135M-Instruct`, split into input-column
pages with Cauchy–Schwarz residual bounds. The transformer body runs exactly, and
all weights stay resident in memory.

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
```

`run.py` writes raw per-input records, validation records, the prompts, the
config and the environment metadata. It stops at the first hard failure.
`report.py` aggregates the results and evaluates the Phase 1 gate. `oracle.py`
computes ceilings that do not depend on the scheduler, to tell whether the
ordering or the bound is the bottleneck.

## Layout

```text
src/awpmi/      reference, paging, bounds, state, certificate, schedulers, executor, tracing
tests/          bound soundness, pages, certificate, reference parity, fallback parity
benchmarks/     run.py, report.py, prompts.py, oracle.py
configs/        smollm2-135m.yaml (pinned model and dataset revisions)
experiments/    raw results per phase and run
```
