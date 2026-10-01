# Architecture

## Overview

This repository hosts **AWPMI — Adaptive Weight-Page Materialization for Inference**, a
research prototype (Python package `awpmi`, at the repository root). AWPMI materializes
independently fetchable weight pages one at a time. It stops only when a conservative,
deterministic certificate proves that the next-token argmax equals that of the fully
materialized reference model. The plan is in `state/AWPMI_Implementation_Roadmap.md`.

Implemented so far: **Phase 1**. Only the LM head is adaptive. The transformer body runs
exactly ("exact prefix", roadmap Mode A), and all weights stay resident, so materialization
is logical rather than physical.

## Major components (`src/awpmi/`)

| Module | Role |
| --- | --- |
| `runtime.py` | Declared numerical environment (determinism, FP32 accumulation, no TF32 or reduced-precision reductions). Call it before any CUDA matmul. |
| `models/smollm2.py` | Loads the pinned model, gives access to the LM-head weight, runs the exact prefix (`final_hidden_state`). |
| `reference.py` | `ReferenceRunner.next_token`: the unmodified HF forward (`logits_to_keep=1`), with the LM-head input captured by a hook. |
| `paging/` | `WeightPage` (column block of a weight), `PageIndex` (pages plus bound metadata), `PageBoundMetadata` (row-page L2 norms, rounded up to float32), `InMemoryPageSource` (fetch-counting views). |
| `bounds/floating.py` | γ_n error bounds; directed rounding of float64 values onto a dtype's grid. |
| `bounds/linear.py` | Cauchy–Schwarz block-norm upper bounds; exact float64 page contributions. |
| `bounds/residual.py` | `ReferenceNumerics` (accumulation model of the reference GEMM); `ResidualBounder` (partial logits and remaining pages → intervals on the reference logits). |
| `state.py` | `ResidualState`: partial logits, lower/upper bounds, materialized and remaining pages. |
| `certificate.py` | `Certificate.check`: CERTIFIED iff lower[w] > max_{j≠w} upper[j] with w = argmax(partial), otherwise UNKNOWN. |
| `schedulers/` | Deterministic page ordering: `sequential`, `largest_residual`, `bound_per_byte`. They see metadata and state, never page values. |
| `executor.py` | `AdaptiveLMHead.run`: the certify → schedule → materialize → refine loop, plus the exact fallback. |
| `tracing.py`, `config.py` | Environment metadata, source-tree hash, JSONL, digests; YAML config. |

Outside the package: `benchmarks/run.py` (experiment driver), `benchmarks/report.py`
(aggregation and gate evaluation), `benchmarks/prompts.py` (deterministic wikitext-2
prompts), `benchmarks/oracle.py` (scheduler-independent ceilings: real-certificate
singleton check plus an ideal magnitude-bound ceiling; diagnostic only),
`configs/smollm2-135m.yaml`, `experiments/phase1/<run>/` (raw results), and `tests/`.

## Data flow (one input)

1. `ReferenceRunner.next_token(ids)` returns the reference token, its BF16 logits, and the
   LM-head input.
2. `final_hidden_state(model, ids)` is the exact prefix. It must be bitwise-equal to the
   captured LM-head input.
3. `AdaptiveLMHead.run(hidden, scheduler)`:
   - computes per-page norms of the hidden state and builds a `ResidualBounder`;
   - loops: `Certificate.check(state)` → if CERTIFIED, stop; else `scheduler.select` →
     `PageSource.get(page)` → `page_contribution` (float64) → `refine_state`;
   - if every page is materialized and the result is still UNKNOWN, the fallback runs
     `F.linear(hidden, cat(pages))`, which is bitwise-equal to the reference.

## Important constraints

- The certificate bounds the **floating-point reference logits** (after accumulation
  error and output rounding), not just the exact product. See
  `decisions/0001-certificate-targets-floating-point-reference.md`.
- The fallback recomputes the reference operation. See
  `decisions/0002-fallback-recomputes-reference-operation.md`.
- Storage, scheduling, partial execution and certification stay in separate modules.
  Schedulers affect efficiency only.
- Bound metadata is resident and counts against savings. For the LM head
  (49152 × 576, BF16, 56.6 MB) it is 1.8 / 3.5 / 7.1 / 14.2 MB for page widths
  64 / 32 / 16 / 8.
