# Architecture

## Overview

This repository hosts **AWPMI — Adaptive Weight-Page Materialization for Inference**, a
research prototype (Python package `awpmi`, at the repository root). AWPMI materializes
independently fetchable weight pages one at a time. It stops only when a conservative,
deterministic certificate proves that the next-token argmax equals that of the fully
materialized reference model. The plan is in `state/AWPMI_Implementation_Roadmap.md`.

Implemented so far:

- **Phase 1A**: a runtime that materializes input-column pages. Only the LM head is
  adaptive. The transformer body runs exactly ("exact prefix", roadmap Mode A), and all
  weights stay resident, so materialization is logical rather than physical.
- **Phase 1B**: an *oracle* for precision-refinement decompositions of the LM head, with
  row-selective refinement. It is a simulator, not a runtime: nothing in
  `awpmi.decomposition` or `awpmi.oracle` is called by `executor.py`.

## Major components (`src/awpmi/`)

| Module | Role |
| --- | --- |
| `runtime.py` | Declared numerical environment (determinism, FP32 accumulation, no TF32 or reduced-precision reductions). Call it before any CUDA matmul. |
| `models/smollm2.py` | Loads the pinned model, gives access to the LM-head weight, runs the exact prefix (`final_hidden_state`). |
| `reference.py` | `ReferenceRunner.next_token`: the unmodified HF forward (`logits_to_keep=1`), with the LM-head input captured by a hook. |
| `paging/` | `WeightPage` (column block of a weight), `PageIndex` (pages plus bound metadata), `PageBoundMetadata` (row-page L2 norms, rounded up to float32), `InMemoryPageSource` (fetch-counting views). |
| `bounds/floating.py` | γ_n error bounds; directed rounding of float64 values onto a dtype's grid. |
| `bounds/linear.py` | Cauchy–Schwarz block-norm upper bounds; exact float64 page contributions. |
| `bounds/residual.py` | `ReferenceNumerics` (accumulation model of the reference GEMM); `ResidualBounder` (partial logits and remaining pages → intervals on the reference logits); `reference_logit_interval` (centre and radius → output-grid interval, shared by Phase 1A and 1B). |
| `bounds/remainder.py` | Phase 1B: per-row remainder norm metadata (`RowNormBounds`); bound min(‖x‖₂‖h‖₂, ‖x‖_∞‖h‖₁) on a missing remainder row. |
| `state.py` | `ResidualState`: partial logits, lower/upper bounds, materialized and remaining pages. |
| `certificate.py` | `Certificate.check`: CERTIFIED iff lower[w] > max_{j≠w} upper[j] with w = argmax(partial), otherwise UNKNOWN. `TieBreak.LOWEST_INDEX` (decision 0003) also accepts equality against higher-index rows. `certify_columns` and `contenders` are batched forms used by the oracle (row elimination). |
| `decomposition/` | Phase 1B: `RefinementDecomposition` (W = base + refinement levels + exact remainder, per-row int-b levels with float32 scales, exactness checked with TwoSum, byte accounting). |
| `oracle/refinement.py` | Phase 1B diagnostic simulator: `RefinementBatch` (per-state intervals for the `realistic`, `abs_mass` and `ideal` bound tiers), `simulate` (global or row-selective refinement), 4 KiB I/O accounting. |
| `schedulers/` | Deterministic page ordering: `sequential`, `largest_residual`, `bound_per_byte`. They see metadata and state, never page values. |
| `executor.py` | `AdaptiveLMHead.run`: the certify → schedule → materialize → refine loop, plus the exact fallback. |
| `tracing.py`, `config.py` | Environment metadata, source-tree hash, JSONL, digests; YAML config. |

Outside the package: `benchmarks/run.py` (experiment driver), `benchmarks/report.py`
(aggregation and gate evaluation), `benchmarks/prompts.py` (deterministic wikitext-2
prompts), `benchmarks/oracle.py` (scheduler-independent ceilings: real-certificate
singleton check plus an ideal magnitude-bound ceiling; diagnostic only),
`configs/smollm2-135m.yaml`, `experiments/phase1/<run>/` (raw results), and `tests/`.
Phase 1B adds `benchmarks/refinement_oracle.py` (driver: reference pass, every decomposition ×
mode × bound tier × tie rule, hard-failure checks), `benchmarks/refinement_report.py`
(aggregation, Phase 1A comparison, decision-gate classification),
`configs/phase1b-refinement.yaml` and `experiments/phase1b/<run>/`.

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
  64 / 32 / 16 / 8. Phase 1B remainder norms cost 0.39 MB per non-exact state.
- Decomposition, bounds, certification and simulation are separate modules. A
  decomposition only builds levels and counts bytes. The oracle's `abs_mass` and `ideal`
  tiers use information a runtime cannot have and must never reach a runtime path.
