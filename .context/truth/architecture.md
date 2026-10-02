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
  row-selective refinement. It is a simulator, not a runtime: nothing in `awpmi.oracle`
  is called by a runtime.
- **Phase 1C**: the runtime of the decomposition Phase 1B chose (q6+q4, row-selective,
  tie-aware), `RefinementLMHead`. It reads a bit-packed store whose every read is
  counted. The store is still resident in memory: materialization is logical, but the
  bytes are those of real packed buffers.
- **Phase 2**: certification extended into the transformer, as an adaptive suffix inside
  the last layer's MLP (`AdaptiveSuffixRuntime`): the final projection (`down_proj`,
  depth 1) or the whole last MLP (depth 2), ahead of the Phase 1C LM head. Enclosures
  are propagated through the reference's own operations under the faithful rounding
  model; a scale-free pairwise certificate works through the final RMSNorm. The masked
  LM-head fallback is the default (decision 0005).

## Major components (`src/awpmi/`)

| Module | Role |
| --- | --- |
| `runtime.py` | Declared numerical environment (determinism, FP32 accumulation, no TF32 or reduced-precision reductions). Call it before any CUDA matmul. |
| `models/smollm2.py` | Loads the pinned model, gives access to the LM-head weight, runs the exact prefix (`final_hidden_state`). Phase 2: `suffix_prefix`, the model's own forward interrupted by a pre-hook where the adaptive suffix begins (`down_proj` or the last `mlp`), optionally with a KV cache. |
| `reference.py` | `ReferenceRunner.next_token`: the unmodified HF forward (`logits_to_keep=1`), with the LM-head input captured by a hook; with `intermediates=True`, also every intermediate of the last MLP and the final norm (hooks only observe). |
| `paging/` | `WeightPage` (column block of a weight; Phase 2 adds `layer` and `tensor_role`), `PageIndex` (pages plus bound metadata), `PageBoundMetadata` (row-page L2 norms, rounded up to float32), `InMemoryPageSource` (fetch-counting views). |
| `bounds/floating.py` | γ_n error bounds; directed rounding of float64 values onto a dtype's grid. |
| `bounds/linear.py` | Cauchy–Schwarz block-norm upper bounds; exact float64 page contributions. |
| `bounds/residual.py` | `ReferenceNumerics` (accumulation model of the reference GEMM); `ResidualBounder` (partial logits and remaining pages → intervals on the reference logits); `reference_logit_interval` (centre and radius → output-grid interval, shared by Phase 1A and 1B). |
| `bounds/remainder.py` | Phase 1B: per-row remainder norm metadata (`RowNormBounds`); bound min(‖x‖₂‖h‖₂, ‖x‖_∞‖h‖₁) on a missing remainder row. |
| `bounds/coarse.py` | Phase 1C: error model of the runtime's binary32 coarse pass (`CoarseArithmetic`, γ_{K+2}(2⁻²²) plus a flush-to-zero term) and the free Hölder bound s·limit·‖h‖₁ on a level's absolute mass. |
| `bounds/rounding.py` | Phase 2: rounding models of the reference's outputs. `FAITHFUL` (certified, decision 0001) and `NEAREST_EVEN` (experimental, decision 0005); `RoundingAssumptions` pairs a model for GEMM epilogues and one for elementwise kernels (`CERTIFIED` is faithful for both); `round_enclosure`, `spacing_upper`, rounding-error bounds. |
| `bounds/enclosure.py` | Phase 2: `Enclosure` (lower, upper, provenance), the residual state generalized to any internal tensor (roadmap §2.2); `UnboundedValue` makes the caller fall back. |
| `bounds/operators.py` | Phase 2: verified propagation through the reference operations of the last MLP and the final norm: `linear` (exact or partly unread weights, interval input), `residual_add`, `multiply`, `silu`, `rms_norm` (returns q, n and h). Documented fp32 assumptions: 2⁻¹⁸ relative error per elementwise fp32 operation, flushing below the normal range, γ_{n+2}(2⁻²²) per reduction. |
| `bounds/pairwise.py` | Phase 2: scale-free pairwise certificate through the final RMSNorm. For a winner w and each contender j it lower-bounds (acc_w − acc_j)/q with a box bound and a decomposed bound that keeps the down projection's cancellation (M = W_dᵀΔ, unread columns by Cauchy–Schwarz), then requires a 2-ulp separation of the faithful logits. |
| `state.py` | `ResidualState`: partial logits, lower/upper bounds, materialized and remaining pages. |
| `certificate.py` | `Certificate.check`: CERTIFIED iff lower[w] > max_{j≠w} upper[j] with w = argmax(partial), otherwise UNKNOWN. `TieBreak.LOWEST_INDEX` (decision 0003) also accepts equality against higher-index rows. `certify_columns` and `contenders` are batched forms used by the oracle (row elimination). |
| `decomposition/` | Phase 1B: `RefinementDecomposition` (W = base + refinement levels + exact remainder, per-row int-b levels with float32 scales, exactness checked with TwoSum, byte accounting). Phase 1C adds `packing.py` (per-row little-endian bitstream of biased codes, `CodeLayout.unpack`) and `accounting.py` (4 KiB block accounting, shared with the oracle). |
| `oracle/refinement.py` | Phase 1B diagnostic simulator: `RefinementBatch` (per-state intervals for the `realistic`, `abs_mass` and `ideal` bound tiers), `simulate` (global or row-selective refinement). |
| `stores/refinement.py` | Phase 1C: `PackedRefinementStore`, packed levels, scales, resident remainder norms and the original rows. `read_level` / `read_exact` / `read_fallback` are the only way to weight values; each read is logged with its rows and bytes (`bytes_read`, `reads`). |
| `stores/suffix.py` | Phase 2: `MLPStore`, neuron-major pages of the last MLP (gate row, up row, down column of each neuron, one contiguous run each), served only for the stage's adaptive roles, every read logged; resident metadata: down column norms, down row norms (L2, L∞), up row norms. |
| `refinement_head.py` | Phase 1C runtime `RefinementLMHead.run`: binary32 coarse pass over every row → certificate → elimination → float64 refinement of the contenders (the oracle's realistic arithmetic) → exact rows → certificate or fallback (`FallbackMode.MASKED`, the default since decision 0005, self-tested and guarded, else `FULL`). Phase 2: `run_enclosure` runs the same states on an `Enclosure` of h (input spread \|L\|·ρ added to every radius, no fallback, optional row limits), and `HeldReads` lets one token's passes share their reads. Optional `RunTrace` and `StageTimer`. |
| `suffix_runtime.py` | Phase 2 runtime `AdaptiveSuffixRuntime.run` for a `SuffixStage` (lm_head, down, mlp): read a budgeted share of the suffix's neuron pages (largest bound contribution first) → enclosures through the suffix → LM-head pass on the enclosure → pairwise certificate → else exact recomputation of the suffix (the reference's operations and shapes, bitwise) and the Phase 1C LM head on the exact h, reusing the reads. Experimental rounding models report `would_certify` only. |
| `schedulers/` | Deterministic page ordering: `sequential`, `largest_residual`, `bound_per_byte`. They see metadata and state, never page values. |
| `executor.py` | `AdaptiveLMHead.run`: the certify → schedule → materialize → refine loop, plus the exact fallback. |
| `tracing.py`, `config.py` | Environment metadata, source-tree hash, JSONL, digests, `StageTimer`; YAML config. |

Outside the package: `benchmarks/run.py` (experiment driver), `benchmarks/report.py`
(aggregation and gate evaluation), `benchmarks/prompts.py` (deterministic wikitext-2
prompts), `benchmarks/oracle.py` (scheduler-independent ceilings: real-certificate
singleton check plus an ideal magnitude-bound ceiling; diagnostic only),
`configs/smollm2-135m.yaml`, `experiments/phase1/<run>/` (raw results), and `tests/`.
Phase 1B adds `benchmarks/refinement_oracle.py` (driver: reference pass, every decomposition ×
mode × bound tier × tie rule, hard-failure checks), `benchmarks/refinement_report.py`
(aggregation, Phase 1A comparison, decision-gate classification),
`configs/phase1b-refinement.yaml` and `experiments/phase1b/<run>/`.
Phase 1C adds `benchmarks/refinement_runtime.py` (driver: both fallback modes per prompt,
envelope and coarse-arithmetic validation, byte audit, comparison with the Phase 1B records,
stage timing, kernel profile), `benchmarks/refinement_runtime_report.py` (aggregation, oracle
comparison, round-to-nearest what-if, gate), `benchmarks/fallback_study.py` (masked-GEMM
row independence across shapes, dtypes and devices; output-rounding probe),
`configs/phase1c-runtime.yaml` and `experiments/phase1c/<run>/`.
Phase 2 adds `benchmarks/suffix_runtime.py` (driver: reference with every intermediate, the
exact prefix of each stage, the depth-0 LM head, then every stage × budget × rounding model
× row-limit policy, with enclosure, pairwise-bound, exact-suffix, byte and single-read
checks), `benchmarks/suffix_report.py` (curves, bound provenance, gate, per-stage decision),
`configs/phase2-suffix.yaml` and `experiments/phase2/<run>/`.

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

Phase 1C runtime (`RefinementLMHead.run(h)`), for the same exact LM-head input:

1. `store.read_level(0)` (every row) → decode in chunks → binary32 `torch.mv` → centre
   s·fl(S) and the coarse interval (`bounds/coarse.py`).
2. Loop: `certify_columns` (tie-aware) → if CERTIFIED, stop; else `contenders` → read only
   those rows of the next level (or their original rows in the exact state) → float64
   intervals → intersect with the previous ones.
3. Still UNKNOWN after the exact state: MASKED fallback (the default: full-shape `F.linear`
   on the survivors only, self-tested and guarded) or FULL fallback (read every unread row,
   `F.linear`, bitwise).

Phase 2 runtime (`AdaptiveSuffixRuntime.run`), stages down and mlp:

1. `suffix_prefix(model, ids, boundary)`: the reference forward up to the boundary, all
   positions (residual, MLP input, and for down the MLP activation).
2. Read the budgeted neuron pages from the `MLPStore` (mlp: every gate row first, then up
   rows and down columns; down: down columns), largest bound contribution first.
3. Enclosures at the last position: [gate → SiLU, up → product →] down projection
   (unread columns bounded by row norms) → residual addition → final RMSNorm → h.
4. `RefinementLMHead.run_enclosure(h)`: the Phase 1C states with the input's spread;
   CERTIFIED stops. Else, if the exact state was reached: `pairwise_certificate` on the
   remaining contenders.
5. Still UNKNOWN: read the rest of the suffix, recompute it with the reference's own
   operations on all positions (bitwise the reference's o, y, h), then
   `RefinementLMHead.run(h, held=…)`: Phase 1C on the exact h, sharing the first pass's
   reads.

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
- The Phase 1C runtime touches weights only through its store. Its refined and exact
  states share their radius formulas with the oracle (`remainder_radius`,
  `exact_radius`), so only the coarse state differs from the oracle's arithmetic
  (decision 0004).
- Only the faithful rounding model certifies (decisions 0001 and 0005), for every rounding
  of the suffix: GEMM epilogues and elementwise kernels alike. Round-to-nearest-even is an
  experimental what-if: `AdaptiveSuffixRuntime` refuses it unless `experimental=True`, and
  then never sets `certified`.
- The masked LM-head fallback assumes row independence of the reference GEMM. It is used
  only in the fallback path, behind its self-test and guard, never in a certificate.
- The adaptive suffix starts after the last layer's attention, so the KV cache is written by
  the exact prefix and stays the reference's (roadmap §2.8, Mode A; tested over cached
  greedy generations).
