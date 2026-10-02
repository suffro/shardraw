# 0003 — Precision-refinement decompositions: oracle methodology, byte accounting, gate

Status: accepted (Phase 1B, 2026-10-01)

## Context

Phase 1A (input-column pages with Cauchy–Schwarz bounds) was correct, but
certification needed 98.5% of the LM head on average. Its oracle showed that the
column decomposition, not the ordering, was the binding constraint
(`history/2026-10-01-awpmi-phase1-report.md`). Phase 1B looks for a better
decomposition. It evaluates candidates with an oracle before any runtime, storage
or streaming work.

## Decision

1. **The reference is unchanged.** It is still the unmodified BF16 SmolLM2 forward.
   A candidate is an exact additive decomposition of the original weights,
   W = L₀ + L₁ + … + L_{n−1} + X, never a separate quantized model.
   - Each level is a per-row symmetric round-to-nearest int-b quantization of the
     current remainder, with a float32 scale (`awpmi.decomposition`).
   - Each level is exact in float64. `RefinementDecomposition.build` checks every
     remainder subtraction with TwoSum and refuses an inexact one.
   - The final state of a row is its **original BF16 row**, charged in full. The
     fallback is still the reference operation over the full BF16 matrix
     (decision 0002).
2. **The realistic bound** on a missing remainder row x is
   min(‖x‖₂‖h‖₂, ‖x‖_∞‖h‖₁) ≥ Σ_k |x_k h_k| ≥ |x·h| (`awpmi.bounds.remainder`).
   - The norms are resident float32 metadata, rounded up: 8 bytes per row for
     each non-exact state.
   - The same quantity bounds the remainder's share of the accumulation-error
     term.
   - In the exact state, the accumulation term uses the exact absolute mass
     Σ_k |W_jk h_k|. This is tighter than Phase 1's per-page Cauchy–Schwarz
     value, so every run re-validates it with the envelope check.
   - Decision 0001's accumulation model, float64 error term and output-grid
     directed rounding are reused unchanged
     (`awpmi.bounds.residual.reference_logit_interval`).
3. **Two diagnostic tiers, never usable at runtime** (`awpmi.oracle.refinement`):
   - `abs_mass`: Σ_k |x_k h_k|, the best sign-agnostic bound;
   - `ideal`: |x·h|, the true missing contribution.

   They separate the losses due to the bound from those due to the
   decomposition.
4. **Row-selective refinement.**
   - A row is eliminated once some row is provably at least as large and wins
     the comparison (`awpmi.certificate.contenders`). Only the remaining rows
     advance to the next state.
   - A row's intervals from successive states are intersected.
   - Eliminated rows can never block a certificate, so global and row-selective
     refinement certify at the same state. Tests assert this.
5. **The tie-aware certificate** (`TieBreak.LOWEST_INDEX`) is now implemented. It
   accepts lower[w] ≥ upper[j] for j > w, and still requires a strict
   inequality for j < w.
   - It relies on `torch.argmax` returning the lowest index among equal maxima,
     which a test asserts in the reference's own form.
   - `TieBreak.STRICT` stays the default, so Phase 1A results are unchanged (they
     were re-checked bitwise).
   - This revisits the relaxation that decision 0001 deferred.
6. **Byte accounting.** The effective bytes of a decision are:
   - all resident bound metadata;
   - the base level for every row (bit-packed codes plus float32 scales);
   - every refinement row and exact row read;
   - for a fallback, every BF16 row not yet read (conservative).

   Fractions are relative to the original BF16 LM head (56,623,104 bytes). Two
   secondary measures are reported but are not primary:
   - 4 KiB-block-granular I/O;
   - a *masked fallback*, which recomputes only the remaining rows in a
     full-shape GEMM and is checked bitwise against the reference.
7. **Decision gate, fixed before the full run**
   (`configs/phase1b-refinement.yaml`). It is evaluated on the primary
   configuration: realistic bound, `lowest_index` ties, row-selective mode,
   conservative fallback.
   - PROMISING: realistic mean fraction ≤ 0.60, and the realistic saving is at
     least half the ideal saving of the same decomposition.
   - NOT PROMISING, decomposition-limited: the ideal mean fraction is above 0.5.
   - NOT PROMISING, bound-limited: anything else.

## Rejected

- *A quantized model as the reference.* It would change the question: AWPMI
  certifies the original model's decision.
- *Probabilistic or confidence-based bounds on the remainder.* They are not
  conservative. The ideal tier shows what they would be chasing.
- *Counting only payload bytes.* Scales, bound metadata and fallback reads are
  real costs. Phase 1A's 25% metadata overhead at width 8 showed they can cancel
  any saving.
- *Encoding the exact remainder compactly in Phase 1B.* Charging the full BF16
  row is conservative and matches what the checkpoint already stores. A compact
  exact remainder is a Phase 1C question.
