# 0001 — The certificate targets the floating-point reference logits

Status: accepted (Phase 1, 2026-10-01)

## Context

The roadmap's certificate (`lower[w] > max_{j≠w} upper[j]`) is stated for intervals
on the logits. The reference logits are not exact real numbers: with the default
BF16 model they are the output of a BF16 GEMM with an FP32 accumulator, rounded to
BF16, and the reference token is `torch.argmax` over them (first index on ties).

Bounds on the exact product `W h` are therefore not enough:

- Two exact logits that differ can round to the same BF16 value, and argmax then
  picks the lower index. A constructed case (`tests/test_certificate.py::
  test_bf16_rounding_tie_is_not_certified`) has exact argmax 1 and reference
  argmax 0. With output-grid rounding disabled, the certificate certified token 1,
  which is wrong. This was observed when the guard was deliberately removed.
- The reference accumulator has its own rounding error. With that term set to 0,
  real reference logits fell outside the envelope (1 of 8192 on the BF16 grid,
  594 of 8192 without grid rounding).
- By default `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`
  is True. cuBLAS may then reduce split-K partial sums in BF16, and no useful
  error bound exists for that.

## Decision

1. **A declared numerical environment** (`awpmi.runtime.configure_reproducible_numerics`).
   It sets deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no TF32,
   and no reduced-precision (BF16/FP16) reductions. It applies to the reference and
   to AWPMI alike and is recorded in every run's `environment.json`. It does not
   modify the model. It pins the reference arithmetic to its most precise
   configuration, so that the reference can be modelled soundly.
2. **An accumulation model** (`awpmi.bounds.residual.ReferenceNumerics`).
   |accumulator − exact| ≤ γ_{K+2}(u_acc) · S_j, with u_acc = 2⁻²² (4× the binary32
   unit roundoff, which covers truncating and tensor-core accumulation). S_j is the
   per-page Cauchy–Schwarz bound on Σ|W_jk h_k|. Every benchmark checks this
   empirically: each reference logit must lie in the full-materialization envelope
   (`envelope_violations` must be 0, and a violation is a hard failure).
3. **Directed rounding onto the output grid** (`awpmi.bounds.floating`). Lower bounds
   are rounded down and upper bounds rounded up to the reference output dtype. Any
   faithful rounding is monotone, so this holds whatever rounding mode the backend
   uses. **Re-examined in decision 0004:** a probe found round-to-nearest-even on this
   platform, which would certify 85 of Phase 1C's 96 fallbacks. The faithful model is
   kept. **Confirmed in decision 0005 (Phase 2):** the faithful model is the certified one
   for every rounding of the adaptive suffix too (GEMM epilogues and elementwise kernels);
   round-to-nearest-even is an experimental what-if that never certifies.
4. **A strict certificate**, exactly as the roadmap states it. Strict separation of
   the grid-rounded bounds implies a unique reference maximum, so tie-breaking
   never matters.

All float64 bound arithmetic is made conservative: norms are inflated by γ and
rounded up, radii get a relative slack of 1e-10, and sums are moved one ulp
outward.

## Rejected

- *Certificate on exact real logits.* Shown to be unsound, as described above.
- *Switching the reference to FP32 to sidestep BF16 ties.* The roadmap requires BF16
  where supported. FP32 is still supported via `model.dtype: float32`, and the
  same machinery applies with the FP32 grid.
- *Tie-aware relaxation* (allowing `lower[w] == upper[j]` when j > w). It would be
  sound, but it adds a rule that depends on the reference's tie-breaking for little
  gain. It can be revisited if the coverage lost to exact grid ties matters.
  **Revisited in decision 0003:** it is now available as `TieBreak.LOWEST_INDEX`.
  `TieBreak.STRICT` remains the default, which keeps the Phase 1A behaviour.
