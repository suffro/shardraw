"""Error model of the runtime's coarse pass over every vocabulary row (Phase 1C).

The coarse pass reads a b-bit level L with integer codes |q_jk| ≤ limit and float32
scales s_j, and computes for every row

    S_j = Σ_k q_jk·h_k          in binary32 (`awpmi.refinement_head.coarse_matvec`), any order
    c_j = s_j · fl(S_j)         in float64, exact (two 24-bit significands)

Arithmetic error. A length-K binary32 dot product satisfies
|fl(S) − S| ≤ γ_K(u)·Σ_k |q_k h_k| for any evaluation order (Higham, *Accuracy and
Stability of Numerical Algorithms*, §3.1); the bound also covers rounded products,
although they are exact for BF16 inputs. The runtime takes u = 2⁻²², four times the
binary32 unit roundoff as for the reference accumulator (decision 0001), and γ_{K+2},
so truncating or tensor-core accumulation would still be covered. Flush-to-zero
arithmetic loses at most 2⁻¹²⁶ per flushed input, product or partial sum, hence the
absolute term τ = K·(limit + 2)·2⁻¹²⁶ in code units.

Absolute mass without extra reads. ‖q_j‖_∞ ≤ limit, so by Hölder

    s_j·Σ_k |q_jk|·|h_k| ≤ B_j := s_j·limit·‖h‖₁

and therefore |c_j − L_j·h| ≤ γ·B_j + s_j·τ. B_j also stands in for the level's exact
absolute mass in the reference accumulator's error term and in the centre's float64
term, where the oracle uses the exact mass (`awpmi.oracle.refinement`). It costs no
metadata and no second pass over the codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.floating import gamma, next_up
from awpmi.bounds.residual import ReferenceNumerics, remainder_radius

SMALLEST_NORMAL_FP32 = 2.0**-126


@dataclass(frozen=True)
class CoarseArithmetic:
    """How the runtime evaluates the coarse dot products: binary32 with unit roundoff `unit_roundoff`."""

    unit_roundoff: float
    reduction_length: int

    @property
    def gamma(self) -> float:
        # K products and K − 1 additions; +2 leaves room for an alpha/beta epilogue.
        return gamma(self.reduction_length + 2, self.unit_roundoff)

    def underflow(self, limit: int) -> float:
        """τ: flush-to-zero loss per dot product, in code units."""
        return self.reduction_length * (limit + 2) * SMALLEST_NORMAL_FP32


def coded_mass_bound(scales: torch.Tensor, limit: int, input_l1: torch.Tensor) -> torch.Tensor:
    """B_j ≥ s_j·Σ_k |q_jk|·|h_k| for every row, from the float32 scales alone (float64)."""
    # s_j·limit is exact in float64 (24 + 7 bits); one rounding remains, stepped over upward.
    return next_up(scales.to(torch.float64) * limit * input_l1)


def coarse_error_bound(
    scales: torch.Tensor, mass_bound: torch.Tensor, limit: int, arithmetic: CoarseArithmetic
) -> torch.Tensor:
    """Bound on |s_j·fl(S_j) − s_j·S_j|: γ·B_j + s_j·τ (a float64 sum of non-negative terms)."""
    return arithmetic.gamma * mass_bound + scales.to(torch.float64) * arithmetic.underflow(limit)


def coarse_radius(
    mass_bound: torch.Tensor, miss: torch.Tensor, arithmetic_error: torch.Tensor, numerics: ReferenceNumerics
) -> torch.Tensor:
    """Radius around c_j = s_j·fl(S_j) that encloses the reference accumulator, for every row.

    `miss` bounds the missing remainder's absolute mass, so B_j + miss_j ≥ Σ_k |W_jk h_k|
    bounds the reference's absolute mass. This is the oracle's realistic radius with
    the level's exact mass replaced by B_j, plus the coarse arithmetic error. Pass it
    to `reference_logit_interval` with the centre.
    """
    mass = mass_bound + miss
    return remainder_radius(miss, mass, mass, numerics) + arithmetic_error
