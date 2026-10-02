"""Conservative interval on every reference logit given a set of materialized pages.

Let M be the materialized pages, U the remaining ones, N[j, p] ≥ ‖W[j, p]‖₂ and
n_p ≥ ‖h_p‖₂. With ẑ_j = Σ_{p∈M} W[j,p]·h_p evaluated in float64:

    exact logit      z_j ∈ ẑ_j ± ( r_j + γ₆₄ · S_j )
    accumulator      a_j ∈ z_j ± γ_acc · S_j            (reference GEMM, any order)
    reference logit  ℓ_j = rnd_out(a_j)                  (faithful rounding to output dtype)

where r_j = Σ_{p∈U} N[j,p]·n_p bounds the missing contribution and
S_j = Σ_p N[j,p]·n_p ≥ Σ_k |W_jk h_k| (Cauchy–Schwarz per page) bounds the
absolute mass used by both floating-point error terms. Because faithful rounding
is monotone, ℓ_j ∈ [round_down(a_j⁻), round_up(a_j⁺)] on the output grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.floating import (
    FLOAT64_BOUND_SLACK,
    FLOAT64_UNIT_ROUNDOFF,
    MAX_BOUND_TERMS,
    gamma,
    next_down,
    next_up,
    round_down_to_grid,
    round_up_to_grid,
)

# Four times the IEEE binary32 unit roundoff. Covers round-to-nearest (u),
# truncating accumulators (2u) and tensor-core block-FMA alignment effects; the
# assumption is validated empirically by the reference-envelope check.
FP32_ACCUMULATION_UNIT_ROUNDOFF = 2.0**-22


@dataclass(frozen=True)
class ReferenceNumerics:
    """Conservative model of how the reference computes one logit.

    `output_dtype` is the dtype the reference GEMM writes (its rounding grid),
    `accumulation_unit_roundoff` bounds the relative error of each accumulator
    operation, and `reduction_length` is the dot-product length K.
    """

    output_dtype: torch.dtype
    accumulation_unit_roundoff: float
    reduction_length: int

    @property
    def accumulation_gamma(self) -> float:
        # K products + K-1 additions per logit; +2 covers an input rounding and the epilogue.
        return gamma(self.reduction_length + 2, self.accumulation_unit_roundoff)

    @property
    def float64_gamma(self) -> float:
        # Partial logits: per-page dot products plus the sum over pages, ≤ 2K operations.
        return gamma(2 * self.reduction_length + 2, FLOAT64_UNIT_ROUNDOFF)


class ResidualBounder:
    """Per-input bound engine: maps (partial logits, remaining pages) to logit intervals."""

    def __init__(
        self,
        row_page_norms: torch.Tensor,
        hidden_page_norms: torch.Tensor,
        numerics: ReferenceNumerics,
    ) -> None:
        if row_page_norms.dtype != torch.float64 or hidden_page_norms.dtype != torch.float64:
            raise TypeError("bound inputs must be float64")
        if row_page_norms.shape[1] != hidden_page_norms.shape[0]:
            raise ValueError("row_page_norms and hidden_page_norms disagree on page count")
        if row_page_norms.shape[1] > MAX_BOUND_TERMS:
            raise ValueError("too many pages for FLOAT64_BOUND_SLACK to be valid")
        self.row_page_norms = row_page_norms
        self.hidden_page_norms = hidden_page_norms
        self.numerics = numerics
        absolute_mass = row_page_norms @ hidden_page_norms
        self._error_radius = (numerics.accumulation_gamma + numerics.float64_gamma) * absolute_mass

    def residual_radius(self, remaining_mask: torch.Tensor) -> torch.Tensor:
        """r_j = Σ_{p∈U} N[j,p]·n_p, the bound on the missing contribution."""
        weights = torch.where(remaining_mask, self.hidden_page_norms, torch.zeros_like(self.hidden_page_norms))
        return self.row_page_norms @ weights

    def logit_bounds(
        self, partial_logits: torch.Tensor, remaining_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower/upper bounds on every reference logit, on the reference output grid."""
        radius = self.residual_radius(remaining_mask) + self._error_radius
        return reference_logit_interval(partial_logits, radius, self.numerics.output_dtype)


def reference_logit_interval(
    center: torch.Tensor, radius: torch.Tensor, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output-grid interval enclosing every reference logit within `radius` of `center`.

    `radius` is a float64 sum of non-negative terms that already bounds the missing
    contribution and both floating-point error terms; FLOAT64_BOUND_SLACK covers the
    rounding of that sum.
    """
    radius = radius * (1.0 + FLOAT64_BOUND_SLACK)
    # One more float64 rounding in each of the additions below: step outward one ulp.
    lower = round_down_to_grid(next_down(center - radius), output_dtype)
    upper = round_up_to_grid(next_up(center + radius), output_dtype)
    return lower, upper
