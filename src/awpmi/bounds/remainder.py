"""Bounds on the contribution of a missing additive remainder.

When W = A + X and only the approximation A has been materialized, every row x of
the remainder X contributes x·h to the logit, and for any input h

    |x·h| ≤ Σ_k |x_k|·|h_k| ≤ min( ‖x‖₂·‖h‖₂ , ‖x‖_∞·‖h‖₁ )      (Cauchy–Schwarz, Hölder)

The middle term (the remainder's absolute mass) also bounds the remainder's share of
the reference accumulator's error term, so one bound serves both purposes. The
row norms are resident metadata, rounded up onto the float32 grid; the input norms
are computed per input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.floating import next_up, round_up_to_grid
from awpmi.bounds.linear import block_l2_norm_upper, l1_norm_upper

METADATA_DTYPE = torch.float32


@dataclass(frozen=True)
class RowNormBounds:
    """Per-row upper bounds on ‖x_j‖₂ and ‖x_j‖_∞ of a remainder, stored as float32."""

    l2: torch.Tensor
    linf: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.l2.numel() * self.l2.element_size() + self.linf.numel() * self.linf.element_size()


@dataclass(frozen=True)
class InputNorms:
    """Upper bounds on ‖h‖₂ and ‖h‖₁ (float64), one entry per input."""

    l2: torch.Tensor
    l1: torch.Tensor


def row_norm_bounds(remainder: torch.Tensor) -> RowNormBounds:
    """Metadata for a float64 remainder [V, K]."""
    if remainder.dtype != torch.float64:
        raise TypeError("remainders are analysed in float64")
    whole_row = [slice(0, remainder.shape[1])]
    l2 = block_l2_norm_upper(remainder, whole_row, storage_dtype=METADATA_DTYPE)[:, 0]
    linf = round_up_to_grid(remainder.abs().amax(dim=1), METADATA_DTYPE)
    return RowNormBounds(l2=l2.to(METADATA_DTYPE), linf=linf.to(METADATA_DTYPE))


def input_norms(hidden: torch.Tensor) -> InputNorms:
    """`hidden` is [B, K] (or [K]); its values are taken exactly in float64."""
    hidden = hidden.to(torch.float64)
    l2 = block_l2_norm_upper(hidden, [slice(0, hidden.shape[-1])])[..., 0]
    return InputNorms(l2=l2, l1=l1_norm_upper(hidden))


def remainder_mass_bound(rows: RowNormBounds, inputs: InputNorms) -> torch.Tensor:
    """Upper bound on Σ_k |x_jk|·|h_k| for every row j and input: [V] for one input, [V, B] for a batch."""
    l2 = rows.l2.to(torch.float64)
    linf = rows.linf.to(torch.float64)
    if inputs.l2.dim() == 0:
        cauchy_schwarz, holder = l2 * inputs.l2, linf * inputs.l1
    else:
        cauchy_schwarz, holder = torch.outer(l2, inputs.l2), torch.outer(linf, inputs.l1)
    # Each product rounds once; step one ulp up so the result stays an upper bound.
    return next_up(torch.minimum(cauchy_schwarz, holder))
