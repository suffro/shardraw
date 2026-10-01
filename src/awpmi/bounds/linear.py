"""Cauchy–Schwarz bounds for a column-paged linear map z = W h.

For a column block p of W and the matching block h_p of the input:

    |(W_p h_p)[j]| ≤ ‖W[j, p]‖₂ · ‖h_p‖₂

`block_l2_norm_upper` returns float64 upper bounds on these norms that remain
valid despite the rounding of the norm computation itself.
"""

from __future__ import annotations

import torch

from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma, next_up, round_up_to_grid


def block_l2_norm_upper(
    values: torch.Tensor,
    column_slices: list[slice],
    storage_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Upper bounds on ‖values[..., s]‖₂ for every slice s, stacked on the last axis.

    `values` is [..., K]; the result is [..., len(column_slices)] in float64 holding
    values from the `storage_dtype` grid. Each entry is ≥ the exact real norm.
    """
    blocks = []
    for column_slice in column_slices:
        block = values[..., column_slice].to(torch.float64)
        width = block.shape[-1]
        norm = torch.sqrt((block * block).sum(dim=-1))
        # Sum of `width` squares (γ_width) and the sqrt (one rounding) can each
        # under-estimate; inflate by γ_{width+2} and round the product upward.
        inflated = next_up(norm * (1.0 + gamma(width + 2, FLOAT64_UNIT_ROUNDOFF)))
        blocks.append(round_up_to_grid(inflated, storage_dtype))
    return torch.stack(blocks, dim=-1)


def page_contribution(weight_page: torch.Tensor, hidden_block: torch.Tensor) -> torch.Tensor:
    """W_p h_p in float64. Error ≤ γ_width(2⁻⁵³)·Σ|W_jk h_k|, accounted for in residual bounds."""
    return weight_page.to(torch.float64) @ hidden_block.to(torch.float64)
