from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.linear import block_l2_norm_upper

# Stored metadata is rounded *up* to this grid, so it stays an upper bound while
# taking half the memory of float64.
METADATA_DTYPE = torch.float32


@dataclass(frozen=True)
class PageBoundMetadata:
    """Precomputed bound metadata: row_page_norms[j, p] ≥ ‖W[j, page p]‖₂.

    This is what lets the residual of an unmaterialized page be bounded without
    reading its values.
    """

    row_page_norms: torch.Tensor

    @property
    def max_row_norms(self) -> torch.Tensor:
        return self.row_page_norms.max(dim=0).values

    @property
    def nbytes(self) -> int:
        return self.row_page_norms.numel() * self.row_page_norms.element_size()


def compute_page_bound_metadata(weight: torch.Tensor, column_slices: list[slice]) -> PageBoundMetadata:
    norms = block_l2_norm_upper(weight, column_slices, storage_dtype=METADATA_DTYPE)
    return PageBoundMetadata(row_page_norms=norms.to(METADATA_DTYPE))
