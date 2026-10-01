from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.paging.metadata import PageBoundMetadata, compute_page_bound_metadata
from awpmi.paging.page import WeightPage


@dataclass(frozen=True)
class PageIndex:
    """All pages of one parameter, ordered by page_id, with their bound metadata."""

    parameter_name: str
    parameter_shape: tuple[int, int]
    dtype: torch.dtype
    pages: tuple[WeightPage, ...]
    bounds: PageBoundMetadata

    def __len__(self) -> int:
        return len(self.pages)

    def page(self, page_id: int) -> WeightPage:
        return self.pages[page_id]

    @property
    def total_bytes(self) -> int:
        return sum(page.storage_bytes for page in self.pages)

    @property
    def column_slices(self) -> list[slice]:
        return [page.column_slice for page in self.pages]


def column_partition(in_features: int, page_width: int) -> list[slice]:
    """Contiguous column blocks of `page_width`; the last block may be narrower."""
    if page_width <= 0:
        raise ValueError(f"page_width must be positive, got {page_width}")
    return [slice(start, min(start + page_width, in_features)) for start in range(0, in_features, page_width)]


def build_column_page_index(weight: torch.Tensor, parameter_name: str, page_width: int) -> PageIndex:
    """Partition a [out_features, in_features] weight into input-column pages.

    z = W h = Σ_p W[:, p] h_p is an exact additive decomposition over these pages.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    slices = column_partition(in_features, page_width)
    bounds = compute_page_bound_metadata(weight, slices)
    max_row_norms = bounds.max_row_norms.tolist()
    pages = tuple(
        WeightPage(
            page_id=page_id,
            parameter_name=parameter_name,
            offset=column_slice.start,
            shape=(out_features, column_slice.stop - column_slice.start),
            dtype=weight.dtype,
            storage_bytes=out_features * (column_slice.stop - column_slice.start) * weight.element_size(),
            metadata={"axis": 1, "max_row_norm": max_row_norms[page_id]},
        )
        for page_id, column_slice in enumerate(slices)
    )
    return PageIndex(
        parameter_name=parameter_name,
        parameter_shape=(out_features, in_features),
        dtype=weight.dtype,
        pages=pages,
        bounds=bounds,
    )
