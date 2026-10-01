from __future__ import annotations

import pytest
import torch

from awpmi.bounds.linear import block_l2_norm_upper
from awpmi.models.smollm2 import LM_HEAD_PARAMETER, lm_head_weight
from awpmi.paging.index import build_column_page_index, column_partition
from awpmi.paging.metadata import METADATA_DTYPE
from awpmi.paging.source import InMemoryPageSource


@pytest.mark.parametrize(("in_features", "width"), [(576, 32), (576, 64), (100, 32), (7, 8), (5, 1)])
def test_column_partition_covers_every_column_once(in_features, width):
    slices = column_partition(in_features, width)
    covered = [c for s in slices for c in range(s.start, s.stop)]
    assert covered == list(range(in_features))
    assert all(0 < s.stop - s.start <= width for s in slices)


def test_column_partition_rejects_invalid_width():
    with pytest.raises(ValueError):
        column_partition(10, 0)


@pytest.mark.parametrize("width", [32, 48, 64])
def test_page_index_describes_pages_exactly(width):
    weight = torch.randn(300, 160).to(torch.bfloat16)
    index = build_column_page_index(weight, "w", width)
    assert len(index) == -(-160 // width)
    assert [p.page_id for p in index.pages] == list(range(len(index)))
    assert index.total_bytes == weight.numel() * weight.element_size()
    for page in index.pages:
        assert page.shape == (300, page.column_slice.stop - page.column_slice.start)
        assert page.storage_bytes == 300 * page.width * 2
        assert page.dtype == torch.bfloat16
    assert index.bounds.row_page_norms.shape == (300, len(index))
    assert index.bounds.row_page_norms.dtype == METADATA_DTYPE


def test_in_memory_source_returns_exact_slices_and_counts_fetches():
    weight = torch.randn(64, 40).to(torch.bfloat16)
    index = build_column_page_index(weight, "w", 16)
    source = InMemoryPageSource(weight)
    pages = [source.get(page) for page in index.pages]
    for page, values in zip(index.pages, pages):
        assert torch.equal(values, weight[:, page.offset : page.offset + page.width])
    assert torch.equal(torch.cat(pages, dim=1), weight)
    assert source.fetch_count == len(index)
    assert source.bytes_fetched == index.total_bytes


@pytest.mark.model
@pytest.mark.parametrize("width", [32, 64])
def test_lm_head_metadata_is_an_upper_bound(model, width):
    weight = lm_head_weight(model)
    index = build_column_page_index(weight, LM_HEAD_PARAMETER, width)
    norms64 = block_l2_norm_upper(weight, index.column_slices).cpu()
    stored = index.bounds.row_page_norms.to(torch.float64).cpu()
    assert bool((stored >= norms64).all())
    # Rounding up to float32 must stay tight: within one float32 ulp.
    assert bool(((stored - norms64) <= norms64 * 2.0**-23).all())
    assert index.parameter_shape == tuple(weight.shape)
    assert index.total_bytes == weight.numel() * weight.element_size()
