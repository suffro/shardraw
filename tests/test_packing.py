"""Bit-packed level storage: lossless round trip, exact layout, byte sizes that match the accounting."""

from __future__ import annotations

import pytest
import torch

from awpmi.decomposition import RefinementDecomposition
from awpmi.decomposition.base import RefinementLevel
from awpmi.decomposition.packing import CodeLayout, pack_codes, payload_bytes, unpack_codes
from awpmi.stores.refinement import PackedRefinementStore
from tests.conftest import DEVICES


def random_codes(rows: int, columns: int, bits: int, seed: int) -> torch.Tensor:
    limit = 2 ** (bits - 1) - 1
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(-limit, limit + 1, (rows, columns), generator=generator)
    codes[0] = limit  # both extremes, every column
    codes[1] = -limit
    return codes.to(torch.int8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("columns", [1, 7, 13, 48, 576])
def test_pack_unpack_round_trip(device, bits, columns):
    codes = random_codes(37, columns, bits, seed=bits * 1000 + columns).to(device)
    packed = pack_codes(codes, bits)
    assert packed.dtype == torch.uint8 and packed.shape == (37, payload_bytes(columns, bits))
    assert torch.equal(unpack_codes(packed, bits, columns), codes.to(torch.int16))
    # Rows are independent: any subset decodes on its own.
    rows = torch.tensor([36, 2, 2, 0], device=device)
    assert torch.equal(unpack_codes(packed.index_select(0, rows), bits, columns), codes[rows].to(torch.int16))


def test_bitstream_layout_is_little_endian_and_biased():
    """6-bit codes [−31, 0, 31, 1] are stored biased as [0, 31, 62, 32], code k at bits 6k..6k+5."""
    packed = pack_codes(torch.tensor([[-31, 0, 31, 1]], dtype=torch.int8), 6)
    stream = (0 << 0) | (31 << 6) | (62 << 12) | (32 << 18)
    assert packed.tolist() == [[stream & 255, (stream >> 8) & 255, (stream >> 16) & 255]]
    assert packed.tolist() == [[192, 231, 131]]


def test_pack_rejects_codes_outside_the_range():
    with pytest.raises(ValueError):
        pack_codes(torch.tensor([[8]], dtype=torch.int8), 4)
    with pytest.raises(TypeError):
        pack_codes(torch.zeros(2, 3), 4)
    with pytest.raises(ValueError):
        CodeLayout(4, 10, "cpu").unpack(torch.zeros(2, 4, dtype=torch.uint8))


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_payload_size_is_the_accounted_size(bits):
    for columns in (576, 13):
        level = RefinementLevel(bits, columns, None, None)
        assert payload_bytes(columns, bits) == level.payload_bytes_per_row


@pytest.mark.parametrize("device", DEVICES)
def test_store_packs_every_level_and_counts_reads(device):
    weight = (torch.randn(40, 24, generator=torch.Generator().manual_seed(3)) * 0.1).to(torch.bfloat16).to(device)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    store = PackedRefinementStore.from_decomposition(decomposition)
    assert store.num_states == decomposition.num_states and store.exact_state == 2
    assert store.storage_bytes() == decomposition.storage_bytes()
    assert store.metadata_bytes == decomposition.metadata_bytes()
    for state in range(store.num_states):
        assert store.row_bytes(state) == decomposition.row_bytes(state)

    store.begin_run()
    payload, scales = store.read_level(0)
    assert payload.shape == (40, 18) and scales.shape == (40,)
    rows = torch.tensor([3, 17], device=device)
    payload, scales = store.read_level(1, rows)
    assert torch.equal(unpack_codes(payload, 4, 24), decomposition.levels[1].codes[rows].to(torch.int16))
    assert torch.equal(store.read_exact(rows[:1]), weight[rows[:1]])
    assert torch.equal(store.read_fallback(rows[1:]), weight[rows[1:]])
    assert [(read.kind, read.state, read.row_count) for read in store.reads] == [
        ("level", 0, 40),
        ("level", 1, 2),
        ("exact", 2, 1),
        ("fallback", 2, 1),
    ]
    assert store.bytes_read() == decomposition.materialized_bytes([40, 2, 1], fallback_rows=1)
    store.begin_run()
    assert store.reads == () and store.bytes_read()["total"] == store.metadata_bytes


def test_store_refuses_lossy_packing(monkeypatch):
    """The round-trip guard fires: a packing that does not decode to the decomposition's codes is refused."""
    import awpmi.stores.refinement as store_module

    weight = torch.randn(8, 16).to(torch.bfloat16)
    decomposition = RefinementDecomposition.build(weight, "q6")
    monkeypatch.setattr(store_module, "unpack_codes", lambda packed, bits, columns: unpack_codes(packed, bits, columns) // 2)
    with pytest.raises(ArithmeticError):
        PackedRefinementStore.from_decomposition(decomposition)


def test_store_refuses_a_decomposition_without_a_stored_base():
    weight = torch.randn(8, 16).to(torch.bfloat16)
    with pytest.raises(ValueError):
        PackedRefinementStore.from_decomposition(RefinementDecomposition.build(weight, "none"))
