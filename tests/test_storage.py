"""Phase 3 storage: segments, read plans, file-backed and in-memory stores, OS cross-check, no hidden reads."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from awpmi.storage import fileio
from awpmi.storage.layout import Segment, row_bytes_of, safetensors_segments, typed_rows
from awpmi.storage.store import IO_BLOCK_BYTES, FileBackedPageStore, InMemoryPageStore, check_rows, plan_reads

TENSORS = {
    # A small first tensor shifts every later one off the 4 KiB grid.
    "a_prefix": ((3,), torch.uint8),
    "level_records": ((700, 292), torch.uint8),
    "base_records": ((500, 436), torch.uint8),
    "exact_rows": ((300, 576), torch.bfloat16),
    "scales": ((1000,), torch.float32),
    "experts": ((6, 64, 128), torch.bfloat16),
    "bytes": ((5000, 1), torch.uint8),
}


def make_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (shape, dtype) in TENSORS.items():
        if dtype == torch.uint8:
            tensors[name] = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        else:
            tensors[name] = torch.randn(shape, generator=generator).to(dtype)
    return tensors


@pytest.fixture
def checkpoint(tmp_path):
    tensors = make_tensors()
    path = tmp_path / "weights.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def random_rows(rows: int, generator: torch.Generator, fraction: float) -> torch.Tensor:
    keep = torch.rand(rows, generator=generator) < fraction
    return keep.nonzero().squeeze(1)


def expected_bytes(tensor: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    data = row_bytes_of(tensor)
    return data if rows is None else data[rows]


def test_safetensors_segments_locate_every_tensor(checkpoint):
    path, tensors = checkpoint
    segments = safetensors_segments(path, "w")
    raw = path.read_bytes()
    for name, tensor in tensors.items():
        segment = segments[name]
        assert segment.shape == tuple(tensor.shape)
        stored = torch.frombuffer(bytearray(raw[segment.offset : segment.offset + segment.nbytes]), dtype=torch.uint8)
        assert torch.equal(stored.view(segment.rows, segment.row_bytes), row_bytes_of(tensor))
        assert torch.equal(typed_rows(stored.view(segment.rows, segment.row_bytes), segment), tensor)
    assert segments["level_records"].offset % IO_BLOCK_BYTES != 0


def brute_blocks(plan) -> set[int]:
    blocks = set()
    for offset, length, _, _ in plan.runs.tolist():
        blocks.update(range(offset // IO_BLOCK_BYTES, (offset + length - 1) // IO_BLOCK_BYTES + 1))
    return blocks


@pytest.mark.parametrize("alignment, max_gap, max_extent", [(4096, 0, 8 << 20), (512, 0, 8 << 20), (4096, 8192, 8 << 20), (4096, 0, 8192)])
def test_plan_covers_exactly_the_requested_rows(alignment, max_gap, max_extent):
    generator = torch.Generator().manual_seed(alignment + max_gap + max_extent)
    for case in range(60):
        segment = Segment("s", "f", int(torch.randint(0, 10_000, (1,), generator=generator)), 3000, [1, 7, 292, 1152, 9000][case % 5], "U8", ([1, 7, 292, 1152, 9000][case % 5],))
        fraction = [0.0, 0.002, 0.05, 0.5, 1.0][case % 5 if case % 7 else 3]
        rows = None if case % 11 == 0 else random_rows(segment.rows, generator, fraction)
        plan = plan_reads(segment, rows, alignment, max_gap, max_extent)
        wanted = list(range(segment.rows)) if rows is None else rows.tolist()
        # Runs, in order, are exactly the requested rows' bytes, written back to back.
        covered, position = [], 0
        for offset, length, output, extent in plan.runs.tolist():
            assert output == position and length % segment.row_bytes == 0 and (offset - segment.offset) % segment.row_bytes == 0
            first = (offset - segment.offset) // segment.row_bytes
            covered.extend(range(first, first + length // segment.row_bytes))
            position += length
            begin, size = plan.extents[extent].tolist()
            assert begin <= offset and offset + length <= begin + size
        assert covered == wanted and plan.logical_bytes == position
        # Extents: aligned, ascending and disjoint, each the aligned hull of its runs. Runs share an
        # extent only within the gap; a run that shares a block with the previous one always does.
        extents = plan.extents.tolist()
        for k, (begin, size) in enumerate(extents):
            assert begin % alignment == 0 and size % alignment == 0 and size > 0
            if k:
                assert begin >= extents[k - 1][0] + extents[k - 1][1]
        runs = plan.runs.tolist()
        aligned = [(o // alignment * alignment, -(-(o + n) // alignment) * alignment) for o, n, _, _ in runs]
        for k in range(len(runs)):
            extent = runs[k][3]
            assert extents[extent][0] <= aligned[k][0] and aligned[k][1] <= sum(extents[extent])
            if k:
                if runs[k - 1][3] == extent:
                    assert aligned[k][0] <= aligned[k - 1][1] + max_gap
                else:
                    assert aligned[k][0] >= aligned[k - 1][1]
                    assert aligned[k][0] > aligned[k - 1][1] + max_gap or max_extent < 8 << 20
        for k, (begin, size) in enumerate(extents):
            members = [i for i, run in enumerate(runs) if run[3] == k]
            assert members and begin == aligned[members[0]][0] and begin + size == max(aligned[i][1] for i in members)
            if size > max_extent and len(members) > 1:
                # Only a chain of runs sharing blocks can outgrow the limit.
                assert all(aligned[i][0] < aligned[i - 1][1] for i in members[1:] if aligned[i][1] - begin > max_extent)
        assert plan.blocks_4k == len(brute_blocks(plan))
        if alignment == IO_BLOCK_BYTES and max_gap == 0 and max_extent >= 8 << 20:
            # Every block read holds a requested byte: nothing is read for its own sake.
            assert plan.physical_bytes == plan.blocks_4k * IO_BLOCK_BYTES


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("workers", [1, 8])
def test_file_store_reads_exactly_the_requested_rows(checkpoint, direct, workers):
    path, tensors = checkpoint
    store = FileBackedPageStore({"w": path}, safetensors_segments(path, "w"), direct=direct, workers=workers)
    generator = torch.Generator().manual_seed(int(direct) * 10 + workers)
    try:
        for name, tensor in tensors.items():
            for fraction in (0.0, 0.01, 0.3, 1.0):
                rows = random_rows(tensor.shape[0], generator, fraction)
                store.stats.reset()
                out = store.read_rows(name, rows)
                assert torch.equal(out, expected_bytes(tensor, rows))
                assert store.stats.logical_bytes == rows.numel() * store.segment(name).row_bytes
                assert store.stats.rows == rows.numel()
            assert torch.equal(store.read_rows(name), expected_bytes(tensor, None))
    finally:
        store.close()


@pytest.mark.skipif(fileio.os_read_counters() is None, reason="no per-process I/O counters on this platform")
@pytest.mark.parametrize("direct", [True, False])
def test_physical_bytes_and_reads_match_the_os_counters(checkpoint, direct):
    path, tensors = checkpoint
    store = FileBackedPageStore({"w": path}, safetensors_segments(path, "w"), direct=direct, workers=4, max_read_bytes=8192)
    generator = torch.Generator().manual_seed(7)
    try:
        for name in ("level_records", "exact_rows", "experts", "bytes"):
            store.stats.reset()
            store.read_rows(name, random_rows(tensors[name].shape[0], generator, 0.2))
            store.read_rows(name)
            stats = store.stats
            assert stats.os_read_calls == stats.read_calls
            assert stats.os_read_bytes == stats.physical_bytes
            assert stats.physical_bytes >= stats.logical_bytes
    finally:
        store.close()


def test_no_hidden_reads(checkpoint, monkeypatch):
    """Every positioned read lies in a planned extent, the extents are read once, and each block read holds a requested byte."""
    path, tensors = checkpoint
    calls = []
    original = fileio.PositionedFile.read_into

    def recording(self, offset, length, address):
        calls.append((offset, length))
        return original(self, offset, length, address)

    monkeypatch.setattr(fileio.PositionedFile, "read_into", recording)
    store = FileBackedPageStore({"w": path}, safetensors_segments(path, "w"), direct=True, workers=4, max_read_bytes=8192)
    generator = torch.Generator().manual_seed(3)
    size = path.stat().st_size
    try:
        for name in ("level_records", "base_records", "exact_rows", "experts"):
            for fraction in (0.01, 0.1, 0.5):
                rows = random_rows(tensors[name].shape[0], generator, fraction)
                plan = store.plan(name, rows)
                calls.clear()
                store.read_rows(name, rows)
                read = sorted(calls)
                # The reads tile the planned extents exactly: no overlap, no gap, nothing outside.
                k = 0
                for offset, length in plan.extents.tolist():
                    position = offset
                    while k < len(read) and read[k][0] < offset + length:
                        assert read[k][0] == position
                        position += read[k][1]
                        k += 1
                    assert position == offset + length
                assert k == len(read)
                requested = brute_blocks(plan)
                for offset, length in read:
                    last = min(offset + length, size)
                    assert set(range(offset // IO_BLOCK_BYTES, (last - 1) // IO_BLOCK_BYTES + 1)) <= requested
    finally:
        store.close()


def test_bytes_outside_the_requested_rows_never_reach_the_output(checkpoint):
    """Overwrite every byte of a segment except the requested rows: the read returns the same rows."""
    path, tensors = checkpoint
    segments = safetensors_segments(path, "w")
    generator = torch.Generator().manual_seed(11)
    for name in ("level_records", "exact_rows"):
        segment = segments[name]
        rows = random_rows(segment.rows, generator, 0.05)
        store = FileBackedPageStore({"w": path}, segments, direct=True)
        before = store.read_rows(name, rows).clone()
        store.close()
        raw = bytearray(path.read_bytes())
        kept = set(rows.tolist())
        for row in range(segment.rows):
            if row not in kept:
                start = segment.offset + row * segment.row_bytes
                raw[start : start + segment.row_bytes] = b"\xff" * segment.row_bytes
        path.write_bytes(bytes(raw))
        store = FileBackedPageStore({"w": path}, segments, direct=True)
        try:
            assert torch.equal(store.read_rows(name, rows), before)
        finally:
            store.close()


def test_in_memory_store_gathers_and_checks_rows():
    tensors = make_tensors(1)
    store = InMemoryPageStore(tensors)
    generator = torch.Generator().manual_seed(2)
    for name, tensor in tensors.items():
        rows = random_rows(tensor.shape[0], generator, 0.3)
        assert torch.equal(store.read_rows(name, rows), expected_bytes(tensor, rows))
        assert torch.equal(typed_rows(store.read_rows(name, rows), store.segment(name)), tensor[rows])
    segment = store.segment("exact_rows")
    with pytest.raises(ValueError):
        check_rows(torch.tensor([3, 2]), segment)
    with pytest.raises(ValueError):
        check_rows(torch.tensor([2, 2]), segment)
    with pytest.raises(IndexError):
        check_rows(torch.tensor([segment.rows]), segment)
    with pytest.raises(ValueError):
        InMemoryPageStore({"a": torch.zeros(2, 2), "b": torch.zeros(2, 2, device="meta")})


def test_direct_reads_refuse_unaligned_requests(checkpoint):
    path, _ = checkpoint
    buffer = fileio.aligned_host_buffer(2 * fileio.DIRECT_ALIGNMENT, pin=False)
    with fileio.PositionedFile(path, direct=True) as file:
        assert file.read_into(0, fileio.DIRECT_ALIGNMENT, buffer.data_ptr()) == fileio.DIRECT_ALIGNMENT
        for offset, length, shift in ((1, 4096, 0), (0, 100, 0), (0, 4096, 1)):
            with pytest.raises(ValueError):
                file.read_into(offset, length, buffer.data_ptr() + shift)
    with pytest.raises(ValueError):
        FileBackedPageStore({"w": path}, safetensors_segments(path, "w"), direct=True, alignment=512)


def test_store_refuses_segments_beyond_the_file(checkpoint):
    path, _ = checkpoint
    segment = Segment("x", "w", path.stat().st_size - 10, 2, 8, "U8", (8,))
    with pytest.raises(ValueError):
        FileBackedPageStore({"w": path}, {"x": segment})
    with pytest.raises(KeyError):
        FileBackedPageStore({"other": path}, {"x": Segment("x", "w", 0, 1, 1, "U8", (1,))})
