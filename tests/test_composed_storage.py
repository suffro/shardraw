"""Phase 4A storage (decision 0007): composed segments, rows made of spans of several files, read in place."""

from __future__ import annotations

import json

import pytest
import torch

from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage import fileio
from awpmi.storage.cache import LRUPolicy, PageCache
from awpmi.storage.layout import ComposedSegment, segment_from_json
from awpmi.storage.pack import MANIFEST, PackWriter, SourceFile, open_pack, sha256_file_direct
from awpmi.storage.store import IO_BLOCK_BYTES, FileBackedPageStore, plan_reads
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import DEVICES

FILES = ("a", "b", "c")


def build_layout(generator: torch.Generator, rows: int, part_bytes: tuple[int, ...]):
    """Spans of every (row, part) laid out in three files in a random order, sometimes adjacent."""
    cursor = {f: int(torch.randint(0, 5000, (1,), generator=generator)) for f in FILES}
    spans: dict[tuple[int, int], tuple[str, int]] = {}
    for row in torch.randperm(rows, generator=generator).tolist():
        together = bool(torch.rand(1, generator=generator) < 0.5)
        file = FILES[int(torch.randint(0, len(FILES), (1,), generator=generator))]
        for part, size in enumerate(part_bytes):
            if not together:
                file = FILES[int(torch.randint(0, len(FILES), (1,), generator=generator))]
            choice = float(torch.rand(1, generator=generator))
            gap = 0 if together and part else (0 if choice < 0.4 else int(torch.randint(1, 100, (1,), generator=generator)) if choice < 0.7 else int(torch.randint(4096, 10000, (1,), generator=generator)))
            spans[(row, part)] = (file, cursor[file] + gap)
            cursor[file] += gap + size
    sizes = {f: cursor[f] + int(torch.randint(0, 3000, (1,), generator=generator)) for f in FILES}
    return spans, sizes


@pytest.fixture
def composed(tmp_path):
    generator = torch.Generator().manual_seed(42)
    part_bytes = (8192, 4096, 100)
    rows = 14
    spans, sizes = build_layout(generator, rows, part_bytes)
    data = {f: torch.randint(0, 256, (sizes[f],), dtype=torch.uint8, generator=generator) for f in FILES}
    paths = {}
    for f in FILES:
        paths[f] = tmp_path / f"{f}.bin"
        paths[f].write_bytes(data[f].numpy().tobytes())
    segment = ComposedSegment(
        "layer.weight", rows, sum(part_bytes), "U8", (sum(part_bytes),), part_bytes,
        tuple(tuple(spans[(r, p)] for p in range(len(part_bytes))) for r in range(rows)),
    )
    expected = torch.stack([
        torch.cat([data[spans[(r, p)][0]][spans[(r, p)][1] : spans[(r, p)][1] + n] for p, n in enumerate(part_bytes)]) for r in range(rows)
    ])
    return segment, paths, expected


def requested_blocks(plan) -> set[tuple[int, int]]:
    blocks = set()
    for (offset, length, _, _), file in zip(plan.runs.tolist(), plan.run_files.tolist()):
        blocks.update((file, b) for b in range(offset // IO_BLOCK_BYTES, (offset + length - 1) // IO_BLOCK_BYTES + 1))
    return blocks


def subsets(segment: ComposedSegment, generator: torch.Generator):
    yield None
    for fraction in (0.1, 0.4, 0.8, 1.0):
        keep = (torch.rand(segment.rows, generator=generator) < fraction).nonzero().squeeze(1)
        if keep.numel():
            yield keep


def test_composed_segment_validates_and_round_trips(composed):
    segment, _, _ = composed
    assert segment_from_json(segment.name, json.loads(json.dumps(segment.to_json()))) == segment
    assert segment.files == FILES
    with pytest.raises(ValueError):
        ComposedSegment("x", 2, 10, "U8", (10,), (4, 5), ((("a", 0), ("a", 4)), (("a", 9), ("a", 13))))  # parts make 9 bytes
    with pytest.raises(ValueError):
        ComposedSegment("x", 2, 10, "U8", (10,), (4, 6), ((("a", 0), ("a", 4)),))  # one row of spans for two rows


@pytest.mark.parametrize("max_gap, max_extent", [(0, 8 << 20), (8192, 8 << 20), (0, 8192)])
def test_composed_plan_covers_exactly_the_requested_spans(composed, max_gap, max_extent):
    segment, _, _ = composed
    generator = torch.Generator().manual_seed(max_gap + max_extent)
    for rows in subsets(segment, generator):
        count = segment.rows if rows is None else rows.numel()
        for positions in (None, torch.randperm(count + 3, generator=generator)[:count] if rows is not None else None):
            plan = plan_reads(segment, rows, IO_BLOCK_BYTES, max_gap, max_extent, positions)
            wanted = list(range(segment.rows)) if rows is None else rows.tolist()
            where = list(range(count)) if positions is None else positions.tolist()
            # Every requested byte is in exactly one run, at its output place; nothing else is.
            expected = {}
            for row, out in zip(wanted, where):
                start = 0
                for (file, offset), size in zip(segment.spans[row], segment.part_bytes):
                    for k in range(size):
                        expected[(FILES.index(file), offset + k)] = out * segment.row_bytes + start + k
                    start += size
            got = {}
            for (offset, length, output, extent), file in zip(plan.runs.tolist(), plan.run_files.tolist()):
                begin, size = plan.extents[extent].tolist()
                assert plan.extent_files[extent] == file and begin <= offset and offset + length <= begin + size
                for k in range(length):
                    assert (file, offset + k) not in got
                    got[(file, offset + k)] = output + k
            assert got == expected
            # Runs and extents are sorted by (file, offset); extents of a file are aligned and disjoint.
            keys = list(zip(plan.run_files.tolist(), plan.runs[:, 0].tolist()))
            assert keys == sorted(keys)
            extents = list(zip(plan.extent_files.tolist(), plan.extents.tolist()))
            for k, (file, (begin, size)) in enumerate(extents):
                assert begin % IO_BLOCK_BYTES == 0 and size % IO_BLOCK_BYTES == 0 and size > 0
                if k and extents[k - 1][0] == file:
                    assert begin >= sum(extents[k - 1][1])
            assert plan.blocks_4k == len(requested_blocks(plan))
            if max_gap == 0 and max_extent >= 8 << 20:
                assert plan.physical_bytes == plan.blocks_4k * IO_BLOCK_BYTES
            assert plan.output_rows == (count if positions is None else int(positions.max()) + 1)


@pytest.mark.parametrize("direct", [True, False])
def test_composed_reads_return_exactly_the_rows(composed, direct):
    segment, paths, expected = composed
    store = FileBackedPageStore(paths, {segment.name: segment}, direct=direct, workers=4, max_read_bytes=8192)
    generator = torch.Generator().manual_seed(int(direct))
    try:
        for rows in subsets(segment, generator):
            store.stats.reset()
            out = store.read_rows(segment.name, rows)
            assert torch.equal(out, expected if rows is None else expected[rows])
            if store.stats.os_read_bytes is not None:
                assert store.stats.os_read_bytes == store.stats.physical_bytes
                assert store.stats.os_read_calls == store.stats.read_calls
    finally:
        store.close()


def test_composed_reads_have_no_hidden_reads(composed, monkeypatch):
    segment, paths, _ = composed
    calls = []
    original = fileio.PositionedFile.read_into

    def recording(self, offset, length, address):
        calls.append((self.path, offset, length))
        return original(self, offset, length, address)

    monkeypatch.setattr(fileio.PositionedFile, "read_into", recording)
    store = FileBackedPageStore(paths, {segment.name: segment}, direct=True, workers=4, max_read_bytes=4096)
    generator = torch.Generator().manual_seed(5)
    try:
        for rows in subsets(segment, generator):
            plan = store.plan(segment.name, rows)
            calls.clear()
            store.read_rows(segment.name, rows)
            requested = requested_blocks(plan)
            planned = {}
            for (offset, length), file in zip(plan.extents.tolist(), plan.extent_files.tolist()):
                planned.setdefault(str(paths[plan.files[file]]), []).append((offset, length))
            for path, offset, length in calls:
                assert any(begin <= offset and offset + length <= begin + size for begin, size in planned[path])
                file = plan.files.index(next(k for k, p in paths.items() if str(p) == path))
                last = min(offset + length, paths[plan.files[file]].stat().st_size)
                assert {(file, b) for b in range(offset // IO_BLOCK_BYTES, (last - 1) // IO_BLOCK_BYTES + 1)} <= requested
            assert sum(length for _, _, length in calls) == plan.physical_bytes
    finally:
        store.close()


def test_bytes_outside_the_spans_never_reach_the_output(composed):
    segment, paths, expected = composed
    rows = torch.tensor([1, 4, 5, 11])
    keep = {f: torch.zeros(paths[f].stat().st_size, dtype=torch.bool) for f in FILES}
    for row in rows.tolist():
        for (file, offset), size in zip(segment.spans[row], segment.part_bytes):
            keep[file][offset : offset + size] = True
    for f in FILES:
        raw = torch.frombuffer(bytearray(paths[f].read_bytes()), dtype=torch.uint8)
        raw[~keep[f]] = 0xFF
        paths[f].write_bytes(raw.numpy().tobytes())
    store = FileBackedPageStore(paths, {segment.name: segment}, direct=True)
    try:
        assert torch.equal(store.read_rows(segment.name, rows), expected[rows])
    finally:
        store.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slot_bytes", [8192, 64 * 1024, 8 << 20])
def test_streamer_writes_composed_rows_into_chosen_rows_of_a_buffer(composed, device, slot_bytes):
    segment, paths, expected = composed
    store = FileBackedPageStore(paths, {segment.name: segment}, direct=True, max_extent_bytes=16384)
    streamer = PageStreamer(device, slot_bytes=slot_bytes)
    generator = torch.Generator().manual_seed(slot_bytes)
    try:
        for rows in subsets(segment, generator):
            count = segment.rows if rows is None else rows.numel()
            streamer.stats.reset()
            out = streamer.fetch(store, segment.name, rows)
            assert torch.equal(out.cpu(), expected if rows is None else expected[rows])
            if rows is None:
                continue
            buffer = torch.zeros(count + 2, segment.row_bytes, dtype=torch.uint8, device=device)
            positions = torch.randperm(count + 2, generator=generator)[:count]
            streamer.stats.reset()
            streamer.fetch(store, segment.name, rows, out=buffer, positions=positions)
            assert torch.equal(buffer.cpu()[positions], expected[rows])
            untouched = torch.ones(count + 2, dtype=torch.bool)
            untouched[positions] = False
            assert int(buffer.cpu()[untouched].sum()) == 0
            if device == "cuda":
                assert streamer.stats.h2d_bytes == count * segment.row_bytes
    finally:
        store.close()
        streamer.close()


@pytest.mark.parametrize("device", DEVICES)
def test_streamer_copies_long_runs_straight_from_staging(tmp_path, device):
    """Pieces mixing runs above and below the direct-copy size, scattered in the destination."""
    generator = torch.Generator().manual_seed(17)
    part_bytes = (300_000, 4096, 700)
    rows = 9
    spans, sizes = build_layout(generator, rows, part_bytes)
    data = {f: torch.randint(0, 256, (sizes[f],), dtype=torch.uint8, generator=generator) for f in FILES}
    paths = {f: tmp_path / f"{f}.bin" for f in FILES}
    for f in FILES:
        paths[f].write_bytes(data[f].numpy().tobytes())
    segment = ComposedSegment(
        "big", rows, sum(part_bytes), "U8", (sum(part_bytes),), part_bytes,
        tuple(tuple(spans[(r, p)] for p in range(len(part_bytes))) for r in range(rows)),
    )
    expected = torch.stack([
        torch.cat([data[spans[(r, p)][0]][spans[(r, p)][1] : spans[(r, p)][1] + n] for p, n in enumerate(part_bytes)]) for r in range(rows)
    ])
    store = FileBackedPageStore(paths, {"big": segment}, direct=True)
    streamer = PageStreamer(device, slot_bytes=1 << 20)
    try:
        rows_wanted = torch.tensor([0, 2, 3, 5, 8])
        buffer = torch.zeros(7, segment.row_bytes, dtype=torch.uint8, device=device)
        positions = torch.tensor([6, 0, 3, 1, 4])
        streamer.fetch(store, "big", rows_wanted, out=buffer, positions=positions)
        assert torch.equal(buffer.cpu()[positions], expected[rows_wanted])
        assert int(buffer.cpu()[[2, 5]].sum()) == 0
        if device == "cuda":
            assert streamer.stats.h2d_bytes == 5 * segment.row_bytes
            assert streamer.stats.gathered_bytes < 5 * (4096 + 700) + 1  # only the short runs were gathered
    finally:
        store.close()
        streamer.close()


@pytest.mark.parametrize("device", DEVICES)
def test_backend_assembles_hits_and_misses_into_a_caller_buffer(composed, device):
    segment, paths, expected = composed
    store = FileBackedPageStore(paths, {segment.name: segment}, direct=True)
    backend = MaterializationBackend(store, device, PageStreamer(device), PageCache(4 * segment.row_bytes, LRUPolicy()))
    try:
        backend.materialize(segment.name, torch.tensor([2, 7]))
        backend.reset_stats()
        rows = torch.tensor([0, 2, 5, 7, 9])
        out = torch.full((5, segment.row_bytes), 7, dtype=torch.uint8, device=device)
        assert backend.materialize(segment.name, rows, out=out) is out
        assert torch.equal(out.cpu(), expected[rows])
        served = backend.report()["materialization"]
        assert served["cache_hit_rows"] == 2 and served["fetched_rows"] == 3
        assert backend.report()["storage"]["logical_bytes"] == 3 * segment.row_bytes
        with pytest.raises(ValueError):
            backend.materialize(segment.name, rows, out=out[:4])
    finally:
        store.close()


def test_pack_index_of_composed_segments_refers_to_files_in_place(composed, tmp_path):
    segment, paths, expected = composed
    writer = PackWriter(tmp_path / "index", "test-index")
    declared = {f: sha256_file_direct(paths[f]) for f in FILES}
    for f in FILES:
        writer.add_source(f, SourceFile("example/split", "0" * 40, f"{f}.bin"), paths[f], sha256=declared[f])
    writer.add_composed_segment(segment)
    pack = writer.write({"note": "index"})
    manifest = json.loads((tmp_path / "index" / MANIFEST).read_text())
    assert manifest["format_version"] == 2 and manifest["segments"][segment.name]["sha256"] is None
    assert all(entry["sha256_from"] == "publisher" for entry in manifest["files"].values())
    assert not any(p.suffix == ".safetensors" for p in (tmp_path / "index").iterdir())  # nothing copied
    resolve = lambda source: paths[source.filename.removesuffix(".bin")]  # noqa: E731
    reopened = open_pack(tmp_path / "index", verify="files", resolve=resolve)
    assert reopened.segments == pack.segments
    with pytest.raises(ValueError):
        open_pack(tmp_path / "index", verify="segments", resolve=resolve)  # no segment digests to check
    store = reopened.store(direct=True)
    try:
        assert torch.equal(store.read_rows(segment.name, torch.tensor([3, 4])), expected[[3, 4]])
    finally:
        store.close()
    raw = bytearray(paths["b"].read_bytes())
    raw[len(raw) // 2] ^= 1
    paths["b"].write_bytes(bytes(raw))
    open_pack(tmp_path / "index", verify="size", resolve=resolve)
    with pytest.raises(ValueError):
        open_pack(tmp_path / "index", verify="files", resolve=resolve)


def test_store_refuses_spans_beyond_their_file(composed):
    segment, paths, _ = composed
    (file, offset), size = segment.spans[0][0], segment.part_bytes[0]
    bad = ComposedSegment(
        "bad", 1, segment.row_bytes, "U8", (segment.row_bytes,), segment.part_bytes,
        (((file, paths[file].stat().st_size - 10), *segment.spans[0][1:]),),
    )
    with pytest.raises(ValueError):
        FileBackedPageStore(paths, {"bad": bad})
