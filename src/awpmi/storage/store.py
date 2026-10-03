"""Page stores: rows of segments read by row index, every read counted.

`PageStore.read_rows(segment, rows)` returns the requested rows, in ascending row order, as
raw bytes [n, row_bytes] (uint8) where the store keeps them:

  InMemoryPageStore    segments are tensors held in memory (host or device); a read is a gather
  FileBackedPageStore  segments live in files; a read plans the aligned byte extents that cover
                       the requested rows, reads exactly those extents with positioned reads
                       (direct or through the OS page cache) and gathers the rows

A plan (`plan_reads`) groups consecutive requested rows into *runs* (one contiguous byte
range each), widens every run to the I/O alignment, and merges runs whose aligned ranges
touch or lie within `max_gap` bytes of each other into *extents*. A file-backed store reads
the extents and nothing else, so a row that was not requested is read only when it shares
an aligned block (or a merged gap) with a requested one, and it is never handed out.

A composed segment's row is several spans, possibly in several files (decision 0007): its
runs are its spans (merged where file and output are both contiguous), sorted by file and
offset; extents never cross files. Every run carries its output offset, so the requested
rows can also be written to chosen rows of a caller's buffer (`positions`).

`IOStats` counts, per store: requests, rows and logical bytes (the requested rows), physical
bytes (the extents actually read), read calls, extents, the distinct 4 KiB blocks of the
requested rows (the block accounting of decisions 0003 and 0004), host time in reads, and
what the OS says this process read meanwhile (`fileio.os_read_counters`).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from awpmi.storage.fileio import DIRECT_ALIGNMENT, PositionedFile, aligned_host_buffer, os_read_counters
from awpmi.storage.layout import DTYPE_NAMES, AnySegment, ComposedSegment, Segment, row_bytes_of

IO_BLOCK_BYTES = 4096
DEFAULT_MAX_EXTENT_BYTES = 8 << 20
DEFAULT_MAX_READ_BYTES = 1 << 20
_SLICE_BYTES = 64 * 1024
_BYTE_GATHER_PARTS = 256


def check_rows(rows: torch.Tensor | None, segment: AnySegment) -> torch.Tensor | None:
    """Requested rows as a CPU int64 vector, ascending and unique, within the segment (None: every row)."""
    if rows is None:
        return None
    rows = rows.reshape(-1).to("cpu", torch.int64)
    if rows.numel():
        if int(rows[0]) < 0 or int(rows[-1]) >= segment.rows:
            raise IndexError(f"{segment.name}: rows out of range [0, {segment.rows})")
        if rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()):
            raise ValueError(f"{segment.name}: rows must be ascending and unique")
    return rows


@dataclass(frozen=True)
class ReadPlan:
    """The extents a file-backed read of `rows` of `segment` must read, and where every requested byte is.

    runs          int64 [R, 4]: (file offset, length, output offset, extent) of each run, sorted by
                  (file, offset); runs never overlap
    extents       int64 [E, 2]: (file offset, length), aligned; ascending and disjoint within a file
    files         the file keys the plan reads; run_files [R] and extent_files [E] index them
    output_rows   rows of the output buffer the runs write into (the request count, unless
                  `positions` scattered the rows into a larger buffer)
    """

    segment: AnySegment
    rows: torch.Tensor | None
    row_count: int
    runs: torch.Tensor
    extents: torch.Tensor
    alignment: int
    files: tuple[str, ...]
    run_files: torch.Tensor
    extent_files: torch.Tensor
    output_rows: int

    @property
    def logical_bytes(self) -> int:
        return self.row_count * self.segment.row_bytes

    @property
    def physical_bytes(self) -> int:
        return int(self.extents[:, 1].sum()) if self.extents.numel() else 0

    @property
    def blocks_4k(self) -> int:
        """Distinct 4 KiB blocks that hold a requested byte (independent of this plan's alignment and gap)."""
        return _distinct_blocks(self.runs, IO_BLOCK_BYTES, self.run_files)


def _distinct_blocks(runs: torch.Tensor, block: int, run_files: torch.Tensor | None = None) -> int:
    if not runs.numel():
        return 0
    first = runs[:, 0] // block
    last = (runs[:, 0] + runs[:, 1] - 1) // block
    blocks = int((last - first + 1).sum())
    # Consecutive runs (ascending, disjoint) of one file can share at most a boundary block.
    shared = first[1:] == last[:-1]
    if run_files is not None:
        shared &= run_files[1:] == run_files[:-1]
    return blocks - int(shared.sum())


def plan_reads(
    segment: AnySegment,
    rows: torch.Tensor | None,
    alignment: int = DIRECT_ALIGNMENT,
    max_gap: int = 0,
    max_extent_bytes: int = DEFAULT_MAX_EXTENT_BYTES,
    positions: torch.Tensor | None = None,
) -> ReadPlan:
    """Runs and aligned extents for a read of `rows` (checked by `check_rows`) of `segment`.

    Runs merge into one extent when they are in the same file and their aligned ranges
    overlap, touch, or are at most `max_gap` bytes apart, as long as the extent stays within
    `max_extent_bytes` (a single run longer than that is an extent of its own). Request i is
    written to output row `positions[i]` (default i).
    """
    if alignment <= 0 or max_gap < 0 or max_gap % alignment or max_extent_bytes < alignment:
        raise ValueError("bad alignment, gap or extent limit")
    row_count = segment.rows if rows is None else rows.numel()
    if positions is None:
        output_rows = row_count
    else:
        positions = positions.reshape(-1).to("cpu", torch.int64)
        if positions.numel() != row_count or (row_count and int(positions.min()) < 0):
            raise ValueError("one non-negative position per requested row")
        output_rows = int(positions.max()) + 1 if row_count else 0
        if row_count > 1 and int(torch.unique(positions).numel()) != row_count:
            raise ValueError("positions must be distinct")
    files = segment.files
    found = segment.byte_runs(rows, positions)
    if not found.shape[0]:
        empty = torch.zeros(0, dtype=torch.int64)
        return ReadPlan(
            segment, rows, 0, torch.zeros(0, 4, dtype=torch.int64), torch.zeros(0, 2, dtype=torch.int64), alignment,
            files, empty, empty, output_rows,
        )
    # Sort by (file, offset); stable, so a plain segment's runs (already ascending) keep their order.
    order = torch.sort(found[:, 1], stable=True).indices
    order = order[torch.sort(found[order, 0], stable=True).indices]
    run_file, offsets, lengths, outputs = found[order].unbind(1)
    same_file = run_file[1:] == run_file[:-1]
    if bool((same_file & (offsets[1:] < offsets[:-1] + lengths[:-1])).any()):
        raise ValueError(f"{segment.name}: the requested rows overlap in their file")
    begin = offsets // alignment * alignment
    end = (offsets + lengths + alignment - 1) // alignment * alignment
    new = torch.ones_like(begin, dtype=torch.bool)
    new[1:] = ~same_file | (begin[1:] > end[:-1] + max_gap)
    extent_of_run = torch.cumsum(new.long(), 0) - 1
    group_first = new.nonzero().squeeze(1)
    group_last = torch.cat([group_first[1:] - 1, torch.tensor([begin.numel() - 1])])
    extent_begin, extent_end = begin[group_first], end[group_last]
    if bool(((extent_end - extent_begin) > max_extent_bytes).any()):
        extent_of_run, extent_begin, extent_end = _split_long_extents(begin, end, new, max_extent_bytes)
    extent_files = torch.zeros(extent_begin.numel(), dtype=torch.int64)
    extent_files[extent_of_run] = run_file
    runs = torch.stack([offsets, lengths, outputs, extent_of_run], dim=1)
    extents = torch.stack([extent_begin, extent_end - extent_begin], dim=1)
    return ReadPlan(segment, rows, row_count, runs, extents, alignment, files, run_file, extent_files, output_rows)


def _split_long_extents(begin, end, new, max_extent_bytes):
    """Greedy re-grouping that keeps multi-run extents within `max_extent_bytes`.

    Two runs that share an aligned block always stay together, so extents never overlap
    and no block is read twice.
    """
    extent_of_run, extent_begin, extent_end = [], [], []
    for start, stop, starts_group in zip(begin.tolist(), end.tolist(), new.tolist()):
        if starts_group or (stop - extent_begin[-1] > max_extent_bytes and start >= extent_end[-1]):
            extent_begin.append(start)
            extent_end.append(stop)
        else:
            extent_end[-1] = max(extent_end[-1], stop)
        extent_of_run.append(len(extent_begin) - 1)
    as_tensor = lambda values: torch.tensor(values, dtype=torch.int64)  # noqa: E731
    return as_tensor(extent_of_run), as_tensor(extent_begin), as_tensor(extent_end)


def gather_runs(
    staging: torch.Tensor,
    sources: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    row_bytes: int,
    destinations: torch.Tensor | None = None,
) -> None:
    """Copy runs (byte offsets `sources` in `staging`, `lengths`) into `out` (all uint8, 1-D).

    Run k goes to byte `destinations[k]` of `out`; by default the runs are written back to
    back. Long runs are sliced. Short runs made of whole rows that start at a row boundary
    of `out` are gathered row by row in one indexing operation; any other short run is
    gathered byte by byte, in batches.
    """
    if destinations is None:
        destinations = torch.cumsum(lengths, 0) - lengths
    row_sources, row_positions = [], []
    byte_parts: list[tuple[int, int, int]] = []
    for source, length, position in zip(sources.tolist(), lengths.tolist(), destinations.tolist()):
        if length >= _SLICE_BYTES:
            out[position : position + length].copy_(staging[source : source + length])
        elif length % row_bytes == 0 and position % row_bytes == 0:
            first = position // row_bytes
            row_sources.extend(range(source, source + length, row_bytes))
            row_positions.extend(range(first, first + length // row_bytes))
        else:
            byte_parts.append((source, length, position))
    if row_sources:
        rows = staging.unfold(0, row_bytes, 1).index_select(0, torch.tensor(row_sources, dtype=torch.int64))
        whole = out[: out.numel() // row_bytes * row_bytes].view(-1, row_bytes)
        whole.index_copy_(0, torch.tensor(row_positions, dtype=torch.int64), rows)
    for start in range(0, len(byte_parts), _BYTE_GATHER_PARTS):
        batch = torch.tensor(byte_parts[start : start + _BYTE_GATHER_PARTS], dtype=torch.int64)
        counts = batch[:, 1]
        within = torch.arange(int(counts.sum())) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        source_index = torch.repeat_interleave(batch[:, 0], counts) + within
        out_index = torch.repeat_interleave(batch[:, 2], counts) + within
        out.index_copy_(0, out_index, staging.index_select(0, source_index))


@dataclass
class IOStats:
    """Counters of one store; `reset` starts a new window (e.g. one token)."""

    requests: int = 0
    rows: int = 0
    logical_bytes: int = 0
    physical_bytes: int = 0
    read_calls: int = 0
    extents: int = 0
    blocks_4k: int = 0
    io_ms: float = 0.0
    os_read_calls: int | None = 0
    os_read_bytes: int | None = 0
    ranges: list[tuple[str, int, int]] | None = None
    by_segment: dict[str, dict[str, int]] = field(default_factory=dict)

    def reset(self, record_ranges: bool = False) -> None:
        self.__init__(ranges=[] if record_ranges else None)

    def _entry(self, segment: str) -> dict[str, int]:
        return self.by_segment.setdefault(segment, {"requests": 0, "rows": 0, "logical_bytes": 0, "physical_bytes": 0})

    def count_request(self, segment: str, rows: int, logical: int, blocks: int) -> None:
        self.requests += 1
        self.rows += rows
        self.logical_bytes += logical
        self.blocks_4k += blocks
        entry = self._entry(segment)
        entry["requests"] += 1
        entry["rows"] += rows
        entry["logical_bytes"] += logical

    def count_physical(self, segment: str, nbytes: int) -> None:
        self.physical_bytes += nbytes
        self._entry(segment)["physical_bytes"] += nbytes

    def as_dict(self) -> dict:
        data = {
            "requests": self.requests,
            "rows": self.rows,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "read_calls": self.read_calls,
            "extents": self.extents,
            "blocks_4k": self.blocks_4k,
            "os_read_calls": self.os_read_calls,
            "os_read_bytes": self.os_read_bytes,
            "by_segment": {name: dict(entry) for name, entry in self.by_segment.items()},
        }
        if self.ranges is not None:
            data["ranges"] = [list(r) for r in self.ranges]
        return data


class PageStore(ABC):
    """Rows of named segments. `device` is where `read_rows` returns them; `in_memory` stores keep them there."""

    segments: Mapping[str, Segment]
    device: torch.device
    stats: IOStats
    in_memory = False

    @abstractmethod
    def read_rows(self, segment: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Raw bytes [n, row_bytes] (uint8) of `rows` (ascending, unique; None: every row). Read-only."""

    def segment(self, name: str) -> Segment:
        try:
            return self.segments[name]
        except KeyError:
            raise KeyError(f"no segment {name!r}") from None

    @property
    def resident_bytes(self) -> int:
        """Bytes this store keeps in memory (host or device)."""
        return 0

    def close(self) -> None:
        pass


class InMemoryPageStore(PageStore):
    """Segments held as tensors in memory (all on one device). A read is a gather; physical bytes = logical.

    Either typed tensors [rows, ...] (segments are described from them), or raw row bytes
    [rows, row_bytes] (uint8) with the `segments` that describe them (e.g. a pack loaded into
    host memory, `Pack.load`).
    """

    in_memory = True

    def __init__(self, tensors: Mapping[str, torch.Tensor], segments: Mapping[str, Segment] | None = None) -> None:
        if not tensors:
            raise ValueError("no segments")
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            raise ValueError("every segment must be on the same device")
        self.device = devices.pop()  # a tensor's device always carries its index
        self._bytes: dict[str, torch.Tensor] = {}
        described = {}
        for name, tensor in tensors.items():
            data = row_bytes_of(tensor.contiguous())
            self._bytes[name] = data
            if segments is None:
                row_shape = tuple(tensor.shape[1:])
                described[name] = Segment(name, "memory", 0, data.shape[0], data.shape[1], DTYPE_NAMES[tensor.dtype], row_shape)
            else:
                info = segments[name]
                if tensor.dtype != torch.uint8 or data.shape != (info.rows, info.row_bytes):
                    raise ValueError(f"{name}: expected uint8 [{info.rows}, {info.row_bytes}] row bytes")
                described[name] = Segment(name, "memory", 0, info.rows, info.row_bytes, info.dtype, info.row_shape)
        self.segments = described
        self.stats = IOStats()

    def segment_bytes(self, segment: str) -> torch.Tensor:
        """The whole segment's row bytes [rows, row_bytes] (read-only; not counted)."""
        return self._bytes[segment]

    def count_gather(self, segment: str, rows: int) -> None:
        """Charge a gather of `rows` rows done by a caller (the streamer) to the stats."""
        nbytes = rows * self.segment(segment).row_bytes
        self.stats.count_request(segment, rows, nbytes, 0)
        self.stats.count_physical(segment, nbytes)
        self.stats.read_calls += 1

    def read_rows(self, segment: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        info = self.segment(segment)
        data = self._bytes[segment]
        if rows is None:
            result = data
        else:
            check_rows(rows, info)  # same contract as the file-backed store
            result = data.index_select(0, rows.to(data.device))
        self.count_gather(segment, result.shape[0])
        return result

    @property
    def resident_bytes(self) -> int:
        return sum(data.numel() for data in self._bytes.values())


class FileBackedPageStore(PageStore):
    """Segments in files, read with positioned reads of the planned extents only.

    `direct` bypasses the OS page cache (every read reaches the device). `workers`
    threads keep reads in flight; an extent is read in calls of at most `max_read_bytes`.
    """

    def __init__(
        self,
        files: Mapping[str, str | Path],
        segments: Mapping[str, Segment],
        direct: bool = True,
        alignment: int = DIRECT_ALIGNMENT,
        max_gap: int = 0,
        workers: int = 8,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_extent_bytes: int = DEFAULT_MAX_EXTENT_BYTES,
    ) -> None:
        if direct and alignment % DIRECT_ALIGNMENT:
            raise ValueError(f"direct I/O needs a multiple of {DIRECT_ALIGNMENT}-byte alignment")
        if max_read_bytes % alignment or max_extent_bytes % alignment:
            raise ValueError("read and extent limits must be multiples of the alignment")
        missing = {file for segment in segments.values() for file in segment.files} - set(files)
        if missing:
            raise KeyError(f"segments refer to unknown files {sorted(missing)}")
        self.files = {key: PositionedFile(path, direct) for key, path in files.items()}
        sizes = {key: file.size for key, file in self.files.items()}
        for segment in segments.values():
            if isinstance(segment, ComposedSegment):
                if not segment.spans_within(sizes):
                    raise ValueError(f"{segment.name} has a span beyond the end of its file")
            elif segment.offset + segment.nbytes > sizes[segment.file]:
                raise ValueError(f"{segment.name} extends beyond {self.files[segment.file].path}")
        self.segments = dict(segments)
        self.device = torch.device("cpu")
        self.direct = direct
        self.alignment = alignment
        self.max_gap = max_gap
        self.max_read_bytes = max_read_bytes
        self.max_extent_bytes = max_extent_bytes
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="awpmi-io") if workers > 1 else None
        self.stats = IOStats()

    def plan(self, segment: str, rows: torch.Tensor | None, positions: torch.Tensor | None = None) -> ReadPlan:
        info = self.segment(segment)
        return plan_reads(info, check_rows(rows, info), self.alignment, self.max_gap, self.max_extent_bytes, positions)

    def count_plan(self, plan: ReadPlan) -> None:
        """Charge a plan's request to the stats; the bytes its reads return are charged by `read_extents`."""
        self.stats.count_request(plan.segment.name, plan.row_count, plan.logical_bytes, plan.blocks_4k)

    def read_extents(
        self, plan: ReadPlan, extents: torch.Tensor, staging: torch.Tensor, extent_files: torch.Tensor | None = None
    ) -> None:
        """Read `extents` ([k, 2] file offset, length) back to back into `staging` (uint8, aligned).

        `extent_files` ([k], indices into `plan.files`) says which file each extent is in; it may
        be omitted when the plan reads one file.
        """
        if extents.numel() == 0:
            return
        if extent_files is None:
            if len(plan.files) != 1:
                raise ValueError("a plan over several files needs the file of every extent")
            extent_files = torch.zeros(extents.shape[0], dtype=torch.int64)
        files = [self.files[key] for key in plan.files]
        base = staging.data_ptr()
        if self.direct and base % DIRECT_ALIGNMENT:
            raise ValueError("direct reads need an aligned staging buffer")
        calls = []
        position = 0
        for (offset, length), index in zip(extents.tolist(), extent_files.tolist()):
            for start in range(0, length, self.max_read_bytes):
                size = min(self.max_read_bytes, length - start)
                calls.append((files[index], offset + start, size, base + position + start))
            position += length
        if position > staging.numel():
            raise ValueError("staging buffer too small for these extents")
        before = os_read_counters()
        started = time.perf_counter()
        if self._pool is None or len(calls) == 1:
            counts = [file.read_into(*call) for file, *call in calls]
        else:
            # One task per worker over an interleaved share of the calls: per-task overhead is paid once.
            tasks = min(self.workers, len(calls))
            shares = list(self._pool.map(lambda k: [file.read_into(*call) for file, *call in calls[k::tasks]], range(tasks)))
            counts = [0] * len(calls)
            for k, share in enumerate(shares):
                counts[k::tasks] = share
        self.stats.io_ms += (time.perf_counter() - started) * 1e3
        after = os_read_counters()
        for (file, offset, size, _), count in zip(calls, counts):
            if count < size and offset + count < min(file.size, offset + size):
                raise OSError(f"short read of {file.path} at {offset}: {count} of {size} bytes")
        self.stats.read_calls += len(calls)
        self.stats.extents += extents.shape[0]
        # Bytes the reads returned: the planned extents, less what lies beyond the end of the file.
        self.stats.count_physical(plan.segment.name, sum(counts))
        if before is None or after is None or self.stats.os_read_calls is None:
            self.stats.os_read_calls = self.stats.os_read_bytes = None
        else:
            self.stats.os_read_calls += after[0] - before[0]
            self.stats.os_read_bytes += after[1] - before[1]
        if self.stats.ranges is not None:
            self.stats.ranges.extend(
                (plan.files[index], int(o), int(n)) for (o, n), index in zip(extents.tolist(), extent_files.tolist())
            )

    def read_rows(self, segment: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Plan, read every extent into a staging buffer and gather the rows (pageable host memory).

        The `PageStreamer` uses `plan` and `read_extents` instead, with its own pinned buffers.
        """
        plan = self.plan(segment, rows)
        self.count_plan(plan)
        out = aligned_host_buffer(plan.logical_bytes, pin=False)
        if plan.row_count:
            staging = aligned_host_buffer(plan.physical_bytes, self.alignment, pin=False)
            self.read_extents(plan, plan.extents, staging, plan.extent_files)
            starts = torch.cumsum(plan.extents[:, 1], 0) - plan.extents[:, 1]
            sources = starts[plan.runs[:, 3]] + plan.runs[:, 0] - plan.extents[plan.runs[:, 3], 0]
            gather_runs(staging, sources, plan.runs[:, 1], out, plan.segment.row_bytes, plan.runs[:, 2])
        return out.view(plan.row_count, plan.segment.row_bytes)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
        for file in self.files.values():
            file.close()
