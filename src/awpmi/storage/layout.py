"""Where weight bytes are: segments of fixed-size rows, and safetensors headers.

A *segment* is a 2-D array of fixed-size records ("rows") stored contiguously: row r
occupies bytes [offset + r·row_bytes, offset + (r+1)·row_bytes) of its file. A *page* is
one row of one segment. Whatever a runtime materializes is a set of pages: an LM-head
row, the record of a refinement level (packed codes and scale), one expert's slice of a
stacked parameter.

A safetensors tensor of shape [R, ...] is a segment as it is: R rows of
prod(shape[1:])·itemsize bytes (roadmap §3.3, safetensors first). Its header gives the
offset; no other index is needed.

A *composed segment* has the same rows, but each row is assembled from byte spans that
need not be contiguous, nor in one file: row r is the concatenation of its parts, part p
being `part_bytes[p]` bytes at (file, offset) `spans[r][p]`. It describes a logical tensor
that a checkpoint stores as several tensors (Phase 4A, decision 0007), e.g. one row per
expert made of two published tensors, without copying them.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import torch

SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
DTYPE_NAMES: dict[torch.dtype, str] = {dtype: name for name, dtype in SAFETENSORS_DTYPES.items()}


@dataclass(frozen=True)
class Segment:
    """`rows` records of `row_bytes` bytes at `offset` in file `file` (a key of the store's file table)."""

    name: str
    file: str
    offset: int
    rows: int
    row_bytes: int
    dtype: str  # safetensors name of the element dtype of a row
    row_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{self.name}: unknown dtype {self.dtype!r}")
        itemsize = SAFETENSORS_DTYPES[self.dtype].itemsize
        if self.rows <= 0 or self.offset < 0 or self.row_bytes != math.prod(self.row_shape) * itemsize:
            raise ValueError(f"{self.name}: inconsistent segment {self}")

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def torch_dtype(self) -> torch.dtype:
        return SAFETENSORS_DTYPES[self.dtype]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, *self.row_shape)

    @property
    def files(self) -> tuple[str, ...]:
        return (self.file,)

    def byte_runs(self, rows: torch.Tensor | None, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Runs of `rows` (checked, ascending; None: every row): int64 [R, 4] (file index, offset, length, output offset).

        Request i goes to output row `positions[i]` (default i). Consecutive rows whose output
        rows are consecutive too form one run; runs come in request order.
        """
        if rows is None:
            if positions is not None:
                raise ValueError("positions need explicit rows")
            return torch.tensor([[0, self.offset, self.nbytes, 0]], dtype=torch.int64)
        count = rows.numel()
        if count == 0:
            return torch.zeros(0, 4, dtype=torch.int64)
        positions = torch.arange(count) if positions is None else positions.reshape(-1).to("cpu", torch.int64)
        if positions.numel() != count:
            raise ValueError("one position per requested row")
        breaks = (rows[1:] != rows[:-1] + 1) | (positions[1:] != positions[:-1] + 1)
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), breaks.nonzero().squeeze(1) + 1])
        stops = torch.cat([starts[1:], torch.tensor([count])])
        lengths = (stops - starts) * self.row_bytes
        offsets = self.offset + rows[starts] * self.row_bytes
        outputs = positions[starts] * self.row_bytes
        return torch.stack([torch.zeros_like(offsets), offsets, lengths, outputs], dim=1)

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "offset": self.offset,
            "rows": self.rows,
            "row_bytes": self.row_bytes,
            "dtype": self.dtype,
            "row_shape": list(self.row_shape),
        }

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> Segment:
        return cls(
            name, data["file"], int(data["offset"]), int(data["rows"]), int(data["row_bytes"]), data["dtype"],
            tuple(int(n) for n in data["row_shape"]),
        )


@dataclass(frozen=True)
class ComposedSegment:
    """`rows` records of `row_bytes` bytes; row r is the concatenation of spans[r][p] for every part p.

    Span (file, offset) of part p holds part_bytes[p] bytes of file `file` (a key of the store's
    file table). Parts have the same sizes in every row; rows may lie in different files.
    """

    name: str
    rows: int
    row_bytes: int
    dtype: str
    row_shape: tuple[int, ...]
    part_bytes: tuple[int, ...]
    spans: tuple[tuple[tuple[str, int], ...], ...]  # [rows][parts] (file key, offset)

    def __post_init__(self) -> None:
        if self.dtype not in SAFETENSORS_DTYPES:
            raise ValueError(f"{self.name}: unknown dtype {self.dtype!r}")
        itemsize = SAFETENSORS_DTYPES[self.dtype].itemsize
        if self.rows <= 0 or self.row_bytes != math.prod(self.row_shape) * itemsize:
            raise ValueError(f"{self.name}: inconsistent row shape {self.row_shape} for {self.row_bytes} bytes")
        if not self.part_bytes or any(n <= 0 for n in self.part_bytes) or sum(self.part_bytes) != self.row_bytes:
            raise ValueError(f"{self.name}: parts {self.part_bytes} do not make a row of {self.row_bytes} bytes")
        if len(self.spans) != self.rows or any(len(row) != len(self.part_bytes) for row in self.spans):
            raise ValueError(f"{self.name}: expected {self.rows} rows of {len(self.part_bytes)} spans")
        if any(offset < 0 for row in self.spans for _, offset in row):
            raise ValueError(f"{self.name}: negative offset")

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def torch_dtype(self) -> torch.dtype:
        return SAFETENSORS_DTYPES[self.dtype]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, *self.row_shape)

    @cached_property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted({file for row in self.spans for file, _ in row}))

    @cached_property
    def _tables(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = {file: k for k, file in enumerate(self.files)}
        files = torch.tensor([[index[file] for file, _ in row] for row in self.spans], dtype=torch.int64)
        offsets = torch.tensor([[offset for _, offset in row] for row in self.spans], dtype=torch.int64)
        lengths = torch.tensor(self.part_bytes, dtype=torch.int64)
        starts = torch.cumsum(lengths, 0) - lengths
        return files, offsets, lengths, starts

    def spans_within(self, sizes: dict[str, int]) -> bool:
        """Whether every span lies within its file, given the file sizes."""
        return all(offset + n <= sizes[file] for row in self.spans for (file, offset), n in zip(row, self.part_bytes))

    def byte_runs(self, rows: torch.Tensor | None, positions: torch.Tensor | None = None) -> torch.Tensor:
        """As `Segment.byte_runs`: one run per span, merged with the next when both file and output are contiguous."""
        if rows is None:
            if positions is not None:
                raise ValueError("positions need explicit rows")
            rows = torch.arange(self.rows)
        count = rows.numel()
        if count == 0:
            return torch.zeros(0, 4, dtype=torch.int64)
        positions = torch.arange(count) if positions is None else positions.reshape(-1).to("cpu", torch.int64)
        if positions.numel() != count:
            raise ValueError("one position per requested row")
        files, offsets, lengths, starts = self._tables
        parts = lengths.numel()
        file = files[rows].reshape(-1)
        offset = offsets[rows].reshape(-1)
        length = lengths.repeat(count)
        output = (positions[:, None] * self.row_bytes + starts[None, :]).reshape(-1)
        new = torch.ones_like(file, dtype=torch.bool)
        new[1:] = (file[1:] != file[:-1]) | (offset[1:] != offset[:-1] + length[:-1]) | (output[1:] != output[:-1] + length[:-1])
        if parts * count == 1 or bool(new.all()):
            return torch.stack([file, offset, length, output], dim=1)
        first = new.nonzero().squeeze(1)
        run = torch.cumsum(new.long(), 0) - 1
        merged = torch.zeros(first.numel(), dtype=torch.int64).index_add_(0, run, length)
        return torch.stack([file[first], offset[first], merged, output[first]], dim=1)

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "row_bytes": self.row_bytes,
            "dtype": self.dtype,
            "row_shape": list(self.row_shape),
            "part_bytes": list(self.part_bytes),
            "spans": [[[file, offset] for file, offset in row] for row in self.spans],
        }

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> ComposedSegment:
        return cls(
            name, int(data["rows"]), int(data["row_bytes"]), data["dtype"], tuple(int(n) for n in data["row_shape"]),
            tuple(int(n) for n in data["part_bytes"]),
            tuple(tuple((str(file), int(offset)) for file, offset in row) for row in data["spans"]),
        )


AnySegment = Segment | ComposedSegment


def segment_from_json(name: str, data: dict[str, Any]) -> AnySegment:
    """A segment of a manifest: composed when it lists spans."""
    return ComposedSegment.from_json(name, data) if "spans" in data else Segment.from_json(name, data)


def read_safetensors_header(path: str | Path) -> tuple[int, dict[str, Any]]:
    """(data_start, header): the absolute offset of the data section and the parsed JSON header."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return 8 + length, header


def safetensors_segments(path: str | Path, file_key: str) -> dict[str, Segment]:
    """Every tensor of at least one dimension in a safetensors file, as a segment of shape[0] rows."""
    data_start, header = read_safetensors_header(path)
    segments = {}
    for name, entry in header.items():
        if name == "__metadata__" or not entry["shape"]:
            continue
        rows, *row_shape = entry["shape"]
        begin, end = entry["data_offsets"]
        if rows == 0:
            continue
        segment = Segment(
            name, file_key, data_start + begin, rows, (end - begin) // rows, entry["dtype"], tuple(row_shape)
        )
        if segment.nbytes != end - begin:
            raise ValueError(f"{path}: {name} is not {rows} rows of equal size")
        segments[name] = segment
    return segments


def typed_rows(data: torch.Tensor, segment: Segment) -> torch.Tensor:
    """View raw row bytes [n, row_bytes] (uint8, contiguous) as [n, *row_shape] of the segment's dtype."""
    if data.dtype != torch.uint8 or data.dim() != 2 or data.shape[1] != segment.row_bytes:
        raise ValueError(f"expected uint8 [rows, {segment.row_bytes}] for {segment.name}")
    return data.view(segment.torch_dtype).view(data.shape[0], *segment.row_shape)


def row_bytes_of(tensor: torch.Tensor) -> torch.Tensor:
    """The bytes of a contiguous [rows, ...] tensor as uint8 [rows, row_bytes] (a view, no copy)."""
    if not tensor.is_contiguous() or tensor.dim() == 0:
        raise ValueError("expected a contiguous tensor with a row dimension")
    rows = tensor.shape[0]
    flat = tensor.reshape(rows, -1)
    if flat.dtype == torch.bool:
        flat = flat.view(torch.uint8)
    return flat.view(torch.uint8).view(rows, -1)
