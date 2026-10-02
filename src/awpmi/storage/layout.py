"""Where weight bytes are: segments of fixed-size rows, and safetensors headers.

A *segment* is a 2-D array of fixed-size records ("rows") stored contiguously: row r
occupies bytes [offset + r·row_bytes, offset + (r+1)·row_bytes) of its file. A *page* is
one row of one segment. Whatever a runtime materializes is a set of pages: an LM-head
row, the record of a refinement level (packed codes and scale), one expert's slice of a
stacked parameter.

A safetensors tensor of shape [R, ...] is a segment as it is: R rows of
prod(shape[1:])·itemsize bytes (roadmap §3.3, safetensors first). Its header gives the
offset; no other index is needed.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
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
