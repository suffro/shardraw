"""Packed store of one refinement-decomposed weight: every read counted, wherever the bytes live.

It holds what a deployment of the decomposition stores (decisions 0004 and 0006):

  levels      one record per row and level: the bit-packed codes
              (`awpmi.decomposition.packing`) followed by the row's float32 scale
              (little-endian), ceil(K·b/8) + 4 bytes. That is the unit decision 0004
              charges, so a row moving into a state is one contiguous read.
  exact rows  the original weight, row-major: the checkpoint's own bytes
  metadata    per-row remainder norms for every non-exact state, resident on the compute device

The runtime reaches weight values only through `read_level`, `read_exact` and
`read_fallback`. Each read is logged with the rows it returned and the bytes those rows
occupy, so a run's effective (logical) bytes are counted from what was actually handed
out, and tests can check that no row was read that the certificate did not ask for.

Where the bytes come from is the `MaterializationBackend`'s concern (Phase 3):

  `from_decomposition`  segments resident in memory on the weight's device (Phase 1C)
  `from_pack`           the segments of a refinement pack (`write_refinement_pack`), read from
                        storage through a page store, a streamer and optionally a page cache

The values handed out are the same bytes either way, so the runtime's arithmetic, and
therefore its certificates and fallbacks, are identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from awpmi.bounds.remainder import RowNormBounds
from awpmi.decomposition.base import RefinementDecomposition
from awpmi.decomposition.packing import CODE_DTYPE, pack_codes, payload_bytes, unpack_codes
from awpmi.decomposition.quantization import SCALE_DTYPE
from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage.layout import DTYPE_NAMES, SAFETENSORS_DTYPES
from awpmi.storage.pack import Pack, PackWriter, SourceFile
from awpmi.storage.store import InMemoryPageStore
from awpmi.tracing import tensor_digest

BYTE_PARTS = (
    "metadata",
    "base_payload",
    "base_scales",
    "refinement_payload",
    "refinement_scales",
    "exact_rows",
    "fallback_rows",
)
PACK_KIND = "refinement-lm-head"
SCALE_BYTES = SCALE_DTYPE.itemsize
EXACT_SEGMENT = "exact"


def level_segment(level: int) -> str:
    return f"level{level}"


def norm_segment(state: int, norm: str) -> str:
    return f"remainder{state}.{norm}"


@dataclass(frozen=True)
class RefinementLayout:
    """Shape of a stored decomposition: rows, columns, dtype, and the bits of each stored level."""

    out_features: int
    in_features: int
    dtype: torch.dtype
    bits: tuple[int, ...]

    def payload_bytes(self, level: int) -> int:
        return payload_bytes(self.in_features, self.bits[level])

    def record_bytes(self, level: int) -> int:
        return self.payload_bytes(level) + SCALE_BYTES

    def to_json(self) -> dict:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "dtype": DTYPE_NAMES[self.dtype],
            "bits": list(self.bits),
        }

    @classmethod
    def from_json(cls, data: dict) -> RefinementLayout:
        return cls(data["out_features"], data["in_features"], SAFETENSORS_DTYPES[data["dtype"]], tuple(data["bits"]))


def level_records(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Records [V, width + 4] (uint8): packed codes, then the float32 scale's bytes."""
    if scales.dtype != SCALE_DTYPE or scales.shape != (payload.shape[0],):
        raise ValueError("one float32 scale per packed row is required")
    return torch.cat([payload, scales.contiguous().view(torch.uint8).view(-1, SCALE_BYTES)], dim=1)


def split_records(records: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(payload [n, width] uint8, scales [n] float32), both contiguous, from records [n, width + 4]."""
    payload = records[:, :width].contiguous()
    scales = records[:, width : width + SCALE_BYTES].contiguous().view(SCALE_DTYPE).reshape(-1)
    return payload, scales


def packed_levels(decomposition: RefinementDecomposition) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(payload, scales) of every stored level, checked to unpack to the decomposition's codes exactly."""
    levels = []
    for level in decomposition.levels:
        if not level.bits or level.codes is None or level.scales is None:
            raise ValueError("a zero base level stores nothing; the runtime needs a quantized base")
        payload = pack_codes(level.codes, level.bits)
        if payload.shape[1] != level.payload_bytes_per_row:
            raise ValueError(f"{level.name}: packed width {payload.shape[1]} != accounted {level.payload_bytes_per_row}")
        if not torch.equal(unpack_codes(payload, level.bits, level.in_features), level.codes.to(CODE_DTYPE)):
            raise ArithmeticError(f"{level.name}: packing is not lossless")
        levels.append((payload, level.scales.clone()))
    return levels


@dataclass(frozen=True)
class Read:
    """One logged read. `rows` is None when every row was read."""

    kind: str  # "level", "exact" or "fallback"
    state: int  # the state the rows move into (a level index, or the exact state)
    rows: torch.Tensor | None
    row_count: int
    payload_bytes: int
    scale_bytes: int


class PackedRefinementStore:
    def __init__(
        self, layout: RefinementLayout, remainder_norms: Sequence[RowNormBounds], backend: MaterializationBackend
    ) -> None:
        if not layout.bits or len(remainder_norms) != len(layout.bits):
            raise ValueError("one remainder-norm table per stored level is required")
        for level in range(len(layout.bits)):
            segment = backend.segment(level_segment(level))
            if segment.rows != layout.out_features or segment.row_bytes != layout.record_bytes(level):
                raise ValueError(f"level {level}: stored records do not match the layout")
        exact = backend.segment(EXACT_SEGMENT)
        if exact.rows != layout.out_features or exact.row_bytes != layout.in_features * layout.dtype.itemsize:
            raise ValueError("the exact rows do not match the layout")
        if SAFETENSORS_DTYPES[exact.dtype] != layout.dtype:
            raise ValueError("the exact rows' dtype does not match the layout")
        self.layout = layout
        self.backend = backend
        self._norms = tuple(
            RowNormBounds(norms.l2.to(backend.device), norms.linf.to(backend.device)) for norms in remainder_norms
        )
        self._log: list[Read] = []

    @classmethod
    def from_decomposition(cls, decomposition: RefinementDecomposition) -> PackedRefinementStore:
        """A resident store on the weight's device (Phase 1C): pack every level, keep it in memory."""
        weight = decomposition.weight
        tensors = {level_segment(k): level_records(*packed) for k, packed in enumerate(packed_levels(decomposition))}
        tensors[EXACT_SEGMENT] = weight
        store = InMemoryPageStore({name: tensor.to(weight.device) for name, tensor in tensors.items()})
        layout = RefinementLayout(weight.shape[0], weight.shape[1], weight.dtype, tuple(l.bits for l in decomposition.levels))
        return cls(layout, decomposition.remainder_norms, MaterializationBackend(store, weight.device))

    @classmethod
    def from_pack(cls, pack: Pack, backend: MaterializationBackend) -> PackedRefinementStore:
        """A store over a refinement pack's segments; `backend` reads them (its store must serve the pack)."""
        if pack.kind != PACK_KIND:
            raise ValueError(f"{pack.directory} is a {pack.kind!r} pack")
        layout = RefinementLayout.from_json(pack.metadata["layout"])
        reader = pack.store(direct=True)
        try:
            norms = [
                RowNormBounds(
                    reader.read_rows(norm_segment(state, "l2")).clone().view(torch.float32).reshape(-1),
                    reader.read_rows(norm_segment(state, "linf")).clone().view(torch.float32).reshape(-1),
                )
                for state in range(len(layout.bits))
            ]
        finally:
            reader.close()
        return cls(layout, norms, backend)

    # Structure

    @property
    def out_features(self) -> int:
        return self.layout.out_features

    @property
    def in_features(self) -> int:
        return self.layout.in_features

    @property
    def dtype(self) -> torch.dtype:
        return self.layout.dtype

    @property
    def device(self) -> torch.device:
        return self.backend.device

    @property
    def num_levels(self) -> int:
        return len(self.layout.bits)

    @property
    def exact_state(self) -> int:
        return self.num_levels

    @property
    def num_states(self) -> int:
        return self.num_levels + 1

    def level_bits(self, level: int) -> int:
        return self.layout.bits[level]

    def remainder_norms(self, state: int) -> RowNormBounds:
        """Resident metadata for the remainder after `state`; charged once per run in `metadata`."""
        return self._norms[state]

    @property
    def metadata_bytes(self) -> int:
        return sum(norms.nbytes for norms in self._norms)

    @property
    def exact_row_bytes(self) -> int:
        return self.in_features * self.dtype.itemsize

    @property
    def weight_bytes(self) -> int:
        return self.out_features * self.exact_row_bytes

    def row_bytes(self, state: int) -> tuple[int, int]:
        """(payload, scale) bytes that move one row into `state`."""
        if state == self.exact_state:
            return self.exact_row_bytes, 0
        return self.layout.payload_bytes(state), SCALE_BYTES

    def storage_bytes(self) -> dict[str, int]:
        levels = sum(self.out_features * sum(self.row_bytes(state)) for state in range(self.exact_state))
        storage = {"levels": levels, "original": self.weight_bytes, "metadata": self.metadata_bytes}
        storage["total"] = sum(storage.values())
        return storage

    def content_digest(self) -> str:
        """sha256 of everything stored (payloads, scales, metadata, original weight), as in Phase 1C.

        Reads every segment once from the backend's store, outside the read log.
        """
        store = self.backend.store
        levels = []
        for level in range(self.num_levels):
            records = store.read_rows(level_segment(level)).to("cpu")
            levels.extend(split_records(records, self.layout.payload_bytes(level)))
        weight = store.read_rows(EXACT_SEGMENT).to("cpu").view(self.dtype).view(self.out_features, self.in_features)
        return tensor_digest(
            *levels,
            *[tensor for norms in self._norms for tensor in (norms.l2, norms.linf)],
            weight,
        )

    # Reads

    def begin_run(self) -> None:
        """Start a new read log (one run of the runtime)."""
        self._log = []

    @property
    def reads(self) -> tuple[Read, ...]:
        return tuple(self._log)

    def read_level(self, level: int, rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed codes and scales of `rows` (every row if None; else ascending and unique)."""
        records = self.backend.materialize(level_segment(level), rows)
        count = records.shape[0]
        width = self.layout.payload_bytes(level)
        self._log.append(Read("level", level, rows, count, count * width, count * SCALE_BYTES))
        return split_records(records, width)

    def read_exact(self, rows: torch.Tensor) -> torch.Tensor:
        """Original rows, for rows entering the exact state."""
        return self._read_weight("exact", rows)

    def read_fallback(self, rows: torch.Tensor) -> torch.Tensor:
        """Original rows not read before, for the conservative fallback."""
        return self._read_weight("fallback", rows)

    def _read_weight(self, kind: str, rows: torch.Tensor) -> torch.Tensor:
        self._log.append(Read(kind, self.exact_state, rows, rows.numel(), rows.numel() * self.exact_row_bytes, 0))
        data = self.backend.materialize(EXACT_SEGMENT, rows)
        return data.view(self.dtype).view(rows.numel(), self.in_features)

    def bytes_read(self) -> dict[str, int]:
        """Effective bytes of the current run, in the breakdown of `RefinementDecomposition.materialized_bytes`."""
        breakdown = dict.fromkeys(BYTE_PARTS, 0)
        breakdown["metadata"] = self.metadata_bytes
        for read in self._log:
            if read.kind == "level":
                prefix = "base" if read.state == 0 else "refinement"
                breakdown[f"{prefix}_payload"] += read.payload_bytes
                breakdown[f"{prefix}_scales"] += read.scale_bytes
            else:
                breakdown[f"{read.kind}_rows"] += read.payload_bytes
        breakdown["total"] = sum(breakdown.values())
        return breakdown


def write_refinement_pack(
    decomposition: RefinementDecomposition,
    directory: str | Path,
    source: tuple[SourceFile, str] | None = None,
    packing: dict | None = None,
    source_path: str | Path | None = None,
) -> Pack:
    """Write a refinement pack: level records and remainder norms in `pack.safetensors`, the exact rows
    either as the tensor `source[1]` of the checkpoint file `source[0]` (whose bytes must be the
    weight's; `source_path` overrides where that file is) or, without a source, as a copy in the pack.
    """
    weight = decomposition.weight
    writer = PackWriter(directory, PACK_KIND)
    for level, packed in enumerate(packed_levels(decomposition)):
        writer.add_tensor(level_segment(level), level_records(*packed))
    for state, norms in enumerate(decomposition.remainder_norms):
        writer.add_tensor(norm_segment(state, "l2"), norms.l2)
        writer.add_tensor(norm_segment(state, "linf"), norms.linf)
    if source is None:
        writer.add_tensor(EXACT_SEGMENT, weight)
    else:
        writer.add_source("source", source[0], source_path)
        writer.add_source_segment(EXACT_SEGMENT, "source", source[1], expected=weight)
    layout = RefinementLayout(weight.shape[0], weight.shape[1], weight.dtype, tuple(l.bits for l in decomposition.levels))
    levels_sha256 = tensor_digest(
        *[t for level in decomposition.levels for t in (level.codes, level.scales)],
        *[t for norms in decomposition.remainder_norms for t in (norms.l2, norms.linf)],
    )
    metadata = {
        "decomposition": decomposition.spec,
        "layout": layout.to_json(),
        "record": {"payload": "bit-packed biased codes, little-endian bitstream", "scale": "float32 little-endian, after the payload"},
        "levels_sha256": levels_sha256,
        "weight_sha256": tensor_digest(weight),
        "source": None if source is None else {**source[0].to_json(), "tensor": source[1]},
    }
    return writer.write(metadata, packing)
