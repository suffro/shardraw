"""Packed store of one refinement-decomposed weight (Phase 1C): in memory, every read counted.

It holds what a deployment of the decomposition stores (decision 0004):

  levels      bit-packed codes (`awpmi.decomposition.packing`) and a float32 scale per row
  exact rows  the original weight, row-major: the checkpoint's own bytes
  metadata    per-row remainder norms for every non-exact state, resident

The runtime reaches weight values only through `read_level`, `read_exact` and
`read_fallback`. Each read is logged with the rows it returned and the bytes those
rows occupy, so a run's effective bytes are counted from what was actually handed
out, and tests can check that no row was read that the certificate did not ask
for. In Phase 1C everything is resident and a read is a gather from memory; a
file-backed store (Phase 3) has to provide the same reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from awpmi.bounds.remainder import RowNormBounds
from awpmi.decomposition.base import RefinementDecomposition
from awpmi.decomposition.packing import CODE_DTYPE, pack_codes, unpack_codes
from awpmi.decomposition.quantization import SCALE_DTYPE, code_limit
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


@dataclass(frozen=True)
class PackedLevel:
    """One stored level: packed codes [V, width] (uint8) and scales [V] (float32)."""

    bits: int
    payload: torch.Tensor
    scales: torch.Tensor

    @property
    def limit(self) -> int:
        return code_limit(self.bits)

    @property
    def payload_bytes_per_row(self) -> int:
        return self.payload.shape[1]

    @property
    def scale_bytes_per_row(self) -> int:
        return self.scales.element_size()


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
        self, levels: Sequence[PackedLevel], remainder_norms: Sequence[RowNormBounds], weight: torch.Tensor
    ) -> None:
        if weight.dim() != 2:
            raise ValueError("expected a [out_features, in_features] weight")
        if not levels or len(levels) != len(remainder_norms):
            raise ValueError("one remainder-norm table per stored level is required")
        rows = weight.shape[0]
        for level in levels:
            if level.payload.shape[0] != rows or level.scales.shape != (rows,) or level.scales.dtype != SCALE_DTYPE:
                raise ValueError("every level needs one packed row and one float32 scale per output row")
        self._levels = tuple(levels)
        self._norms = tuple(remainder_norms)
        self._weight = weight
        self._log: list[Read] = []

    @classmethod
    def from_decomposition(cls, decomposition: RefinementDecomposition) -> PackedRefinementStore:
        """Pack every level and check that unpacking returns the decomposition's codes exactly."""
        levels = []
        for level in decomposition.levels:
            if not level.bits or level.codes is None or level.scales is None:
                raise ValueError("a zero base level stores nothing; the runtime needs a quantized base")
            payload = pack_codes(level.codes, level.bits)
            if payload.shape[1] != level.payload_bytes_per_row:
                raise ValueError(f"{level.name}: packed width {payload.shape[1]} != accounted {level.payload_bytes_per_row}")
            if not torch.equal(unpack_codes(payload, level.bits, level.in_features), level.codes.to(CODE_DTYPE)):
                raise ArithmeticError(f"{level.name}: packing is not lossless")
            levels.append(PackedLevel(level.bits, payload, level.scales.clone()))
        return cls(levels, decomposition.remainder_norms, decomposition.weight)

    # Structure

    @property
    def out_features(self) -> int:
        return self._weight.shape[0]

    @property
    def in_features(self) -> int:
        return self._weight.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self._weight.dtype

    @property
    def device(self) -> torch.device:
        return self._weight.device

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    @property
    def exact_state(self) -> int:
        return len(self._levels)

    @property
    def num_states(self) -> int:
        return len(self._levels) + 1

    def level_bits(self, level: int) -> int:
        return self._levels[level].bits

    def remainder_norms(self, state: int) -> RowNormBounds:
        """Resident metadata for the remainder after `state`; charged once per run in `metadata`."""
        return self._norms[state]

    @property
    def metadata_bytes(self) -> int:
        return sum(norms.nbytes for norms in self._norms)

    @property
    def exact_row_bytes(self) -> int:
        return self.in_features * self._weight.element_size()

    @property
    def weight_bytes(self) -> int:
        return self._weight.numel() * self._weight.element_size()

    def row_bytes(self, state: int) -> tuple[int, int]:
        """(payload, scale) bytes that move one row into `state`."""
        if state == self.exact_state:
            return self.exact_row_bytes, 0
        level = self._levels[state]
        return level.payload_bytes_per_row, level.scale_bytes_per_row

    def storage_bytes(self) -> dict[str, int]:
        levels = sum(self.out_features * sum(self.row_bytes(state)) for state in range(self.exact_state))
        storage = {"levels": levels, "original": self.weight_bytes, "metadata": self.metadata_bytes}
        storage["total"] = sum(storage.values())
        return storage

    def content_digest(self) -> str:
        """sha256 of everything stored: packed levels, scales, metadata and the original weight."""
        return tensor_digest(
            *[tensor for level in self._levels for tensor in (level.payload, level.scales)],
            *[tensor for norms in self._norms for tensor in (norms.l2, norms.linf)],
            self._weight,
        )

    # Reads

    def begin_run(self) -> None:
        """Start a new read log (one run of the runtime)."""
        self._log = []

    @property
    def reads(self) -> tuple[Read, ...]:
        return tuple(self._log)

    def read_level(self, level: int, rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed codes and scales of `rows` (every row if None). The result must be treated as read-only."""
        stored = self._levels[level]
        if rows is None:
            payload, scales, count = stored.payload, stored.scales, self.out_features
        else:
            payload, scales, count = stored.payload.index_select(0, rows), stored.scales.index_select(0, rows), rows.numel()
        self._log.append(
            Read("level", level, rows, count, count * stored.payload_bytes_per_row, count * stored.scale_bytes_per_row)
        )
        return payload, scales

    def read_exact(self, rows: torch.Tensor) -> torch.Tensor:
        """Original rows, for rows entering the exact state."""
        return self._read_weight("exact", rows)

    def read_fallback(self, rows: torch.Tensor) -> torch.Tensor:
        """Original rows not read before, for the conservative fallback."""
        return self._read_weight("fallback", rows)

    def _read_weight(self, kind: str, rows: torch.Tensor) -> torch.Tensor:
        self._log.append(Read(kind, self.exact_state, rows, rows.numel(), rows.numel() * self.exact_row_bytes, 0))
        return self._weight.index_select(0, rows)

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
