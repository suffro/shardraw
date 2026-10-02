"""Precision-refinement decomposition of a weight matrix.

    W = L_0 + L_1 + … + L_{n-1} + X_{n-1}         exactly, in real arithmetic

L_0 is the base (coarse) representation, L_1… are refinement levels, and
X_s = W − Σ_{l≤s} L_l is the remainder after state s. A row moves through states:

    state s < n   levels 0..s materialized; X_s is missing and bounded by metadata
    state n       exact: the original row of W (whose stored bytes are the
                  checkpoint's own), so the reference operation can be recomputed

The final refinement is therefore the original BF16 row, not a separately encoded
remainder. Its bytes are charged in full whenever a row reaches the exact state.

This module only builds levels, checks exactness and accounts for bytes. Bounds
on missing remainders live in `awpmi.bounds.remainder`; certification and
simulation live elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from awpmi.bounds.remainder import InputNorms, RowNormBounds, remainder_mass_bound, row_norm_bounds
from awpmi.decomposition.packing import payload_bytes
from awpmi.decomposition.quantization import SCALE_DTYPE, code_limit, dequantize_rows, quantize_rows


@dataclass(frozen=True)
class RefinementLevel:
    """One stored level: integer codes and a float32 scale per row (bits = 0: the zero matrix)."""

    bits: int
    in_features: int
    codes: torch.Tensor | None
    scales: torch.Tensor | None

    @property
    def name(self) -> str:
        return f"q{self.bits}" if self.bits else "zero"

    @property
    def payload_bytes_per_row(self) -> int:
        """Bit-packed codes (`awpmi.decomposition.packing`)."""
        return payload_bytes(self.in_features, self.bits)

    @property
    def scale_bytes_per_row(self) -> int:
        return torch.finfo(SCALE_DTYPE).bits // 8 if self.bits else 0

    def values(self) -> torch.Tensor:
        """The level as an exact float64 matrix."""
        if self.codes is None or self.scales is None:
            raise ValueError("the zero level has no stored values")
        return dequantize_rows(self.codes, self.scales)


def subtraction_error(minuend: torch.Tensor, subtrahend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fl(a − b) and its exact rounding error (Knuth's TwoSum); the error is zero iff the subtraction is exact."""
    negated = -subtrahend
    total = minuend + negated
    virtual = total - minuend
    error = (minuend - (total - virtual)) + (negated - virtual)
    return total, error


def parse_spec(spec: str) -> tuple[int, ...]:
    """'none' → a zero base level (row-norm metadata only); 'q6+q4' → int6 base, int4 refinement."""
    if spec == "none":
        return (0,)
    levels = []
    for part in spec.split("+"):
        if not part.startswith("q") or not part[1:].isdigit():
            raise ValueError(f"bad decomposition spec {spec!r}")
        bits = int(part[1:])
        code_limit(bits)
        levels.append(bits)
    return tuple(levels)


@dataclass(frozen=True)
class RefinementDecomposition:
    spec: str
    weight: torch.Tensor
    levels: tuple[RefinementLevel, ...]
    remainder_norms: tuple[RowNormBounds, ...]

    @classmethod
    def build(cls, weight: torch.Tensor, spec: str) -> RefinementDecomposition:
        """Quantize the remainder level by level and verify that every subtraction is exact."""
        if weight.dim() != 2:
            raise ValueError("expected a [out_features, in_features] weight")
        in_features = weight.shape[1]
        remainder = weight.to(torch.float64)
        levels: list[RefinementLevel] = []
        norms: list[RowNormBounds] = []
        for bits in parse_spec(spec):
            if bits == 0:
                level = RefinementLevel(0, in_features, None, None)
            else:
                codes, scales = quantize_rows(remainder, bits)
                level = RefinementLevel(bits, in_features, codes, scales)
                remainder, error = subtraction_error(remainder, level.values())
                if bool((error != 0).any()):
                    raise ArithmeticError(f"{spec}: remainder after {level.name} is not exact in float64")
            levels.append(level)
            norms.append(row_norm_bounds(remainder))
        return cls(spec, weight, tuple(levels), tuple(norms))

    # Structure

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def exact_state(self) -> int:
        return len(self.levels)

    @property
    def num_states(self) -> int:
        return len(self.levels) + 1

    def state_name(self, state: int) -> str:
        return "exact" if state == self.exact_state else self.levels[state].name

    def base_representation(self) -> RefinementLevel:
        return self.levels[0]

    def refinement_levels(self) -> tuple[RefinementLevel, ...]:
        """Stored levels after the base; the exact state follows them."""
        return self.levels[1:]

    def remainder(self, state: int) -> torch.Tensor:
        """X_state = W − Σ_{l≤state} L_l in float64, by the same subtractions `build` verified exact."""
        if not 0 <= state < self.exact_state:
            raise ValueError(f"state {state} has no remainder")
        remainder = self.weight.to(torch.float64)
        for level in self.levels[: state + 1]:
            if level.bits:
                remainder = remainder - level.values()
        return remainder

    # Bounds

    def residual_bound(self, state: int, inputs: InputNorms) -> torch.Tensor:
        """Realistic bound on Σ_k |X_state[j,k]|·|h_k| ≥ |X_state[j]·h| for every row (resident metadata only)."""
        return remainder_mass_bound(self.remainder_norms[state], inputs)

    # Bytes

    @property
    def weight_bytes(self) -> int:
        """The original matrix, i.e. the reference LM-head bytes."""
        return self.weight.numel() * self.weight.element_size()

    @property
    def exact_row_bytes(self) -> int:
        return self.in_features * self.weight.element_size()

    def row_bytes(self, state: int) -> tuple[int, int]:
        """(payload, scale) bytes that move one row into `state`."""
        if state == self.exact_state:
            return self.exact_row_bytes, 0
        level = self.levels[state]
        return level.payload_bytes_per_row, level.scale_bytes_per_row

    def metadata_bytes(self) -> int:
        """Resident bound metadata: per-row remainder norms for every non-exact state."""
        return sum(norms.nbytes for norms in self.remainder_norms)

    def materialized_bytes(self, rows_per_state: Sequence[int], fallback_rows: int = 0) -> dict[str, int]:
        """Bytes read when `rows_per_state[s]` rows enter state s, plus `fallback_rows` exact rows for a fallback.

        The resident metadata is always charged.
        """
        if len(rows_per_state) != self.num_states:
            raise ValueError("one row count per state is required")
        breakdown = {
            "metadata": self.metadata_bytes(),
            "base_payload": 0,
            "base_scales": 0,
            "refinement_payload": 0,
            "refinement_scales": 0,
            "exact_rows": 0,
            "fallback_rows": fallback_rows * self.exact_row_bytes,
        }
        for state, rows in enumerate(rows_per_state):
            payload, scales = self.row_bytes(state)
            if state == self.exact_state:
                breakdown["exact_rows"] += rows * payload
            elif state == 0:
                breakdown["base_payload"] += rows * payload
                breakdown["base_scales"] += rows * scales
            else:
                breakdown["refinement_payload"] += rows * payload
                breakdown["refinement_scales"] += rows * scales
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def storage_bytes(self) -> dict[str, int]:
        """Everything stored: every level for every row, the original matrix, the metadata."""
        rows = self.out_features
        levels = sum(rows * sum(self.row_bytes(state)) for state in range(self.exact_state))
        storage = {"levels": levels, "original": self.weight_bytes, "metadata": self.metadata_bytes()}
        storage["total"] = sum(storage.values())
        return storage

    def describe(self) -> dict:
        return {
            "spec": self.spec,
            "states": [self.state_name(s) for s in range(self.num_states)],
            "row_bytes": {self.state_name(s): list(self.row_bytes(s)) for s in range(self.num_states)},
            "metadata_bytes": self.metadata_bytes(),
            "storage_bytes": self.storage_bytes(),
            "weight_bytes": self.weight_bytes,
        }
