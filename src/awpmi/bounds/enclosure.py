"""Bounds on the reference's values of an internal tensor (Phase 2, roadmap §2.2).

An `Enclosure` generalizes the LM head's residual state to any point of the model:
elementwise, lower ≤ (the reference's value) ≤ upper, in float64. `provenance` names the
sources of uncertainty that reached the tensor: missing pages of a parameter
("missing:<parameter>") and the reference's own roundings ("rounding:<tensor>"). It holds
no pages and knows nothing of storage or streaming.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.floating import next_up


class UnboundedValue(ArithmeticError):
    """A bound cannot be propagated soundly (non-finite or out-of-range values): the caller must fall back."""


@dataclass(frozen=True)
class Enclosure:
    lower: torch.Tensor
    upper: torch.Tensor
    provenance: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.lower.dtype != torch.float64 or self.upper.dtype != torch.float64:
            raise TypeError("enclosures are float64")
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper bounds disagree on shape")
        if bool(torch.isnan(self.lower).any() or torch.isnan(self.upper).any()):
            raise ValueError("NaN bound")
        if bool((self.lower > self.upper).any()):
            raise ValueError("empty enclosure: lower > upper")

    @classmethod
    def exact(cls, values: torch.Tensor, provenance: frozenset[str] = frozenset()) -> Enclosure:
        """A point enclosure: the reference's values are known exactly."""
        values = values.to(torch.float64)
        return cls(values, values.clone(), provenance)

    @property
    def is_exact(self) -> bool:
        return bool(torch.equal(self.lower, self.upper))

    @property
    def finite(self) -> bool:
        return bool(torch.isfinite(self.lower).all() and torch.isfinite(self.upper).all())

    @property
    def center(self) -> torch.Tensor:
        """A point inside the enclosure (halving is exact for normal values; one rounding remains)."""
        return self.lower * 0.5 + self.upper * 0.5

    def radius_about(self, center: torch.Tensor) -> torch.Tensor:
        """r ≥ 0 with [lower, upper] ⊆ center ± r, rounded up."""
        return next_up(torch.maximum(self.upper - center, center - self.lower)).clamp_min(0.0)

    @property
    def magnitude(self) -> torch.Tensor:
        """max |value| over the enclosure."""
        return torch.maximum(self.lower.abs(), self.upper.abs())

    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower

    def contains(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.float64)
        return (values >= self.lower) & (values <= self.upper)

    def violations(self, values: torch.Tensor) -> int:
        return int((~self.contains(values)).sum())

    def select(self, index: torch.Tensor) -> Enclosure:
        return Enclosure(self.lower[index], self.upper[index], self.provenance)

    def tagged(self, *labels: str) -> Enclosure:
        return Enclosure(self.lower, self.upper, self.provenance | frozenset(labels))
