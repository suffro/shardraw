"""Typed weights and expert groups over a `MaterializationBackend`.

`WeightStore.rows(name, rows)` is a weight's rows as tensors of the weight's dtype and row
shape. `ExpertStore` serves mixture-of-experts layers: a group is one layer's expert-sliced
parameters (each a segment with one row per expert); `load` returns the requested experts'
slices and `fill` writes them into caller-owned full-shape buffers at their expert indices.
Group keys and parameter names are whatever the model adapter registers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage.layout import Segment, typed_rows


class WeightStore:
    def __init__(self, backend: MaterializationBackend) -> None:
        self.backend = backend

    def segment(self, name: str) -> Segment:
        return self.backend.segment(name)

    def rows(self, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Rows of weight `name` as [n, *row_shape] tensors of its dtype, on the compute device."""
        return typed_rows(self.backend.materialize(name, rows), self.segment(name))


@dataclass(frozen=True)
class ExpertGroup:
    """The expert-sliced parameters of one MoE layer: parameter name → segment (one row per expert)."""

    key: str
    experts: int
    segments: Mapping[str, str]


class ExpertStore:
    def __init__(self, weights: WeightStore, groups: Mapping[str, ExpertGroup]) -> None:
        for group in groups.values():
            for segment in group.segments.values():
                if weights.segment(segment).rows != group.experts:
                    raise ValueError(f"{group.key}: segment {segment} does not have one row per expert")
        self.weights = weights
        self.groups = dict(groups)

    def load(self, key: str, experts: torch.Tensor) -> dict[str, torch.Tensor]:
        """The slices of `experts` (ascending, unique) of every parameter of group `key`."""
        group = self.groups[key]
        return {name: self.weights.rows(segment, experts) for name, segment in group.segments.items()}

    def fill(self, key: str, experts: torch.Tensor, buffers: Mapping[str, torch.Tensor]) -> None:
        """Write the slices of `experts` into `buffers[name][experts]` (full-shape device tensors)."""
        if experts.numel() == 0:
            return
        index = experts.to(next(iter(buffers.values())).device)
        for name, data in self.load(key, experts).items():
            buffers[name].index_copy_(0, index, data)
