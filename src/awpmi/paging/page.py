from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class WeightPage:
    """One independently materializable block of a parameter.

    Phase 1 pages are column blocks of a [out_features, in_features] weight:
    `offset` is the first input column and `shape` is (out_features, width).
    Phase 2 pages (roadmap §2.7) also name their `layer` and `tensor_role` (q_proj …
    down_proj, lm_head); a neuron page of an MLP is one gate or up row or one down
    column, `offset` being the neuron. `metadata` carries the page's bound metadata.
    Nothing here says where the bytes live; that is the `PageSource`'s concern.
    """

    page_id: int
    parameter_name: str
    offset: int
    shape: tuple[int, int]
    dtype: torch.dtype
    storage_bytes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    layer: int | None = None
    tensor_role: str | None = None

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def column_slice(self) -> slice:
        return slice(self.offset, self.offset + self.width)
