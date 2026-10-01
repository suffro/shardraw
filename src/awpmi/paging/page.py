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
    Nothing here says where the bytes live; that is the `PageSource`'s concern.
    """

    page_id: int
    parameter_name: str
    offset: int
    shape: tuple[int, int]
    dtype: torch.dtype
    storage_bytes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def column_slice(self) -> slice:
        return slice(self.offset, self.offset + self.width)
