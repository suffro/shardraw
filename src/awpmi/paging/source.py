from __future__ import annotations

from typing import Protocol

import torch

from awpmi.paging.page import WeightPage


class PageSource(Protocol):
    """Materializes the values of a page. Consumers must treat the result as read-only."""

    def get(self, page: WeightPage) -> torch.Tensor: ...


class InMemoryPageSource:
    """Phase 1 source: pages are views of a resident tensor.

    Materialization is logical only, but every fetch is counted so callers can
    audit that only selected pages were requested.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        self._weight = weight
        self.fetch_count = 0
        self.bytes_fetched = 0

    def get(self, page: WeightPage) -> torch.Tensor:
        self.fetch_count += 1
        self.bytes_fetched += page.storage_bytes
        return self._weight[:, page.column_slice]
