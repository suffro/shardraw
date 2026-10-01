from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from awpmi.paging.page import WeightPage
from awpmi.state import ResidualState


@dataclass(frozen=True)
class SchedulingContext:
    """What a scheduler may look at: page descriptors and bound metadata, never page values."""

    pages: tuple[WeightPage, ...]
    row_page_norms: torch.Tensor
    hidden_page_norms: torch.Tensor


class Scheduler(ABC):
    name: str

    @abstractmethod
    def select(self, state: ResidualState, context: SchedulingContext) -> int:
        """Return the id of the next remaining page to materialize."""


def highest_score_lowest_id(page_ids: list[int], scores: list[float]) -> int:
    """Deterministic selection: highest score, ties broken by lowest page_id."""
    return min(zip(page_ids, scores), key=lambda item: (-item[1], item[0]))[0]
