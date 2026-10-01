from __future__ import annotations

from awpmi.schedulers.base import Scheduler, SchedulingContext
from awpmi.state import ResidualState


class SequentialScheduler(Scheduler):
    """Lowest page_id first."""

    name = "sequential"

    def select(self, state: ResidualState, context: SchedulingContext) -> int:
        return min(state.remaining_pages)
