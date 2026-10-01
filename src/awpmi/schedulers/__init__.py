"""Deterministic page schedulers. They affect efficiency only, never correctness."""

from awpmi.schedulers.base import Scheduler, SchedulingContext
from awpmi.schedulers.bound import BoundReductionPerByte, LargestResidualFirst
from awpmi.schedulers.sequential import SequentialScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    SequentialScheduler.name: SequentialScheduler,
    LargestResidualFirst.name: LargestResidualFirst,
    BoundReductionPerByte.name: BoundReductionPerByte,
}


def make_scheduler(name: str) -> Scheduler:
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise ValueError(f"unknown scheduler {name!r}; choose from {sorted(SCHEDULERS)}") from None


__all__ = [
    "SCHEDULERS",
    "BoundReductionPerByte",
    "LargestResidualFirst",
    "Scheduler",
    "SchedulingContext",
    "SequentialScheduler",
    "make_scheduler",
]
