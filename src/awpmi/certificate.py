"""Deterministic, conservative certificate for the next-token argmax.

The answer is CERTIFIED or UNKNOWN. There is no "likely": a certificate either
proves the reference decision or says nothing.

Tie handling (`TieBreak`):

* `STRICT` — lower[w] > upper[j] for every j ≠ w. Implies a unique reference
  maximum, so it holds whatever the reference's tie-breaking rule is.
* `LOWEST_INDEX` — the reference decision is `torch.argmax`, which returns the
  lowest index among equal maxima (asserted by `tests/test_certificate.py`). Then
  w is the reference token iff ℓ_w > ℓ_j for every j < w and ℓ_w ≥ ℓ_j for every
  j > w, so it suffices that lower[w] > upper[j] for j < w and lower[w] ≥ upper[j]
  for j > w. The bounds enclose the reference logits on the output grid, so an
  equality of bounds can only come from an actual tie that w wins.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch

from awpmi.state import ResidualState


class CertificateStatus(enum.Enum):
    CERTIFIED = "CERTIFIED"
    UNKNOWN = "UNKNOWN"


class TieBreak(enum.Enum):
    STRICT = "strict"
    LOWEST_INDEX = "lowest_index"


@dataclass(frozen=True)
class CertificateResult:
    certified: bool
    winner: int
    certificate_margin: float
    winner_lower_bound: float
    competitor_upper_bound: float
    competitor: int

    @property
    def status(self) -> CertificateStatus:
        return CertificateStatus.CERTIFIED if self.certified else CertificateStatus.UNKNOWN


class Certificate:
    @staticmethod
    def check(state: ResidualState, tie_break: TieBreak = TieBreak.STRICT) -> CertificateResult:
        """Certify w = argmax(partial_logits) from the state's bounds on the reference logits."""
        winner = int(torch.argmax(state.partial_logits).item())
        return Certificate.check_winner(state.lower_bounds, state.upper_bounds, winner, tie_break)

    @staticmethod
    def check_winner(
        lower: torch.Tensor, upper: torch.Tensor, winner: int, tie_break: TieBreak = TieBreak.STRICT
    ) -> CertificateResult:
        """Certify a given candidate `winner` against every other row (see the module docstring)."""
        winner_lower = float(lower[winner].item())
        competitors = upper.clone()
        competitors[winner] = -math.inf
        competitor = int(torch.argmax(competitors).item())
        competitor_upper = float(competitors[competitor].item())
        margin = winner_lower - competitor_upper
        if tie_break is TieBreak.STRICT:
            certified = winner_lower > competitor_upper
        else:
            before = float(competitors[:winner].max().item()) if winner > 0 else -math.inf
            after = float(competitors[winner + 1 :].max().item()) if winner + 1 < len(competitors) else -math.inf
            certified = winner_lower > -math.inf and winner_lower > before and winner_lower >= after
        return CertificateResult(
            certified=certified,
            winner=winner,
            certificate_margin=margin,
            winner_lower_bound=winner_lower,
            competitor_upper_bound=competitor_upper,
            competitor=competitor,
        )


def _row_index(lower: torch.Tensor) -> torch.Tensor:
    index = torch.arange(lower.shape[0], device=lower.device)
    return index if lower.dim() == 1 else index[:, None]


def first_argmax(values: torch.Tensor) -> torch.Tensor:
    """Lowest row index attaining the maximum along dim 0 (deterministic on every backend)."""
    best = values.max(dim=0, keepdim=True).values
    return torch.where(values == best, _row_index(values), values.shape[0]).min(dim=0).values


@dataclass(frozen=True)
class ColumnCertificates:
    """`certify_columns` result: one entry per input column."""

    certified: torch.Tensor
    winner: torch.Tensor
    competitor: torch.Tensor
    margin: torch.Tensor


def certify_columns(lower: torch.Tensor, upper: torch.Tensor, tie_break: TieBreak) -> ColumnCertificates:
    """`Certificate.check_winner` for every column of [V, B] bounds, with w = first argmax of `lower`.

    Any choice of w is sound; the row with the largest lower bound is the only one
    that can certify, so this choice certifies whenever any choice would.
    """
    winner = first_argmax(lower)
    rows = _row_index(lower)
    winner_lower = lower.gather(0, winner[None]).squeeze(0)
    competitors = upper.masked_fill(rows == winner, -math.inf)
    competitor_upper, competitor = competitors.max(dim=0)
    if tie_break is TieBreak.STRICT:
        certified = winner_lower > competitor_upper
    else:
        before = competitors.masked_fill(rows > winner, -math.inf).max(dim=0).values
        after = competitors.masked_fill(rows < winner, -math.inf).max(dim=0).values
        certified = (winner_lower > -math.inf) & (winner_lower > before) & (winner_lower >= after)
    return ColumnCertificates(certified, winner, competitor, winner_lower - competitor_upper)


def contenders(lower: torch.Tensor, upper: torch.Tensor, tie_break: TieBreak) -> torch.Tensor:
    """Rows that may still be the reference argmax, for [V] or [V, B] bounds.

    Row j is eliminated when some row i is provably at least as large and wins the
    comparison: upper[j] < lower[i], or (LOWEST_INDEX only) upper[j] = lower[i] with
    i < j. Checking against the largest lower bound, taking its lowest index, finds
    every such row.
    """
    best = lower.max(dim=0, keepdim=True).values
    alive = upper >= best
    if tie_break is TieBreak.LOWEST_INDEX:
        holder = first_argmax(lower)
        alive &= ~((upper == best) & (_row_index(lower) > holder))
    return alive
