"""Deterministic, conservative certificate for the next-token argmax.

The answer is CERTIFIED or UNKNOWN. There is no "likely": a certificate either
proves the reference decision or says nothing.
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
    def check(state: ResidualState) -> CertificateResult:
        """Certify w = argmax(partial_logits) iff lower[w] > max_{j≠w} upper[j].

        The bounds enclose the reference logits as the reference produces them
        (after output rounding), so strict separation implies the reference argmax
        is w with no tie, whatever the reference's tie-breaking rule.
        """
        winner = int(torch.argmax(state.partial_logits).item())
        winner_lower = float(state.lower_bounds[winner].item())
        competitors = state.upper_bounds.clone()
        competitors[winner] = -math.inf
        competitor = int(torch.argmax(competitors).item())
        competitor_upper = float(competitors[competitor].item())
        margin = winner_lower - competitor_upper
        return CertificateResult(
            certified=winner_lower > competitor_upper,
            winner=winner,
            certificate_margin=margin,
            winner_lower_bound=winner_lower,
            competitor_upper_bound=competitor_upper,
            competitor=competitor,
        )
