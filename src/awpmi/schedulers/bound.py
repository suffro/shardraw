from __future__ import annotations

import torch

from awpmi.schedulers.base import Scheduler, SchedulingContext, highest_score_lowest_id
from awpmi.state import ResidualState


class LargestResidualFirst(Scheduler):
    """score(p) = ‖h_p‖₂ × max_j ‖W[j, p]‖₂ — the largest possible residual of the page."""

    name = "largest_residual"

    def select(self, state: ResidualState, context: SchedulingContext) -> int:
        remaining = list(state.remaining_pages)
        max_row_norms = context.row_page_norms[:, remaining].max(dim=0).values
        scores = (context.hidden_page_norms[remaining] * max_row_norms).tolist()
        return highest_score_lowest_id(remaining, scores)


class BoundReductionPerByte(Scheduler):
    """score(p) = estimated certificate-gap reduction / storage_bytes.

    The certificate gap is max_{j≠w} upper[j] − lower[w]. Materializing p shrinks
    the radius of row j by N[j,p]·‖h_p‖₂, so the estimate is
    ‖h_p‖₂ · (N[w,p] + max_{j ∈ C} N[j,p]) where C are the competitors whose upper
    bound still reaches lower[w].
    """

    name = "bound_per_byte"

    def select(self, state: ResidualState, context: SchedulingContext) -> int:
        remaining = list(state.remaining_pages)
        winner = int(torch.argmax(state.partial_logits).item())
        contenders = state.upper_bounds >= state.lower_bounds[winner]
        contenders[winner] = False
        norms = context.row_page_norms[:, remaining]
        winner_term = norms[winner]
        if bool(contenders.any()):
            competitor_term = norms[contenders].max(dim=0).values
        else:
            competitor_term = torch.zeros_like(winner_term)
        reduction = context.hidden_page_norms[remaining] * (winner_term + competitor_term)
        storage = torch.tensor(
            [context.pages[p].storage_bytes for p in remaining], dtype=torch.float64, device=reduction.device
        )
        return highest_score_lowest_id(remaining, (reduction / storage).tolist())
