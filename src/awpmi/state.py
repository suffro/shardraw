"""ResidualState: known contribution plus conservative uncertainty from missing pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.residual import ResidualBounder


@dataclass(frozen=True)
class ResidualState:
    """Snapshot of progressive refinement for one input.

    `partial_logits` is Σ over materialized pages of W_p h_p (float64).
    `lower_bounds`/`upper_bounds` bound every *reference* logit (see
    `awpmi.bounds.residual`); `materialized_pages` is in materialization order.
    """

    partial_logits: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    materialized_pages: tuple[int, ...]
    remaining_pages: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        return not self.remaining_pages


def _remaining_mask(remaining_pages: tuple[int, ...], page_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(page_count, dtype=torch.bool, device=device)
    if remaining_pages:
        mask[list(remaining_pages)] = True
    return mask


def initial_state(bounder: ResidualBounder, vocab_size: int) -> ResidualState:
    page_count = bounder.hidden_page_norms.shape[0]
    device = bounder.hidden_page_norms.device
    remaining = tuple(range(page_count))
    partial = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    lower, upper = bounder.logit_bounds(partial, _remaining_mask(remaining, page_count, device))
    return ResidualState(partial, lower, upper, (), remaining)


def refine_state(
    state: ResidualState, bounder: ResidualBounder, page_id: int, contribution: torch.Tensor
) -> ResidualState:
    """Add the exact contribution of one newly materialized page and tighten the bounds."""
    if page_id not in state.remaining_pages:
        raise ValueError(f"page {page_id} is not remaining")
    page_count = bounder.hidden_page_norms.shape[0]
    remaining = tuple(p for p in state.remaining_pages if p != page_id)
    partial = state.partial_logits + contribution
    lower, upper = bounder.logit_bounds(partial, _remaining_mask(remaining, page_count, partial.device))
    return ResidualState(partial, lower, upper, state.materialized_pages + (page_id,), remaining)
