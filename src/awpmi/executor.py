"""Adaptive LM head: progressive page materialization with certificate and exact fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from awpmi.bounds.linear import block_l2_norm_upper, page_contribution
from awpmi.bounds.residual import ReferenceNumerics, ResidualBounder
from awpmi.certificate import Certificate, CertificateResult
from awpmi.paging.index import PageIndex
from awpmi.paging.source import PageSource
from awpmi.schedulers.base import Scheduler, SchedulingContext
from awpmi.state import ResidualState, initial_state, refine_state


@dataclass(frozen=True)
class StepTrace:
    pages_materialized: int
    winner: int
    certificate_margin: float


@dataclass(frozen=True)
class AWPMIResult:
    token_id: int
    certified: bool
    fallback: bool
    certificate: CertificateResult
    pages_total: int
    materialization_order: tuple[int, ...]
    bytes_materialized: int
    bytes_total: int
    trajectory: tuple[StepTrace, ...]
    fallback_logits: torch.Tensor | None

    @property
    def pages_materialized(self) -> int:
        return len(self.materialization_order)

    @property
    def materialized_fraction(self) -> float:
        return self.pages_materialized / self.pages_total


class AdaptiveLMHead:
    """Computes the next token from the exact final hidden state using as few LM-head pages as the certificate allows.

    Loop: check certificate → if CERTIFIED stop; if pages remain, the scheduler
    picks one, it is materialized and its exact contribution added; once every
    page is materialized without a certificate, the reference operation is
    recomputed from the materialized pages (fallback).
    """

    def __init__(self, index: PageIndex, source: PageSource, numerics: ReferenceNumerics) -> None:
        if numerics.reduction_length != index.parameter_shape[1]:
            raise ValueError("numerics.reduction_length must equal the paged input dimension")
        self.index = index
        self.source = source
        self.numerics = numerics
        self._row_page_norms = index.bounds.row_page_norms.to(torch.float64)

    def _bounder(self, hidden_vector: torch.Tensor) -> ResidualBounder:
        hidden_page_norms = block_l2_norm_upper(hidden_vector, self.index.column_slices)
        return ResidualBounder(self._row_page_norms, hidden_page_norms, self.numerics)

    def _hidden_vector(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dtype != self.index.dtype:
            raise TypeError(f"hidden dtype {hidden.dtype} does not match weight dtype {self.index.dtype}")
        if hidden.numel() != self.index.parameter_shape[1]:
            raise ValueError("hidden state must be a single position")
        return hidden.reshape(-1).to(torch.float64)

    def run(self, hidden: torch.Tensor, scheduler: Scheduler) -> AWPMIResult:
        """`hidden` is the exact LM-head input for one position, shaped as the reference passes it."""
        hidden_vector = self._hidden_vector(hidden)
        bounder = self._bounder(hidden_vector)
        context = SchedulingContext(self.index.pages, self._row_page_norms, bounder.hidden_page_norms)
        state = initial_state(bounder, self.index.parameter_shape[0])
        materialized: dict[int, torch.Tensor] = {}
        trajectory: list[StepTrace] = []

        while True:
            result = Certificate.check(state)
            trajectory.append(StepTrace(len(state.materialized_pages), result.winner, result.certificate_margin))
            if result.certified:
                return self._result(state, result, materialized, trajectory, fallback_logits=None)
            if state.is_complete:
                logits = self._reference_operation(hidden, materialized)
                return self._result(state, result, materialized, trajectory, fallback_logits=logits)
            page_id = scheduler.select(state, context)
            page = self.index.page(page_id)
            values = self.source.get(page)
            materialized[page_id] = values
            contribution = page_contribution(values, hidden_vector[page.column_slice])
            state = refine_state(state, bounder, page_id, contribution)

    def _reference_operation(self, hidden: torch.Tensor, materialized: dict[int, torch.Tensor]) -> torch.Tensor:
        """Recompute nn.Linear(bias=False) exactly as the reference does, from materialized pages."""
        weight = torch.cat([materialized[page.page_id] for page in self.index.pages], dim=1)
        return F.linear(hidden, weight)

    def _result(
        self,
        state: ResidualState,
        certificate: CertificateResult,
        materialized: dict[int, torch.Tensor],
        trajectory: list[StepTrace],
        fallback_logits: torch.Tensor | None,
    ) -> AWPMIResult:
        if fallback_logits is None:
            token_id = certificate.winner
        else:
            token_id = int(torch.argmax(fallback_logits.reshape(-1)).item())
        return AWPMIResult(
            token_id=token_id,
            certified=certificate.certified,
            fallback=fallback_logits is not None,
            certificate=certificate,
            pages_total=len(self.index),
            materialization_order=state.materialized_pages,
            bytes_materialized=sum(self.index.page(p).storage_bytes for p in materialized),
            bytes_total=self.index.total_bytes,
            trajectory=tuple(trajectory),
            fallback_logits=fallback_logits,
        )

    # Validation helpers: compute with every page, outside the adaptive loop.

    def full_materialization_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The fallback operation applied to all pages."""
        materialized = {page.page_id: self.source.get(page) for page in self.index.pages}
        return self._reference_operation(hidden, materialized)

    def full_partial_logits(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ResidualBounder]:
        """Page-accumulated float64 logits with every page materialized, and the bounder used."""
        hidden_vector = self._hidden_vector(hidden)
        partial = torch.zeros(self.index.parameter_shape[0], dtype=torch.float64, device=hidden.device)
        for page in self.index.pages:
            partial = partial + page_contribution(self.source.get(page), hidden_vector[page.column_slice])
        return partial, self._bounder(hidden_vector)
