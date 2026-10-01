"""Full materialization must reproduce the reference; certified and fallback tokens must match it."""

from __future__ import annotations

import pytest
import torch

from awpmi.executor import AdaptiveLMHead
from awpmi.models.smollm2 import LM_HEAD_PARAMETER, final_hidden_state, lm_head_weight
from awpmi.paging.index import build_column_page_index
from awpmi.paging.source import InMemoryPageSource
from awpmi.reference import ReferenceRunner
from awpmi.schedulers import SCHEDULERS, make_scheduler

pytestmark = pytest.mark.model

WIDTHS = (16, 32, 64)


@pytest.fixture(scope="module")
def heads(model, reference_numerics):
    weight = lm_head_weight(model)
    return {
        width: AdaptiveLMHead(
            build_column_page_index(weight, LM_HEAD_PARAMETER, width), InMemoryPageSource(weight), reference_numerics
        )
        for width in WIDTHS
    }


@pytest.fixture(scope="module")
def cases(model, prompt_ids):
    runner = ReferenceRunner(model)
    return [(runner.next_token(ids), final_hidden_state(model, ids)) for ids in prompt_ids]


@pytest.mark.parametrize("width", WIDTHS)
def test_full_materialization_reproduces_reference_bitwise(heads, cases, width):
    for reference, hidden in cases:
        logits = heads[width].full_materialization_logits(hidden).reshape(-1)
        assert torch.equal(logits, reference.logits)
        assert int(torch.argmax(logits)) == reference.token_id


@pytest.mark.parametrize("width", WIDTHS)
def test_page_accumulated_logits_lie_in_reference_envelope(heads, cases, width):
    """AWPMI logits ≈ reference logits, with the tolerance derived from the accumulation model."""
    for reference, hidden in cases:
        partial, bounder = heads[width].full_partial_logits(hidden)
        no_pages_left = torch.zeros(len(heads[width].index), dtype=torch.bool, device=partial.device)
        lower, upper = bounder.logit_bounds(partial, no_pages_left)
        ref64 = reference.logits.to(torch.float64)
        assert bool((ref64 >= lower).all()) and bool((ref64 <= upper).all())
        # Loose sanity check on top: within one BF16 ulp at the logit scale.
        assert float((partial - ref64).abs().max()) <= 0.125


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_decision_parity_on_real_prompts(heads, cases, width, name):
    head = heads[width]
    for reference, hidden in cases:
        before = head.source.fetch_count
        result = head.run(hidden, make_scheduler(name))
        assert result.token_id == reference.token_id
        assert result.certified != result.fallback
        if result.fallback:
            assert result.pages_materialized == result.pages_total
            assert torch.equal(result.fallback_logits.reshape(-1), reference.logits)
        assert head.source.fetch_count - before == result.pages_materialized
