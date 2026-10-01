from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics
from awpmi.certificate import Certificate, CertificateStatus
from awpmi.executor import AdaptiveLMHead
from awpmi.paging.index import build_column_page_index
from awpmi.paging.source import InMemoryPageSource
from awpmi.schedulers import SCHEDULERS, make_scheduler
from awpmi.state import ResidualState
from tests.conftest import DEVICES


def state(partial, lower, upper) -> ResidualState:
    as64 = lambda values: torch.tensor(values, dtype=torch.float64)  # noqa: E731
    return ResidualState(as64(partial), as64(lower), as64(upper), (), (0,))


def test_certifies_only_on_strict_separation():
    result = Certificate.check(state([5.0, 1.0, 2.0], [4.0, 0.0, 1.0], [6.0, 2.0, 3.0]))
    assert result.certified and result.status is CertificateStatus.CERTIFIED
    assert (result.winner, result.competitor) == (0, 2)
    assert result.winner_lower_bound == 4.0 and result.competitor_upper_bound == 3.0
    assert result.certificate_margin == 1.0


@pytest.mark.parametrize("competitor_upper", [4.0, 4.5])
def test_touching_or_overlapping_intervals_are_unknown(competitor_upper):
    result = Certificate.check(state([5.0, 1.0], [4.0, 0.0], [6.0, competitor_upper]))
    assert not result.certified and result.status is CertificateStatus.UNKNOWN
    assert result.certificate_margin <= 0.0


def test_single_candidate_is_trivially_certified():
    result = Certificate.check(state([1.0], [0.5], [1.5]))
    assert result.certified and result.competitor_upper_bound == -math.inf


def make_head(weight: torch.Tensor, width: int) -> AdaptiveLMHead:
    numerics = ReferenceNumerics(weight.dtype, FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1])
    return AdaptiveLMHead(build_column_page_index(weight, "w", width), InMemoryPageSource(weight), numerics)


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_rounding_tie_is_not_certified(device):
    """Exact logits favour row 1, but both round to 16.0 in BF16 and the reference picks row 0.

    A certificate on exact real values would certify the wrong token here.
    """
    weight = torch.tensor([[16.0, 0.0], [16.0, 0.03125]], dtype=torch.bfloat16, device=device)
    hidden = torch.tensor([[[1.0, 1.0]]], dtype=torch.bfloat16, device=device)
    reference = F.linear(hidden, weight).reshape(-1)
    assert reference.tolist() == [16.0, 16.0] and int(reference.argmax()) == 0
    exact = weight.double() @ hidden.reshape(-1).double()
    assert int(exact.argmax()) == 1

    for name in SCHEDULERS:
        result = make_head(weight, 1).run(hidden, make_scheduler(name))
        assert not result.certified and result.fallback
        assert result.token_id == 0
        assert torch.equal(result.fallback_logits.reshape(-1), reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_certified_token_always_matches_reference_on_random_systems(device, dtype):
    """Decision parity on many tiny systems, including near-ties built to stress the certificate."""
    generator = torch.Generator().manual_seed(5)
    certified_early = 0
    for trial in range(60):
        vocab, hidden_size, width = 64, 32, 4
        weight = torch.randn(vocab, hidden_size, generator=generator)
        hidden = torch.randn(hidden_size, generator=generator) * (1 + trial % 7)
        if trial % 3 == 0:
            # Near-duplicate rows: logits within ~one output ulp of each other.
            weight[1] = weight[0] + torch.randn(hidden_size, generator=generator) * 1e-3
        if trial % 3 == 1:
            # A dominant row, so that early certification is actually exercised.
            weight[7] = hidden / hidden.norm() * 4
        weight = weight.to(dtype).to(device)
        hidden = hidden.to(dtype).to(device).view(1, 1, -1)
        reference = F.linear(hidden, weight).reshape(-1)
        reference_token = int(torch.argmax(reference))
        head = make_head(weight, width)
        for name in SCHEDULERS:
            result = head.run(hidden, make_scheduler(name))
            assert result.token_id == reference_token, (trial, name)
            if result.fallback:
                assert torch.equal(result.fallback_logits.reshape(-1), reference)
            if result.certified and result.pages_materialized < result.pages_total:
                certified_early += 1
    assert certified_early > 0


@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_schedulers_are_deterministic_and_only_fetch_selected_pages(name):
    generator = torch.Generator().manual_seed(6)
    weight = torch.randn(128, 64, generator=generator).to(torch.bfloat16)
    hidden = torch.randn(1, 1, 64, generator=generator).to(torch.bfloat16)
    first, second = make_head(weight, 8), make_head(weight, 8)
    a = first.run(hidden, make_scheduler(name))
    b = second.run(hidden, make_scheduler(name))
    assert a.materialization_order == b.materialization_order
    assert a.trajectory == b.trajectory and a.token_id == b.token_id
    assert sorted(a.materialization_order) == sorted(set(a.materialization_order))
    assert first.source.fetch_count == a.pages_materialized
    assert first.source.bytes_fetched == a.bytes_materialized


def test_sequential_scheduler_uses_page_id_order():
    weight = torch.randn(32, 64).to(torch.bfloat16)
    hidden = torch.randn(1, 1, 64).to(torch.bfloat16)
    result = make_head(weight, 8).run(hidden, make_scheduler("sequential"))
    assert list(result.materialization_order) == list(range(result.pages_materialized))
