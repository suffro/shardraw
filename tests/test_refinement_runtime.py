"""Phase 1C runtime: decision parity, enclosure, byte audit, consistency with the oracle, ties, guards."""

from __future__ import annotations

import math
import types

import pytest
import torch
import torch.nn.functional as F

import awpmi.refinement_head as refinement_head
from awpmi.bounds.coarse import CoarseArithmetic
from awpmi.bounds.linear import absolute_mass_upper
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics, remainder_radius
from awpmi.certificate import TieBreak, certify_columns, contenders
from awpmi.decomposition import RefinementDecomposition
from awpmi.oracle.refinement import BoundTier, RefinementBatch
from awpmi.refinement_head import FallbackMode, RefinementLMHead, RefinementRunResult
from awpmi.stores.refinement import PackedRefinementStore
from tests.conftest import DEVICES
from tests.test_certificate import tie_system
from tests.test_refinement_oracle import numerics_for, random_system, reference_logits

SPECS = ["q8", "q6", "q6+q4", "q4+q4", "q2+q2+q4"]


def system_with_a_winning_tie(device: str, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`random_system` plus one input whose two largest logits come from identical rows: it must fall back."""
    weight, hidden = random_system(device, dtype, seed)
    weight[21] = weight[20]
    tie_input = (weight[20].to(torch.float64) * 40).to(dtype)
    return weight, torch.cat([hidden, tie_input[None]])


def make_head(
    decomposition: RefinementDecomposition,
    fallback: FallbackMode = FallbackMode.FULL,
    tie_break: TieBreak = TieBreak.LOWEST_INDEX,
    chunk_rows: int = 64,
) -> RefinementLMHead:
    weight = decomposition.weight
    return RefinementLMHead(
        PackedRefinementStore.from_decomposition(decomposition),
        numerics_for(weight),
        CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1]),
        tie_break=tie_break,
        fallback=fallback,
        chunk_rows=chunk_rows,
    )


def check_run(result: RefinementRunResult, head: RefinementLMHead, decomposition, reference: torch.Tensor) -> None:
    """Every invariant of one run that can be checked against the full reference."""
    reference64 = reference.to(torch.float64)
    token = int(torch.argmax(reference))
    trace = result.trace
    assert result.token_id == token
    assert result.certified == (result.fallback is None)
    if result.certified:
        assert result.winner == token and not result.masked_guard_tripped
    elif result.fallback is FallbackMode.FULL:
        assert torch.equal(result.fallback_logits, reference)
    else:
        alive = trace.final_contenders
        assert torch.equal(result.fallback_logits[alive], reference[alive])

    # Every state's own interval encloses the reference logits of the rows it covers.
    for bounds in trace.states:
        covered = reference64 if bounds.rows is None else reference64[bounds.rows]
        assert bool(((covered >= bounds.lower) & (covered <= bounds.upper)).all()), bounds.state

    # Replay: each state advanced exactly the contenders of the previous one.
    lower, upper = trace.states[0].lower.clone(), trace.states[0].upper.clone()
    for bounds in trace.states[1:]:
        expected = contenders(lower, upper, head.tie_break).nonzero().squeeze(1)
        assert torch.equal(bounds.rows, expected)
        lower[bounds.rows] = torch.maximum(lower[bounds.rows], bounds.lower)
        upper[bounds.rows] = torch.minimum(upper[bounds.rows], bounds.upper)
    assert torch.equal(lower, trace.lower) and torch.equal(upper, trace.upper)
    final = certify_columns(lower, upper, head.tie_break)
    assert bool(final.certified) == result.certified
    if result.certified:
        assert result.decision_state == trace.states[-1].state

    # Bytes: what the store handed out is what the decomposition's accounting charges.
    fallback_rows = result.bytes["fallback_rows"] // decomposition.exact_row_bytes
    assert result.bytes == decomposition.materialized_bytes(list(result.rows_loaded), fallback_rows)
    for read in result.reads:
        if read.kind == "level" and read.state == 0:
            assert read.rows is None
        elif read.kind in ("level", "exact"):
            assert torch.equal(read.rows, trace.states[read.state].rows)
        else:
            exact_rows = trace.states[-1].rows if trace.states[-1].state == decomposition.exact_state else None
            unread = torch.ones(reference.numel(), dtype=torch.bool, device=reference.device)
            if exact_rows is not None:
                unread[exact_rows] = False
            assert torch.equal(read.rows, unread.nonzero().squeeze(1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("spec", SPECS)
def test_runtime_is_correct_on_random_systems(device, dtype, spec):
    early = fallbacks = 0
    for seed in range(2):
        weight, hidden = system_with_a_winning_tie(device, dtype, seed)
        reference = reference_logits(weight, hidden)
        decomposition = RefinementDecomposition.build(weight, spec)
        heads = {mode: make_head(decomposition, mode) for mode in FallbackMode}
        for b in range(hidden.shape[0]):
            results = {mode: head.run(hidden[b], trace=True) for mode, head in heads.items()}
            for mode, result in results.items():
                check_run(result, heads[mode], decomposition, reference[:, b])
            full, masked = results[FallbackMode.FULL], results[FallbackMode.MASKED]
            # The fallback mode changes nothing before the fallback.
            assert full.certified == masked.certified and full.contenders == masked.contenders
            assert full.rows_loaded == masked.rows_loaded and not masked.masked_guard_tripped
            if not full.certified:
                fallbacks += 1
                assert masked.bytes["fallback_rows"] == 0
            early += int(full.certified and full.decision_state < decomposition.exact_state)
    assert early > 0
    assert fallbacks > 0


@pytest.mark.parametrize("device", DEVICES)
def test_refined_states_reproduce_the_oracle_arithmetic(device):
    """From the first refinement on, the runtime computes exactly what the oracle's realistic tier computes."""
    weight, hidden = random_system(device, torch.bfloat16, seed=5)
    numerics = numerics_for(weight)
    for spec in ("q6+q4", "q2+q2+q4"):
        decomposition = RefinementDecomposition.build(weight, spec)
        head = make_head(decomposition)
        batch = RefinementBatch(decomposition, hidden, numerics)
        compared = 0
        for b in range(hidden.shape[0]):
            for bounds in head.run(hidden[b], trace=True).trace.states[1:]:
                rows = bounds.rows
                if bounds.state == decomposition.exact_state:
                    center = batch.exact_center[rows, b]
                else:
                    center = batch.centers[bounds.state][rows, b]
                    miss = batch.missing[BoundTier.REALISTIC][bounds.state][rows, b]
                    mass = batch.level_masses[bounds.state][rows, b] + miss
                    radius = remainder_radius(miss, mass, mass, numerics)
                    torch.testing.assert_close(bounds.radius, radius, rtol=1e-12, atol=0)
                torch.testing.assert_close(bounds.center, center, rtol=1e-12, atol=1e-12)
                compared += rows.numel()
        assert compared > 0


@pytest.mark.parametrize("device", DEVICES)
def test_coarse_centre_is_within_its_arithmetic_bound(device):
    """The binary32 coarse centre differs from the float64 one by less than the two error bounds together."""
    weight, hidden = random_system(device, torch.bfloat16, seed=6)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    head = make_head(decomposition)
    numerics = numerics_for(weight)
    base = decomposition.levels[0].values()
    for b in range(hidden.shape[0]):
        trace = head.run(hidden[b], trace=True).trace
        coarse = trace.states[0]
        exact_center = base @ hidden[b].to(torch.float64)
        float64_error = numerics.float64_gamma * absolute_mass_upper(base, hidden[b].to(torch.float64))
        assert bool(((coarse.center - exact_center).abs() <= trace.coarse_error + float64_error).all())


@pytest.mark.parametrize("device", DEVICES)
def test_lowest_index_tie_is_certified_and_strict_falls_back(device):
    weight, hidden = tie_system(device, larger_exact_row=0)
    reference = F.linear(hidden, weight).reshape(-1)
    decomposition = RefinementDecomposition.build(weight, "q8")
    tie_aware = make_head(decomposition).run(hidden, trace=True)
    check_run(tie_aware, make_head(decomposition), decomposition, reference)
    assert tie_aware.certified and tie_aware.token_id == 0 and tie_aware.decision_state == decomposition.exact_state
    for mode in FallbackMode:
        head = make_head(decomposition, mode, TieBreak.STRICT)
        strict = head.run(hidden, trace=True)
        check_run(strict, head, decomposition, reference)
        assert not strict.certified and strict.fallback is mode and strict.token_id == 0


@pytest.mark.parametrize("device", DEVICES)
def test_tie_lost_by_the_larger_exact_logit_falls_back_to_the_reference(device):
    weight, hidden = tie_system(device, larger_exact_row=1)
    reference = F.linear(hidden, weight).reshape(-1)
    decomposition = RefinementDecomposition.build(weight, "q8")
    for tie_break in TieBreak:
        for mode in FallbackMode:
            head = make_head(decomposition, mode, tie_break)
            result = head.run(hidden, trace=True)
            check_run(result, head, decomposition, reference)
            assert not result.certified and result.token_id == 0


def test_nonfinite_input_goes_straight_to_the_full_fallback():
    weight, hidden = random_system("cpu", torch.bfloat16, seed=0)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    vector = hidden[0].clone()
    vector[3] = math.inf
    result = make_head(decomposition, FallbackMode.MASKED).run(vector)
    assert not result.certified and result.fallback is FallbackMode.FULL
    assert result.fallback_reason == "nonfinite_input"
    assert result.rows_loaded == (0, 0, 0) and result.bytes["fallback_rows"] == decomposition.weight_bytes
    assert result.token_id == int(torch.argmax(F.linear(vector.view(1, 1, -1), weight)))


def test_masked_fallback_guard_falls_back_when_the_masked_gemm_is_wrong(monkeypatch):
    """Sabotage the masked GEMM: the per-call guard must catch it and run the full fallback."""
    weight, hidden = system_with_a_winning_tie("cpu", torch.bfloat16, seed=1)
    reference = reference_logits(weight, hidden)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    head = make_head(decomposition, FallbackMode.MASKED)  # self-test passes before the sabotage
    assert head.self_test == {"trials": 8, "mismatches": 0}

    def sabotaged_linear(inputs, matrix):
        logits = F.linear(inputs, matrix)
        has_zero_rows = bool((matrix == 0).all(dim=1).any())
        return logits + 1.0 if has_zero_rows else logits

    monkeypatch.setattr(refinement_head, "F", types.SimpleNamespace(linear=sabotaged_linear))
    tripped = 0
    for b in range(hidden.shape[0]):
        result = head.run(hidden[b], trace=True)
        assert result.token_id == int(torch.argmax(reference[:, b]))
        if not result.certified:
            assert result.masked_guard_tripped and result.fallback is FallbackMode.FULL
            assert torch.equal(result.fallback_logits, reference[:, b])
            tripped += 1
    assert tripped > 0


def test_masked_mode_refuses_a_platform_that_fails_the_self_test(monkeypatch):
    weight, _ = random_system("cpu", torch.bfloat16, seed=2)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")

    def row_dependent_linear(inputs, matrix):
        return F.linear(inputs, matrix) + float((matrix == 0).all(dim=1).any())

    monkeypatch.setattr(refinement_head, "F", types.SimpleNamespace(linear=row_dependent_linear))
    with pytest.raises(RuntimeError):
        make_head(decomposition, FallbackMode.MASKED)


def test_runtime_rejects_mismatched_inputs():
    weight, hidden = random_system("cpu", torch.bfloat16, seed=0)
    head = make_head(RefinementDecomposition.build(weight, "q6+q4"))
    with pytest.raises(TypeError):
        head.run(hidden[0].to(torch.float32))
    with pytest.raises(ValueError):
        head.run(hidden[:2])
    with pytest.raises(ValueError):
        RefinementLMHead(head.store, ReferenceNumerics(torch.bfloat16, 2.0**-22, 5), head.coarse)


@pytest.mark.model
def test_runtime_on_real_lm_head(model, prompt_ids, reference_numerics):
    from awpmi.models.smollm2 import final_hidden_state, lm_head_weight
    from awpmi.reference import ReferenceRunner

    weight = lm_head_weight(model)
    runner = ReferenceRunner(model)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    store = PackedRefinementStore.from_decomposition(decomposition)
    coarse = CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1])
    heads = {mode: RefinementLMHead(store, reference_numerics, coarse, fallback=mode) for mode in FallbackMode}
    base = decomposition.levels[0].values()
    for ids in prompt_ids:
        reference = runner.next_token(ids)
        hidden = final_hidden_state(model, ids).reshape(-1).contiguous()
        for head in heads.values():
            result = head.run(hidden, trace=True)
            check_run(result, head, decomposition, reference.logits)
            # A q6 base is decisive: most rows are eliminated after the coarse pass.
            assert result.rows_loaded[1] < weight.shape[0] // 2
        trace = heads[FallbackMode.FULL].run(hidden, trace=True).trace
        exact_center = base @ hidden.to(torch.float64)
        float64_error = reference_numerics.float64_gamma * absolute_mass_upper(base, hidden.to(torch.float64))
        assert bool(((trace.states[0].center - exact_center).abs() <= trace.coarse_error + float64_error).all())
