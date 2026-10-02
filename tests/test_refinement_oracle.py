"""Refinement oracle: enclosure of the real reference, decision parity, mode equivalence, and a slow cross-check."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics
from awpmi.certificate import Certificate, TieBreak, contenders
from awpmi.decomposition import RefinementDecomposition
from awpmi.oracle.refinement import BoundTier, Mode, RefinementBatch, blocks_touched, simulate
from tests.conftest import DEVICES

SPECS = ["none", "q8", "q6", "q4", "q4+q4", "q2+q2+q4"]


def random_system(device: str, dtype: torch.dtype, seed: int, vocab: int = 300, width: int = 48, inputs: int = 24):
    """Rows with near-duplicates, exact duplicates and dominant rows, so that every outcome occurs."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab, width, generator=generator) * 0.1
    hidden = torch.randn(inputs, width, generator=generator) * 3
    weight[1] = weight[0] + torch.randn(width, generator=generator) * 1e-3
    weight[2] = weight[0]  # an exact duplicate: a guaranteed tie
    for b in range(0, inputs, 3):
        weight[10 + b] = hidden[b] / hidden[b].norm() * (0.2 + 0.1 * b)
    return weight.to(dtype).to(device), hidden.to(dtype).to(device)


def reference_logits(weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """The reference operation, one input at a time in the reference's own shape: [V, B]."""
    return torch.stack([F.linear(row.view(1, 1, -1), weight).reshape(-1) for row in hidden], dim=1)


def numerics_for(weight: torch.Tensor) -> ReferenceNumerics:
    return ReferenceNumerics(weight.dtype, FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_oracle_encloses_reference_and_never_certifies_a_wrong_token(device, dtype):
    for seed in range(3):
        weight, hidden = random_system(device, dtype, seed)
        reference = reference_logits(weight, hidden)
        tokens = reference.argmax(dim=0)
        early = 0
        for spec in SPECS:
            decomposition = RefinementDecomposition.build(weight, spec)
            row_bytes = [sum(decomposition.row_bytes(s)) for s in range(decomposition.num_states)]
            batch = RefinementBatch(decomposition, hidden, numerics_for(weight))
            for tier in BoundTier:
                intervals = batch.intervals(tier)
                assert int(intervals.violations(reference).sum()) == 0, (seed, spec, tier)
                for tie_break in TieBreak:
                    runs = {mode: simulate(intervals, mode, tie_break, row_bytes) for mode in Mode}
                    for run in runs.values():
                        assert torch.equal(run.winner[run.certified], tokens[run.certified])
                        fallback = ~run.certified
                        assert bool(run.final_contenders[tokens[fallback], fallback.nonzero().squeeze(1)].all())
                    selective, everything = runs[Mode.ROW_SELECTIVE], runs[Mode.GLOBAL]
                    # Eliminated rows never block a certificate: same decisions, fewer rows read.
                    assert torch.equal(selective.certified, everything.certified)
                    assert torch.equal(selective.decision_state, everything.decision_state)
                    assert torch.equal(selective.contenders, everything.contenders)
                    assert bool((selective.rows_loaded <= everything.rows_loaded).all())
                    exact = decomposition.exact_state
                    early += int((selective.certified & (selective.decision_state < exact)).sum())
        assert early > 0


def slow_simulation(lower, upper, mode: Mode, tie_break: TieBreak):
    """One input, written as directly as possible: the vectorized simulator must agree."""
    states, rows = lower.shape
    current_lower, current_upper = lower[0].clone(), upper[0].clone()
    alive = [True] * rows
    loaded = [rows]
    for state in range(states):
        if state > 0:
            advance = [alive[j] or mode is Mode.GLOBAL for j in range(rows)]
            for j in range(rows):
                if advance[j]:
                    current_lower[j], current_upper[j] = lower[state, j], upper[state, j]
            loaded.append(sum(advance))
        winner = int(current_lower.argmax())  # CPU float64 argmax: lowest index on ties
        result = Certificate.check_winner(current_lower, current_upper, winner, tie_break)
        alive = contenders(current_lower, current_upper, tie_break).tolist()
        if result.certified:
            return True, state, winner, loaded + [0] * (states - len(loaded))
    return False, -1, winner, loaded


def test_vectorized_simulation_matches_slow_reference():
    weight, hidden = random_system("cpu", torch.bfloat16, seed=3, vocab=120, inputs=12)
    for spec in ("q6", "q4+q4", "none"):
        decomposition = RefinementDecomposition.build(weight, spec)
        row_bytes = [sum(decomposition.row_bytes(s)) for s in range(decomposition.num_states)]
        intervals = RefinementBatch(decomposition, hidden, numerics_for(weight)).intervals(BoundTier.REALISTIC)
        for mode in Mode:
            for tie_break in TieBreak:
                run = simulate(intervals, mode, tie_break, row_bytes)
                for b in range(hidden.shape[0]):
                    certified, state, winner, loaded = slow_simulation(
                        intervals.lower[:, :, b], intervals.upper[:, :, b], mode, tie_break
                    )
                    assert bool(run.certified[b]) == certified
                    assert int(run.decision_state[b]) == state
                    assert int(run.winner[b]) == winner
                    assert run.rows_loaded[:, b].tolist() == loaded


def test_blocks_touched_matches_brute_force():
    generator = torch.Generator().manual_seed(12)
    rows = torch.rand(5000, 7, generator=generator) < torch.tensor([0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0])
    for row_bytes in (0, 1, 144, 576, 1152, 4096):
        counts = blocks_touched(rows, row_bytes, 4096)
        for b in range(rows.shape[1]):
            expected = set()
            for j in rows[:, b].nonzero().squeeze(1).tolist():
                if row_bytes:
                    expected.update(range(j * row_bytes // 4096, ((j + 1) * row_bytes - 1) // 4096 + 1))
            assert int(counts[b]) == len(expected)
    with pytest.raises(ValueError):
        blocks_touched(rows, 4097, 4096)


@pytest.mark.model
def test_oracle_on_real_lm_head(model, prompt_ids, reference_numerics):
    from awpmi.models.smollm2 import final_hidden_state, lm_head_weight
    from awpmi.reference import ReferenceRunner

    weight = lm_head_weight(model)
    runner = ReferenceRunner(model)
    results = [runner.next_token(ids) for ids in prompt_ids]
    hidden = torch.cat([final_hidden_state(model, ids).reshape(1, -1) for ids in prompt_ids])
    reference = torch.stack([result.logits for result in results], dim=1)
    tokens = torch.tensor([result.token_id for result in results], device=weight.device)
    for spec in ("q8", "q6"):
        decomposition = RefinementDecomposition.build(weight, spec)
        row_bytes = [sum(decomposition.row_bytes(s)) for s in range(decomposition.num_states)]
        batch = RefinementBatch(decomposition, hidden, reference_numerics)
        for tier in BoundTier:
            intervals = batch.intervals(tier)
            assert int(intervals.violations(reference).sum()) == 0
            run = simulate(intervals, Mode.ROW_SELECTIVE, TieBreak.LOWEST_INDEX, row_bytes)
            assert torch.equal(run.winner[run.certified], tokens[run.certified])
            if spec == "q8":
                # An int8 base is decisive: almost no row reaches the exact state.
                assert int(run.rows_loaded[-1].max()) < weight.shape[0] // 100
