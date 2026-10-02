"""Phase 2: the LM-head runtime on an enclosure of its input, and reads shared by one token's passes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from awpmi.bounds.enclosure import Enclosure
from awpmi.certificate import contenders
from awpmi.decomposition import RefinementDecomposition
from awpmi.refinement_head import FallbackMode
from tests.conftest import DEVICES
from tests.test_refinement_oracle import random_system
from tests.test_refinement_runtime import make_head

SPECS = ["q8", "q6+q4", "q2+q2+q4"]


def box_around(vector: torch.Tensor, steps: int, generator) -> Enclosure:
    """A BF16-grid box around `vector`: each coordinate widened by 0..steps ulps on each side."""
    lower, upper = vector.clone(), vector.clone()
    for _ in range(steps):
        down = (torch.rand(vector.shape, generator=generator) < 0.5).to(vector.device)
        up = (torch.rand(vector.shape, generator=generator) < 0.5).to(vector.device)
        lower = torch.where(down, torch.nextafter(lower, torch.full_like(lower, -torch.inf)), lower)
        upper = torch.where(up, torch.nextafter(upper, torch.full_like(upper, torch.inf)), upper)
    return Enclosure(lower.to(torch.float64), upper.to(torch.float64))


def samples_in(box: Enclosure, generator, count: int, dtype) -> list[torch.Tensor]:
    lower, upper = box.lower.cpu(), box.upper.cpu()
    rows = [lower, upper]
    for _ in range(count):
        pick = torch.rand(lower.shape, generator=generator) < 0.5
        rows.append(torch.where(pick, lower, upper))
    return [row.to(dtype).to(box.lower.device) for row in rows]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", SPECS)
def test_enclosure_pass_encloses_every_input_in_the_box(device, spec):
    """Every state's interval holds the reference logits of every sampled input; a certificate holds for all."""
    generator = torch.Generator().manual_seed(11)
    weight, hidden = random_system(device, torch.bfloat16, seed=1)
    decomposition = RefinementDecomposition.build(weight, spec)
    head = make_head(decomposition)
    certified = uncertified = 0
    for b in range(hidden.shape[0]):
        for steps in (0, 1, 3):
            box = box_around(hidden[b], steps, generator)
            result = head.run_enclosure(box, head.begin_token(), trace=True)
            references = [F.linear(s.view(1, 1, -1), weight).reshape(-1) for s in samples_in(box, generator, 6, weight.dtype)]
            for reference in references:
                reference64 = reference.to(torch.float64)
                for bounds in result.trace.states:
                    covered = reference64 if bounds.rows is None else reference64[bounds.rows]
                    assert bool(((covered >= bounds.lower) & (covered <= bounds.upper)).all())
                if result.certified:
                    assert result.token_id == int(torch.argmax(reference))
                else:
                    # The reference argmax is always among the rows left alive.
                    assert bool(result.alive[int(torch.argmax(reference))])
            if result.certified:
                certified += 1
            else:
                uncertified += 1
                assert result.reason == "uncertified" and result.decision_state is None
                assert torch.equal(result.alive, contenders(result.lower, result.upper, head.tie_break))
                assert bool(result.alive[result.exact_rows].sum() == result.alive.sum())
    assert certified > 0 and uncertified > 0


def test_dropping_the_input_spread_is_caught(monkeypatch):
    """Guard check: with the enclosure's radius ignored, other inputs of the box leave the logit intervals."""
    monkeypatch.setattr(Enclosure, "radius_about", lambda self, center: torch.zeros_like(center))
    generator = torch.Generator().manual_seed(13)
    weight, hidden = random_system("cpu", torch.bfloat16, seed=1)
    head = make_head(RefinementDecomposition.build(weight, "q8"))
    escaped = 0
    for b in range(hidden.shape[0]):
        box = box_around(hidden[b], 6, generator)
        result = head.run_enclosure(box, head.begin_token(), trace=True)
        for sample in samples_in(box, generator, 4, weight.dtype):
            reference = F.linear(sample.view(1, 1, -1), weight).reshape(-1).to(torch.float64)
            for bounds in result.trace.states:
                covered = reference if bounds.rows is None else reference[bounds.rows]
                escaped += int(((covered < bounds.lower) | (covered > bounds.upper)).sum())
    assert escaped > 0


@pytest.mark.parametrize("device", DEVICES)
def test_point_enclosure_is_a_slightly_wider_phase_1c_run(device):
    weight, hidden = random_system(device, torch.bfloat16, seed=2)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    head = make_head(decomposition)
    agree = 0
    for b in range(hidden.shape[0]):
        exact = head.run(hidden[b], trace=True)
        point = head.run_enclosure(Enclosure.exact(hidden[b]), head.begin_token(), trace=True)
        coarse_exact, coarse_point = exact.trace.states[0], point.trace.states[0]
        assert bool((coarse_point.lower <= coarse_exact.lower).all() and (coarse_point.upper >= coarse_exact.upper).all())
        assert bool((coarse_point.radius - coarse_exact.radius <= coarse_exact.radius * 1e-6 + 1e-30).all())
        agree += int(point.certified == exact.certified and point.contenders == exact.contenders[: len(point.contenders)])
    assert agree >= hidden.shape[0] - 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", list(FallbackMode))
def test_passes_of_one_token_read_each_row_once(device, mode):
    """An enclosure pass, then the exact pass on the same token: no row of any state is read twice."""
    generator = torch.Generator().manual_seed(12)
    weight, hidden = random_system(device, torch.bfloat16, seed=3)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    head = make_head(decomposition, mode)
    exact_passes = 0
    for b in range(hidden.shape[0]):
        held = head.begin_token()
        first = head.run_enclosure(box_around(hidden[b], 4, generator), held)
        if first.certified:
            continue
        exact_passes += 1
        result = head.run(hidden[b], held=held)
        reference = F.linear(hidden[b].view(1, 1, -1), weight).reshape(-1)
        assert result.token_id == int(torch.argmax(reference))
        if result.fallback is FallbackMode.FULL:
            assert torch.equal(result.fallback_logits, reference)
        seen: dict[tuple[str, int], torch.Tensor] = {}
        for read in result.reads:
            rows = torch.arange(weight.shape[0], device=weight.device) if read.rows is None else read.rows
            key = ("weight" if read.kind in ("exact", "fallback") else read.kind, read.state)
            mask = seen.setdefault(key, torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device))
            assert not bool(mask[rows].any()), key
            mask[rows] = True
        per_state = [0] * decomposition.num_states
        for read in result.reads:
            if read.kind in ("level", "exact"):
                per_state[read.state] += read.row_count
        fallback_rows = result.bytes["fallback_rows"] // decomposition.exact_row_bytes
        assert result.bytes == decomposition.materialized_bytes(per_state, fallback_rows)
    assert exact_passes > 0
