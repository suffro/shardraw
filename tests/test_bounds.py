"""Bound soundness: every bound must enclose the true value on every tested sample."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from awpmi.bounds.floating import gamma, round_down_to_grid, round_up_to_grid
from awpmi.bounds.linear import block_l2_norm_upper
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics, ResidualBounder
from awpmi.paging.index import column_partition
from tests.conftest import DEVICES


def all_finite_bf16(device: str) -> torch.Tensor:
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).to(device)
    return torch.unique(values[torch.isfinite(values)]).to(torch.float64)


def test_gamma_matches_formula_and_rejects_overflow():
    u = 2.0**-24
    assert gamma(576, u) >= 576 * u / (1 - 576 * u)
    assert gamma(0, u) >= 0.0
    with pytest.raises(ValueError):
        gamma(2**24, u)


@pytest.mark.parametrize("device", DEVICES)
def test_directed_rounding_bf16_exhaustive(device):
    """For every adjacent bf16 pair (a, b): points in [a, b] round down to a and up to b."""
    grid = all_finite_bf16(device)
    lower, upper = grid[:-1], grid[1:]
    for t in (0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9):
        x = lower + (upper - lower) * t
        down = round_down_to_grid(x, torch.bfloat16)
        up = round_up_to_grid(x, torch.bfloat16)
        on_grid = x == lower
        assert torch.equal(down, torch.where(x == upper, upper, lower))
        assert torch.equal(up, torch.where(on_grid, lower, upper))


@pytest.mark.parametrize("device", DEVICES)
def test_directed_rounding_float32_random(device):
    generator = torch.Generator().manual_seed(0)
    x = (torch.randn(200_000, generator=generator, dtype=torch.float64) * 10.0 ** torch.randint(-30, 30, (200_000,), generator=generator)).to(device)
    down = round_down_to_grid(x, torch.float32)
    up = round_up_to_grid(x, torch.float32)
    assert bool((down <= x).all()) and bool((up >= x).all())
    assert torch.equal(down.to(torch.float32).to(torch.float64), down)
    assert torch.equal(up.to(torch.float32).to(torch.float64), up)
    # Tightness: no float32 value strictly between the bound and x.
    next_after_down = torch.nextafter(down.to(torch.float32), torch.full_like(x, math.inf, dtype=torch.float32))
    assert bool((next_after_down.to(torch.float64) > x).all())


def test_directed_rounding_overflow_is_conservative():
    big = torch.tensor([1e300, -1e300], dtype=torch.float64)
    bf16_max = torch.finfo(torch.bfloat16).max
    assert round_down_to_grid(big, torch.bfloat16).tolist() == [bf16_max, -math.inf]
    assert round_up_to_grid(big, torch.bfloat16).tolist() == [math.inf, -bf16_max]


def exact_sum_of_squares(row: list[float]) -> Fraction:
    return sum((Fraction(v) ** 2 for v in row), Fraction(0))


@pytest.mark.parametrize("storage_dtype", [torch.float64, torch.float32])
def test_block_norm_upper_bound_is_sound_exactly(storage_dtype):
    """Checked in exact rational arithmetic: upper² ≥ Σ x² for every row and block."""
    generator = torch.Generator().manual_seed(1)
    scale = 10.0 ** torch.randint(-6, 6, (24, 1), generator=generator).to(torch.float64)
    values = (torch.randn(24, 70, generator=generator, dtype=torch.float64) * scale).to(torch.bfloat16)
    values[0] = 1.0  # identical entries: worst case for summation rounding
    slices = column_partition(70, 16)
    upper = block_l2_norm_upper(values, slices, storage_dtype=storage_dtype)
    for row in range(values.shape[0]):
        for page, column_slice in enumerate(slices):
            exact = exact_sum_of_squares(values[row, column_slice].to(torch.float64).tolist())
            assert Fraction(upper[row, page].item()) ** 2 >= exact


def test_cauchy_schwarz_page_bound_exactly():
    """|W[j,p]·h_p| ≤ N[j,p]·‖h_p‖ in exact arithmetic, including the CS-tight case h ∥ W[j]."""
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(16, 48, generator=generator).to(torch.bfloat16)
    hidden = torch.randn(48, generator=generator).to(torch.bfloat16)
    weight[3] = hidden  # parallel row: Cauchy–Schwarz holds with equality
    slices = column_partition(48, 8)
    row_norms = block_l2_norm_upper(weight, slices)
    hidden_norms = block_l2_norm_upper(hidden, slices)
    for row in range(16):
        for page, column_slice in enumerate(slices):
            w = [Fraction(v) for v in weight[row, column_slice].to(torch.float64).tolist()]
            h = [Fraction(v) for v in hidden[column_slice].to(torch.float64).tolist()]
            exact = abs(sum((a * b for a, b in zip(w, h)), Fraction(0)))
            assert exact <= Fraction(row_norms[row, page].item()) * Fraction(hidden_norms[page].item())


def bounder_for(
    weight: torch.Tensor, hidden: torch.Tensor, width: int, output_dtype: torch.dtype
) -> tuple[ResidualBounder, list[slice]]:
    slices = column_partition(weight.shape[1], width)
    numerics = ReferenceNumerics(output_dtype, FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1])
    return ResidualBounder(
        block_l2_norm_upper(weight, slices).to(torch.float64),
        block_l2_norm_upper(hidden, slices),
        numerics,
    ), slices


def partial_logits(weight, hidden, slices, materialized):
    partial = torch.zeros(weight.shape[0], dtype=torch.float64, device=weight.device)
    for p in materialized:
        partial += weight[:, slices[p]].to(torch.float64) @ hidden[slices[p]].to(torch.float64)
    return partial


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_reference_gemm_lies_in_envelope(device, dtype):
    """The accumulation model holds for the real backend GEMM, including cancellation-heavy rows."""
    generator = torch.Generator().manual_seed(3)
    vocab, hidden_size = 8192, 576
    weight = torch.randn(vocab, hidden_size, generator=generator)
    hidden = torch.randn(hidden_size, generator=generator) * 4
    # Cancellation: rows nearly orthogonal to h make |Σ w h| ≪ Σ |w h|.
    projection = (weight[:2048] @ hidden) / (hidden @ hidden)
    weight[:2048] -= projection[:, None] * hidden[None, :]
    # Rows with a heavy-tailed scale.
    weight[2048:4096] *= 10.0 ** torch.randint(-3, 3, (2048, 1), generator=generator)
    weight, hidden = weight.to(dtype).to(device), hidden.to(dtype).to(device)
    reference = F.linear(hidden.view(1, 1, -1), weight).reshape(-1).to(torch.float64)
    bounder, slices = bounder_for(weight, hidden, 32, dtype)
    partial = partial_logits(weight, hidden, slices, range(len(slices)))
    lower, upper = bounder.logit_bounds(partial, torch.zeros(len(slices), dtype=torch.bool, device=device))
    assert bool((reference >= lower).all()) and bool((reference <= upper).all())


@pytest.mark.parametrize("device", DEVICES)
def test_residual_bounds_enclose_reference_for_any_subset(device):
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(512, 96, generator=generator).to(torch.bfloat16).to(device)
    hidden = (torch.randn(96, generator=generator) * 3).to(torch.bfloat16).to(device)
    reference = F.linear(hidden.view(1, 1, -1), weight).reshape(-1).to(torch.float64)
    bounder, slices = bounder_for(weight, hidden, 8, torch.bfloat16)
    for trial in range(50):
        mask = torch.rand(len(slices), generator=generator) < trial / 50
        materialized = [p for p in range(len(slices)) if not mask[p]]
        partial = partial_logits(weight, hidden, slices, materialized)
        lower, upper = bounder.logit_bounds(partial, mask.to(device))
        assert bool((reference >= lower).all()) and bool((reference <= upper).all())
