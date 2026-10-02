"""Rounding models (decision 0005): grid spacing, enclosures of rounded values, error bounds."""

from __future__ import annotations

import pytest
import torch

from awpmi.bounds.floating import round_down_to_grid, round_up_to_grid
from awpmi.bounds.rounding import (
    CERTIFIED,
    EXPERIMENTAL,
    EXPERIMENTAL_ELEMENTWISE,
    RoundingModel,
    relative_rounding_error,
    round_enclosure,
    rounding_error_upper,
    spacing_upper,
    subnormal_floor,
)
from tests.conftest import DEVICES

GRIDS = [torch.bfloat16, torch.float16, torch.float32]


def all_finite_bfloat16(device) -> torch.Tensor:
    """Every finite BF16 value, as float64."""
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).to(torch.float64)
    return values[torch.isfinite(values)].to(device)


def wide_reals(generator, count: int) -> torch.Tensor:
    """Reals over the whole BF16 range, plus subnormals, zeros, grid points and their midpoints."""
    exponents = torch.randint(-140, 120, (count,), generator=generator).to(torch.float64)
    signs = torch.where(torch.rand(count, generator=generator) < 0.5, -1.0, 1.0)
    reals = signs * torch.rand(count, generator=generator, dtype=torch.float64).add(1.0) * torch.exp2(exponents)
    grid = reals.to(torch.bfloat16).to(torch.float64)
    upper = round_up_to_grid(torch.nextafter(grid, torch.full_like(grid, torch.inf)), torch.bfloat16)
    midpoints = grid * 0.5 + upper * 0.5
    return torch.cat([reals, grid, midpoints, torch.tensor([0.0, -0.0, 2.0**-133, -(2.0**-133), 1.0, -1.0])])


@pytest.mark.parametrize("device", DEVICES)
def test_spacing_bounds_every_bfloat16_gap(device):
    values = all_finite_bfloat16(device)
    up = torch.nextafter(values.to(torch.bfloat16), torch.full_like(values, torch.inf).to(torch.bfloat16))
    down = torch.nextafter(values.to(torch.bfloat16), torch.full_like(values, -torch.inf).to(torch.bfloat16))
    finite = torch.isfinite(up.to(torch.float64)) & torch.isfinite(down.to(torch.float64))
    spacing = spacing_upper(values, torch.bfloat16)
    assert bool((up.to(torch.float64) - values <= spacing)[finite].all())
    assert bool((values - down.to(torch.float64) <= spacing)[finite].all())
    # And it is tight: within a factor 2 of the actual gap (exactly the gap above, off a power of two).
    gap = (up.to(torch.float64) - values)[finite & (values > 0)]
    assert bool((spacing[finite & (values > 0)] <= 2 * gap).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", GRIDS)
@pytest.mark.parametrize("model", list(RoundingModel))
def test_rounding_error_bounds_hold_for_actual_conversions(device, dtype, model):
    generator = torch.Generator().manual_seed(0)
    reals = wide_reals(generator, 20000).to(device)
    if dtype != torch.bfloat16:
        reals = reals[reals.abs() < torch.finfo(dtype).max / 2]
    if model is RoundingModel.FAITHFUL:
        candidates = [round_down_to_grid(reals, dtype), round_up_to_grid(reals, dtype)]
    else:
        candidates = [reals.to(torch.float32).to(dtype).to(torch.float64)]
    absolute = rounding_error_upper(reals.abs(), dtype, model)
    relative = relative_rounding_error(dtype, model) * reals.abs() + subnormal_floor(dtype)
    for rounded in candidates:
        finite = torch.isfinite(rounded)
        error = (rounded - reals).abs()[finite]
        assert bool((error <= absolute[finite]).all())
        assert bool((error <= relative[finite]).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", GRIDS)
def test_round_enclosure_contains_every_rounding_of_every_point(device, dtype):
    generator = torch.Generator().manual_seed(1)
    centers = wide_reals(generator, 4000).to(device)
    if dtype != torch.bfloat16:
        centers = centers[centers.abs() < torch.finfo(dtype).max / 4]
    widths = centers.abs() * torch.rand(centers.shape, generator=generator, dtype=torch.float64).to(device) * 0.05
    lower, upper = centers - widths, centers + widths
    faithful = round_enclosure(lower, upper, dtype, RoundingModel.FAITHFUL)
    nearest = round_enclosure(lower, upper, dtype, RoundingModel.NEAREST_EVEN)
    assert bool((faithful[0] <= nearest[0]).all() and (nearest[1] <= faithful[1]).all())
    for t in torch.linspace(0, 1, 9, dtype=torch.float64):
        point = lower + (upper - lower) * t
        for rounded in (round_down_to_grid(point, dtype), round_up_to_grid(point, dtype)):
            assert bool(((rounded >= faithful[0]) & (rounded <= faithful[1])).all())
        # The reference's own conversions: binary32 first (an fp32 accumulator), then the grid.
        actual = point.to(torch.float32).to(dtype).to(torch.float64)
        assert bool(((actual >= nearest[0]) & (actual <= nearest[1])).all())


def test_only_the_faithful_pair_certifies():
    assert CERTIFIED.certified and CERTIFIED.name == "faithful"
    assert not EXPERIMENTAL.certified and EXPERIMENTAL.name == "nearest_even"
    assert not EXPERIMENTAL_ELEMENTWISE.certified and EXPERIMENTAL_ELEMENTWISE.name == "nearest_even_elementwise"
