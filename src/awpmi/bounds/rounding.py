"""Rounding models for the reference's narrow-dtype outputs (decisions 0001, 0004, 0005).

Every reference operation of the adaptive suffix produces a binary32 value (a GEMM's
fp32 accumulator, or an elementwise kernel's fp32 "opmath" result) and rounds it to the
tensor's dtype (BF16 for the default model). A bound on the pre-rounding value becomes
a bound on the stored value through a *rounding model*:

  FAITHFUL      the result is one of the two grid neighbours of the exact value. Any
                monotone rounding, and any composition of them through nested grids
                (binary32, then BF16), is faithful. This is the certified model of
                decision 0001: it assumes nothing about the backend's rounding mode.
  NEAREST_EVEN  round-to-nearest-even, applied to the binary32 value. Probed on this
                platform (decision 0004) but not adopted (decision 0005): it is used only
                for experimental what-if bounds and never participates in `certified=True`.

A `RoundingAssumptions` pair says which model applies to GEMM epilogues and which to
elementwise kernels; `CERTIFIED` is faithful for both.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from awpmi.bounds.floating import round_down_to_grid, round_up_to_grid

# Significand bits (with the implicit bit) and the smallest subnormal exponent of each grid.
_PRECISION = {torch.bfloat16: (8, -133), torch.float16: (11, -24), torch.float32: (24, -149), torch.float64: (53, -1074)}

# A binary32 value that is then rounded again (double rounding) is off by at most half a
# binary32 ulp more: this relative allowance covers it for every narrower grid.
DOUBLE_ROUNDING_ALLOWANCE = 2.0**-24


class RoundingModel(enum.Enum):
    FAITHFUL = "faithful"
    NEAREST_EVEN = "nearest_even"


@dataclass(frozen=True)
class RoundingAssumptions:
    gemm: RoundingModel
    elementwise: RoundingModel

    @property
    def certified(self) -> bool:
        """Only the faithful model may produce a certificate."""
        return self.gemm is RoundingModel.FAITHFUL and self.elementwise is RoundingModel.FAITHFUL

    @property
    def name(self) -> str:
        if self.certified:
            return "faithful"
        if self.gemm is RoundingModel.FAITHFUL:
            return "nearest_even_elementwise"
        return "nearest_even" if self.elementwise is RoundingModel.NEAREST_EVEN else "nearest_even_gemm"


CERTIFIED = RoundingAssumptions(RoundingModel.FAITHFUL, RoundingModel.FAITHFUL)
EXPERIMENTAL_ELEMENTWISE = RoundingAssumptions(RoundingModel.FAITHFUL, RoundingModel.NEAREST_EVEN)
EXPERIMENTAL = RoundingAssumptions(RoundingModel.NEAREST_EVEN, RoundingModel.NEAREST_EVEN)
ASSUMPTIONS = {assumptions.name: assumptions for assumptions in (CERTIFIED, EXPERIMENTAL_ELEMENTWISE, EXPERIMENTAL)}


def _precision(dtype: torch.dtype) -> tuple[int, int]:
    if dtype not in _PRECISION:
        raise ValueError(f"unsupported grid dtype {dtype}")
    return _PRECISION[dtype]


def _nearest_even(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """RN-even of float64 values through binary32 (both conversions are correctly rounded and monotone)."""
    if dtype == torch.float64:
        return x.clone()
    single = x.to(torch.float32)
    return (single if dtype == torch.float32 else single.to(dtype)).to(torch.float64)


def round_enclosure(
    lower: torch.Tensor, upper: torch.Tensor, dtype: torch.dtype, model: RoundingModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid interval holding the rounded value of every real in [lower, upper] (float64 in and out).

    FAITHFUL: [round_down(lower), round_up(upper)]. NEAREST_EVEN: the pre-rounding
    binary32 value v = RN₃₂(a) lies in [RN₃₂(lower), RN₃₂(upper)] by monotonicity, so
    RN(v) lies in [RN(RN₃₂(lower)), RN(RN₃₂(upper))]; a binary32 value inside [lower,
    upper] is covered too.
    """
    if model is RoundingModel.FAITHFUL:
        return round_down_to_grid(lower, dtype), round_up_to_grid(upper, dtype)
    if torch.isnan(lower).any() or torch.isnan(upper).any():
        raise ValueError("NaN cannot be rounded to a grid")
    return _nearest_even(lower, dtype), _nearest_even(upper, dtype)


def spacing_upper(magnitude: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Upper bound on the grid spacing of `dtype` around any value of magnitude ≤ `magnitude` (float64).

    For m = f·2^E with f ∈ [0.5, 1), every |a| ≤ m lies in a binade whose spacing is at
    most 2^(E − p); the subnormal spacing is the floor. Infinite magnitudes give inf.
    """
    precision, subnormal = _precision(dtype)
    magnitude = magnitude.to(torch.float64).abs()
    _, exponent = torch.frexp(magnitude)
    spacing = torch.ldexp(torch.ones_like(magnitude), exponent - precision)
    spacing = torch.where(magnitude > 0, torch.clamp_min(spacing, 2.0**subnormal), 2.0**subnormal)
    return torch.where(torch.isfinite(magnitude), spacing, torch.full_like(spacing, torch.inf))


def rounding_error_upper(magnitude: torch.Tensor, dtype: torch.dtype, model: RoundingModel) -> torch.Tensor:
    """Bound on |rnd(a) − a| for every real |a| ≤ `magnitude` whose rounded value is finite."""
    spacing = spacing_upper(magnitude, dtype)
    if model is RoundingModel.FAITHFUL:
        return spacing
    # Half a spacing, plus the binary32 rounding that precedes it (double rounding).
    return spacing * 0.5 + spacing_upper(magnitude, torch.float32) * 0.5


def relative_rounding_error(dtype: torch.dtype, model: RoundingModel) -> float:
    """r with |rnd(a) − a| ≤ r·|a| + subnormal_floor(dtype), for any real a (finite result)."""
    precision, _ = _precision(dtype)
    if model is RoundingModel.FAITHFUL:
        return 2.0 ** (1 - precision)
    return 2.0**-precision + DOUBLE_ROUNDING_ALLOWANCE


def subnormal_floor(dtype: torch.dtype) -> float:
    """The absolute part of a rounding error near zero: the smallest subnormal spacing."""
    return 2.0 ** _precision(dtype)[1]
