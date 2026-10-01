"""Floating-point error accounting and directed rounding.

A certificate about *exact* real-valued logits is not a certificate about the
reference, whose logits are produced by floating-point arithmetic and then
rounded to the output dtype (BF16 for the default model). This module provides
the two pieces needed to bridge that gap soundly:

* `gamma(n, u)` — Higham's bound γ_n = n·u / (1 − n·u). For any evaluation order,
  a floating-point dot product of length n satisfies
  |fl(xᵀy) − xᵀy| ≤ γ_n · Σ|x_k y_k|   (Higham, *Accuracy and Stability of
  Numerical Algorithms*, 2nd ed., §3.1).
* `round_down_to_grid` / `round_up_to_grid` — directed rounding of float64 values
  onto the value grid of a narrower dtype. Any faithful rounding is monotone and
  lands on one of the two neighbouring grid points, so if a ≥ L then
  rnd(a) ≥ round_down_to_grid(L) regardless of the reference's rounding mode.
"""

from __future__ import annotations

import math

import torch

FLOAT64_UNIT_ROUNDOFF = 2.0**-53

# Relative slack applied to radii computed in float64. Every radius is a sum of at
# most `MAX_BOUND_TERMS` non-negative float64 products, whose relative rounding
# error is at most γ_{MAX_BOUND_TERMS + 4}(2⁻⁵³) ≈ 1.1e-11; the slack dominates it.
FLOAT64_BOUND_SLACK = 1e-10
MAX_BOUND_TERMS = 100_000

GRID_DTYPES = (torch.float64, torch.float32, torch.bfloat16, torch.float16)


def gamma(n: int, unit_roundoff: float) -> float:
    """Upper bound on the relative error of an n-term floating-point dot product."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    nu = n * unit_roundoff
    if nu >= 1.0:
        raise ValueError(f"n·u = {nu} >= 1: no finite error bound")
    # Step one ulp up so the float64 evaluation of the formula cannot under-estimate it.
    return math.nextafter(nu / (1.0 - nu), math.inf)


def next_down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def next_up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, math.inf))


def _check_grid_input(x: torch.Tensor, dtype: torch.dtype) -> None:
    if x.dtype != torch.float64:
        raise TypeError(f"expected a float64 tensor, got {x.dtype}")
    if dtype not in GRID_DTYPES:
        raise ValueError(f"unsupported grid dtype {dtype}")
    if torch.isnan(x).any():
        raise ValueError("NaN cannot be rounded to a grid")


def round_down_to_grid(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Largest value representable in `dtype` that is ≤ x, returned as float64.

    Values below the most negative finite number of `dtype` map to -inf.
    """
    _check_grid_input(x, dtype)
    if dtype == torch.float64:
        return x.clone()
    # The float64 -> dtype conversion rounds to nearest (possibly via float32 first);
    # both steps are monotone and the result is one of the two neighbours of x.
    candidate = x.to(dtype)
    too_high = candidate.to(torch.float64) > x
    candidate = torch.where(too_high, next_down(candidate), candidate)
    return candidate.to(torch.float64)


def round_up_to_grid(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Smallest value representable in `dtype` that is ≥ x, returned as float64."""
    _check_grid_input(x, dtype)
    if dtype == torch.float64:
        return x.clone()
    candidate = x.to(dtype)
    too_low = candidate.to(torch.float64) < x
    candidate = torch.where(too_low, next_up(candidate), candidate)
    return candidate.to(torch.float64)
