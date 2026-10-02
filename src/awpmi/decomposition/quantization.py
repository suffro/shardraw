"""Per-row symmetric round-to-nearest quantization, used to build coarse levels.

values ≈ codes · scale_j, with integer codes in [-(2^(b-1) - 1), 2^(b-1) - 1] and a
float32 scale per row. The approximation codes · scale is exact in float64 (at most
7 + 24 significant bits), so the remainder can be represented exactly.
"""

from __future__ import annotations

import torch

from awpmi.bounds.floating import next_up, round_up_to_grid

SCALE_DTYPE = torch.float32
MAX_BITS = 8


def code_limit(bits: int) -> int:
    if not 2 <= bits <= MAX_BITS:
        raise ValueError(f"bits must be in [2, {MAX_BITS}], got {bits}")
    return 2 ** (bits - 1) - 1


def quantize_rows(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 codes [V, K] and float32 scales [V] for a float64 matrix."""
    if values.dtype != torch.float64 or values.dim() != 2:
        raise TypeError("quantize_rows expects a float64 matrix")
    limit = code_limit(bits)
    row_max = values.abs().amax(dim=1)
    # Round the scale up so that |values / scale| ≤ limit before code rounding.
    scales = round_up_to_grid(next_up(row_max / limit), SCALE_DTYPE)
    scales = torch.where(row_max > 0, scales, torch.zeros_like(scales))
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.clamp(torch.round(values / safe[:, None]), -limit, limit)
    codes = torch.where(scales[:, None] > 0, codes, torch.zeros_like(codes))
    return codes.to(torch.int8), scales.to(SCALE_DTYPE)


def dequantize_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes · scale_j in float64 (exact)."""
    return codes.to(torch.float64) * scales.to(torch.float64)[:, None]
