"""The coarse pass's binary32 arithmetic stays within its error model, checked in exact arithmetic."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from awpmi.bounds.coarse import CoarseArithmetic, coarse_error_bound, coded_mass_bound
from awpmi.bounds.remainder import input_norms
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF
from awpmi.decomposition.packing import CodeLayout, pack_codes
from awpmi.refinement_head import coarse_matvec
from tests.conftest import DEVICES

COLUMNS = 576


def adversarial_inputs(codes: torch.Tensor, seed: int) -> list[torch.Tensor]:
    """Plain, wide dynamic range, subnormal entries, cancellation against row 0, identical entries."""
    generator = torch.Generator().manual_seed(seed)
    plain = torch.randn(COLUMNS, generator=generator, dtype=torch.float64) * 3
    wide = torch.randn(COLUMNS, generator=generator, dtype=torch.float64) * 10.0 ** torch.randint(
        -20, 20, (COLUMNS,), generator=generator
    )
    tiny = plain * 2.0**-135  # BF16 and binary32 subnormals, and subnormal products
    tiny[::7] = plain[::7]
    row = codes[0].to(torch.float64)
    cancel = plain - (plain @ row) / (row @ row) * row
    same = torch.full((COLUMNS,), 1.0 / 3.0, dtype=torch.float64)
    return [plain, wide, tiny, cancel, same]


def exact(values: list[float]) -> list[Fraction]:
    return [Fraction(v) for v in values]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_coarse_matvec_error_is_within_its_bound(device, dtype, bits):
    limit = 2 ** (bits - 1) - 1
    generator = torch.Generator().manual_seed(bits)
    codes = torch.randint(-limit, limit + 1, (10, COLUMNS), generator=generator).to(torch.int8)
    codes[1] = limit  # identical terms: the worst case for accumulated rounding
    scales = torch.rand(10, generator=generator).to(torch.float32) * 1e-2
    packed = pack_codes(codes.to(device), bits)
    layout = CodeLayout(bits, COLUMNS, device)
    arithmetic = CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, COLUMNS)
    for hidden in adversarial_inputs(codes, seed=bits):
        hidden = hidden.to(dtype).to(device)
        # An odd chunk size exercises the chunked decoding.
        sums = coarse_matvec(packed, layout, hidden.to(torch.float32), chunk_rows=3).cpu()
        mass = coded_mass_bound(scales.to(device), limit, input_norms(hidden).l1).cpu()
        error = coarse_error_bound(scales.to(device), mass.to(device), limit, arithmetic).cpu()
        h = exact(hidden.to(torch.float64).cpu().tolist())
        for j in range(codes.shape[0]):
            q = [Fraction(int(v)) for v in codes[j].tolist()]
            true_sum = sum((a * b for a, b in zip(q, h)), Fraction(0))
            true_mass = sum((abs(a) * abs(b) for a, b in zip(q, h)), Fraction(0))
            scale = Fraction(scales[j].item())
            # The raw model: |fl(S) − S| ≤ γ·Σ|q h| + τ.
            raw = Fraction(arithmetic.gamma) * true_mass + Fraction(arithmetic.underflow(limit))
            assert abs(Fraction(sums[j].item()) - true_sum) <= raw
            # What the runtime uses: B ≥ s·Σ|q h| and |s·fl(S) − s·S| ≤ γ·B + s·τ.
            assert Fraction(mass[j].item()) >= scale * true_mass
            assert abs(scale * Fraction(sums[j].item()) - scale * true_sum) <= Fraction(error[j].item())


def test_coarse_centre_is_exact_in_float64():
    """s_j·fl(S_j) has at most 48 significant bits, so the float64 product is exact."""
    generator = torch.Generator().manual_seed(4)
    scales = torch.rand(1000, generator=generator).to(torch.float32) * 10.0 ** torch.randint(-30, 3, (1000,), generator=generator)
    sums = (torch.randn(1000, generator=generator) * 10.0 ** torch.randint(-30, 30, (1000,), generator=generator)).to(torch.float32)
    product = (sums.to(torch.float64) * scales.to(torch.float64)).tolist()
    for value, s, x in zip(product, scales.tolist(), sums.tolist()):
        assert Fraction(value) == Fraction(s) * Fraction(x)
