"""Precision-refinement decompositions: exact reconstruction, metadata soundness, byte accounting."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from awpmi.decomposition import RefinementDecomposition, parse_spec, quantize_rows
from awpmi.decomposition.base import RefinementLevel, subtraction_error
from tests.conftest import DEVICES

SPECS = ["none", "q8", "q6", "q4", "q2", "q4+q4", "q6+q4", "q2+q2+q4"]


def awkward_weight(device: str, rows: int = 12, columns: int = 40) -> torch.Tensor:
    """BF16 rows with heavy-tailed scales, an outlier, tiny entries and an all-zero row."""
    generator = torch.Generator().manual_seed(11)
    scale = 10.0 ** torch.randint(-5, 2, (rows, 1), generator=generator).to(torch.float64)
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float64) * scale
    weight[1, 3] = 300.0  # one outlier dominates the row scale
    weight[2, :20] = 1e-30  # far below the row scale
    weight[4] = 0.0
    return weight.to(torch.bfloat16).to(device)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_quantize_rows_codes_scales_and_error(bits):
    generator = torch.Generator().manual_seed(bits)
    values = torch.randn(64, 96, generator=generator, dtype=torch.float64) * 10.0 ** torch.randint(
        -4, 3, (64, 1), generator=generator
    )
    values[5] = 0.0
    codes, scales = quantize_rows(values, bits)
    limit = 2 ** (bits - 1) - 1
    assert codes.dtype == torch.int8 and scales.dtype == torch.float32
    assert int(codes.abs().max()) <= limit
    assert bool((scales.double() * limit >= values.abs().amax(dim=1)).all())
    assert float(scales[5]) == 0.0 and int(codes[5].abs().max()) == 0
    remainder = values - codes.double() * scales.double()[:, None]
    # Round-to-nearest without clipping: every entry is within half a step.
    assert bool((remainder.abs() <= scales.double()[:, None] / 2).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", SPECS)
def test_reconstruction_is_exact_in_rational_arithmetic(device, spec):
    """W = Σ levels + final remainder, checked entry by entry with Fractions."""
    weight = awkward_weight(device)
    decomposition = RefinementDecomposition.build(weight, spec)
    last = decomposition.exact_state - 1
    remainder = decomposition.remainder(last).cpu().tolist()
    stored = [level.values().cpu().tolist() for level in decomposition.levels if level.bits]
    original = weight.to(torch.float64).cpu().tolist()
    for j, row in enumerate(original):
        for k, value in enumerate(row):
            total = Fraction(remainder[j][k]) + sum((Fraction(level[j][k]) for level in stored), Fraction(0))
            assert total == Fraction(value)


@pytest.mark.parametrize("spec", SPECS)
def test_remainder_metadata_bounds_every_state_exactly(spec):
    weight = awkward_weight("cpu")
    decomposition = RefinementDecomposition.build(weight, spec)
    for state, norms in enumerate(decomposition.remainder_norms):
        remainder = decomposition.remainder(state).tolist()
        for j, row in enumerate(remainder):
            assert Fraction(norms.l2[j].item()) ** 2 >= sum((Fraction(v) ** 2 for v in row), Fraction(0))
            assert Fraction(norms.linf[j].item()) >= max(abs(Fraction(v)) for v in row)


def test_subtraction_error_detects_inexact_subtraction():
    a = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64)
    b = torch.tensor([0.5, 2.0**-60, 2.0**-30], dtype=torch.float64)
    difference, error = subtraction_error(a, b)
    assert difference.tolist()[0] == 0.5 and error.tolist() == [0.0, -(2.0**-60), 0.0]


def test_build_rejects_an_inexact_remainder(monkeypatch):
    """The exactness guard fires: a level whose subtraction rounds is refused."""
    weight = torch.ones(3, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(
        RefinementLevel, "values", lambda self: torch.full((3, 8), 2.0**-60, dtype=torch.float64)
    )
    with pytest.raises(ArithmeticError):
        RefinementDecomposition.build(weight, "q4")


def test_parse_spec():
    assert parse_spec("none") == (0,)
    assert parse_spec("q8") == (8,)
    assert parse_spec("q2+q2+q4") == (2, 2, 4)
    for bad in ("q1", "q9", "int8", "q4+", "q4++q4", ""):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_byte_accounting_counts_every_resident_and_loaded_byte():
    rows, columns = 10, 576
    weight = torch.randn(rows, columns).to(torch.bfloat16)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    assert decomposition.num_states == 3 and decomposition.state_name(2) == "exact"
    assert decomposition.row_bytes(0) == (432, 4)
    assert decomposition.row_bytes(1) == (288, 4)
    assert decomposition.row_bytes(2) == (1152, 0)
    assert decomposition.metadata_bytes() == 2 * rows * 8
    breakdown = decomposition.materialized_bytes([rows, 3, 2], fallback_rows=0)
    assert breakdown == {
        "metadata": 160,
        "base_payload": rows * 432,
        "base_scales": rows * 4,
        "refinement_payload": 3 * 288,
        "refinement_scales": 3 * 4,
        "exact_rows": 2 * 1152,
        "fallback_rows": 0,
        "total": 160 + rows * 436 + 3 * 292 + 2 * 1152,
    }
    assert decomposition.materialized_bytes([rows, 0, 0], fallback_rows=rows)["fallback_rows"] == rows * 1152
    storage = decomposition.storage_bytes()
    assert storage == {"levels": rows * (436 + 292), "original": rows * 1152, "metadata": 160, "total": storage["total"]}
    assert storage["total"] == rows * (436 + 292 + 1152) + 160

    none = RefinementDecomposition.build(weight, "none")
    assert none.row_bytes(0) == (0, 0) and none.metadata_bytes() == rows * 8
    with pytest.raises(ValueError):
        none.base_representation().values()
