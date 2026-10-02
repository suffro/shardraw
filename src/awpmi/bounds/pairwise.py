"""Scale-free pairwise certificate through the final RMSNorm (Phase 2, decision 0005).

Interval bounds treat every logit separately. Behind the final RMSNorm every logit
carries the same positive factor q = rsqrt(mean(y²) + eps) and the same rounding errors
of each h_k, so separate intervals count them twice and lose the cancellation of the
layers in between. For a candidate w against a contender j, with g the norm weight and
Δ_k = (W_wk − W_jk)·g_k, the reference's values satisfy (`awpmi.bounds.operators`)

    h_k = g_k·y_k·q·(1 + ξ_k) + μ_k,      |ξ_k| ≤ ξ, |μ_k| ≤ μ        (roundings of n_k and h_k)
    acc_w − acc_j = Σ_k (W_wk − W_jk)·h_k + e_w − e_j,      |e_i| ≤ γ_lm·Σ_k |W_ik|·|h_k|

so that acc_w − acc_j ≥ q·F − A with Y_k ≥ |y_k|, L ≤ Σ_k Δ_k·y_k and

    F = L − ξ·Σ_k |Δ_k|·Y_k − γ_lm·(1 + ξ)·Σ_k (|W_wk| + |W_jk|)·|g_k|·Y_k
    A = Σ_k |W_wk − W_jk|·μ + γ_lm·Σ_k (|W_wk| + |W_jk|)·μ

Two lower bounds L are used, the larger one per pair:

  box         Σ_k min(Δ_k·y_k⁻, Δ_k·y_k⁺) over the enclosure of y;
  decomposed  y_k = r_k + Σ_i W_d[k,i]·a_i + e_k + τ_k + ε_k: the residual, the MLP's
              down projection with its accumulation error e_k (|e_k| ≤ E_k), and the
              roundings of o_k (τ_k) and y_k (ε_k). With M = W_dᵀ·Δ,
                L = Δ·r + Σ_{i read} min(M_i·a_i⁻, M_i·a_i⁺) − ‖Δ‖₂·Σ_{i unread} C_i·|a_i|⁺
                    − Σ_k |Δ_k|·(E_k + T_k + Ε_k)
              where C_i ≥ ‖W_d[:, i]‖₂ (Cauchy–Schwarz on the unread columns). It keeps
              the cancellation of the down projection, which the box loses.

Each logit is rounded faithfully and independently (decision 0001), so ℓ_w > ℓ_j once
acc_w − acc_j > 2U, U the grid spacing at the two logits' magnitude; the pair is certified
when q⁻·F − A > 2U. That also settles every tie, whatever the indices. Float64
arithmetic is covered by a relative slack on the sum of the absolute values of all terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.bounds.enclosure import Enclosure
from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma, next_up
from awpmi.bounds.operators import FP32_NORMAL_MIN
from awpmi.bounds.rounding import RoundingAssumptions, RoundingModel, relative_rounding_error, rounding_error_upper, subnormal_floor, spacing_upper


@dataclass(frozen=True)
class DownProjection:
    """What the decomposed bound needs to know about y = r + rnd(W_d·a) (one position)."""

    residual: torch.Tensor  # r, exact, float64 [H]
    read_weight: torch.Tensor  # W_d with unread columns zeroed, float64 [H, I]
    read: torch.Tensor  # bool [I]: columns read
    activation: Enclosure  # a over [I]; entries of unread neurons are ignored
    unread_activation: torch.Tensor  # |a_i| bound of unread neurons, float64 [I]
    column_norms: torch.Tensor  # C_i ≥ ‖W_d[:, i]‖₂, float64 [I]
    accumulation_error: torch.Tensor  # E_k ≥ |e_k|, float64 [H]
    output: Enclosure  # o, which also encloses its pre-rounding value


@dataclass(frozen=True)
class PairwiseResult:
    certified: bool
    winner: int
    contenders: torch.Tensor  # rows j checked against the winner
    margin: torch.Tensor  # q⁻·F − A − 2U per contender (> 0: certified pair)
    box_bound: torch.Tensor  # L_box per contender
    decomposed_bound: torch.Tensor | None  # L_decomposed per contender
    terms: dict[str, float]  # the bound's terms for the tightest pair (provenance of the uncertainty)


def pairwise_certificate(
    winner: int,
    rows: torch.Tensor,
    lm_weight: torch.Tensor,
    logit_lower: torch.Tensor,
    logit_upper: torch.Tensor,
    norm_weight: torch.Tensor,
    y: Enclosure,
    scale_lower: float,
    lm_gamma: float,
    assumptions: RoundingAssumptions,
    down: DownProjection | None = None,
) -> PairwiseResult:
    """Certify that `winner` is the reference argmax against every other row of `rows`.

    `lm_weight` holds the exact LM-head rows of `rows` (original dtype, aligned with
    `rows`); `logit_lower/upper` are those rows' certified logit intervals, aligned with
    `rows` (they bound the logits' magnitude); `lm_gamma` is the LM head's accumulation γ;
    `scale_lower` > 0 bounds the norm's q from below.
    """
    if not scale_lower > 0.0:
        raise ValueError("the norm's scale must be bounded away from zero")
    position = (rows == winner).nonzero()
    if position.numel() != 1:
        raise ValueError("the winner must be one of the rows")
    others = rows != winner
    contenders_ = rows[others]
    w64 = lm_weight[position[0, 0]].to(torch.float64)
    j64 = lm_weight[others].to(torch.float64)
    gain = norm_weight.to(torch.float64)
    difference = w64[None, :] - j64  # W_w − W_j
    delta = difference * gain[None, :]
    absolute_delta = delta.abs()
    magnitude = y.magnitude
    elementwise, gemm = assumptions.elementwise, assumptions.gemm

    # ξ: n = rnd16(rnd32(y·q)), then h = rnd16(rnd32(g·n)); μ: their absolute floors (or flushing).
    r32, r16 = relative_rounding_error(torch.float32, elementwise), relative_rounding_error(norm_weight.dtype, elementwise)
    xi = (1.0 + r32) ** 2 * (1.0 + r16) ** 2 - 1.0
    floor = FP32_NORMAL_MIN + subnormal_floor(norm_weight.dtype)
    mu = (gain.abs() * (1.0 + r32) + 1.0) * floor * (1.0 + r16) ** 2

    box = torch.minimum(delta * y.lower[None, :], delta * y.upper[None, :]).sum(dim=1)
    absolute_terms = absolute_delta @ (magnitude + y.lower.abs() + y.upper.abs())
    decomposed = None
    terms: dict[str, torch.Tensor] = {}
    if down is not None:
        mixed = delta @ down.read_weight  # M = W_dᵀ·Δ for every contender
        a = down.activation
        read_part = torch.where(down.read[None, :], torch.minimum(mixed * a.lower[None, :], mixed * a.upper[None, :]), 0.0)
        read_part = read_part.sum(dim=1)
        unread = torch.where(down.read, 0.0, down.column_norms * down.unread_activation).sum()
        delta_norm = next_up(torch.linalg.vector_norm(delta, dim=1) * (1.0 + 2.0**-40))
        rounding_o = rounding_error_upper(down.output.magnitude, norm_weight.dtype, gemm)
        rounding_y = rounding_error_upper(magnitude, norm_weight.dtype, elementwise)
        a_magnitude = torch.where(down.read, a.magnitude, 0.0)
        terms = {
            "residual_and_read_neurons": delta @ down.residual + read_part,
            "unread_neurons": delta_norm * unread,
            "down_accumulation": absolute_delta @ down.accumulation_error,
            "o_rounding": absolute_delta @ rounding_o,
            "y_rounding": absolute_delta @ rounding_y,
        }
        decomposed = terms["residual_and_read_neurons"] - terms["unread_neurons"] - terms["down_accumulation"]
        decomposed = decomposed - terms["o_rounding"] - terms["y_rounding"]
        absolute_terms = absolute_terms + absolute_delta @ down.residual.abs() + mixed.abs() @ a_magnitude
        absolute_terms = absolute_terms + absolute_delta @ (down.read_weight.abs() @ a_magnitude)
        absolute_terms = absolute_terms + terms["unread_neurons"] + absolute_delta @ (down.accumulation_error + rounding_o + rounding_y)
    lower_bound = box if decomposed is None else torch.maximum(box, decomposed)

    lm_mass = (w64.abs()[None, :] + j64.abs()) * gain.abs()[None, :]
    terms["norm_rounding"] = xi * (absolute_delta @ magnitude)
    terms["lm_accumulation"] = lm_gamma * (1.0 + xi) * (lm_mass @ magnitude)
    bound = lower_bound - terms["norm_rounding"] - terms["lm_accumulation"]
    absolute_terms = absolute_terms + terms["norm_rounding"] + terms["lm_accumulation"]
    # Float64 rounding of every sum and product above (at most 2(H + I) + 64 terms per dot product).
    length = 2 * (delta.shape[1] + (0 if down is None else down.read.numel())) + 64
    slack = 4.0 * gamma(length, FLOAT64_UNIT_ROUNDOFF) * absolute_terms
    absolute = difference.abs() @ mu + lm_gamma * ((w64.abs()[None, :] + j64.abs()) @ mu)

    logit_magnitude = torch.maximum(logit_lower.abs(), logit_upper.abs())
    spacing = spacing_upper(torch.maximum(logit_magnitude[others], logit_magnitude[position[0, 0]]), lm_weight.dtype)
    separations = 2.0 if gemm is RoundingModel.FAITHFUL else 1.0
    # q ≥ q⁻ > 0 multiplies a positive F − slack; a non-positive one can never certify.
    margin = (bound - slack) * scale_lower * (1.0 - 2.0**-50)
    margin = margin - next_up(absolute + separations * spacing)
    certified = bool((margin > 0).all()) if contenders_.numel() else True
    tightest = int(margin.argmin()) if contenders_.numel() else None
    summary = {}
    if tightest is not None:
        summary = {name: float(value[tightest]) for name, value in terms.items()}
        summary.update(box=float(box[tightest]), slack=float(slack[tightest]), margin=float(margin[tightest]))
        if decomposed is not None:
            summary["decomposed"] = float(decomposed[tightest])
    return PairwiseResult(certified, winner, contenders_, margin, box, decomposed, summary)
