"""Diagnostic simulator: progressive precision refinement, optionally row-selective.

Nothing here is a runtime path. For a `RefinementDecomposition` and a batch of
exact LM-head inputs, it answers when a certificate could fire and how many bytes
would have been read by then. Inputs are processed as [V, B] columns.

Bound tiers, i.e. how the missing remainder X_s of a row in state s is bounded:

  realistic  min(‖X_s[j]‖₂‖h‖₂, ‖X_s[j]‖_∞‖h‖₁), from resident metadata. A runtime can
             compute this.
  abs_mass   Σ_k |X_s[j,k]|·|h_k|: the best sign-agnostic bound. Diagnostic; it needs X_s.
  ideal      |X_s[j]·h|: the true missing contribution. Unattainable; diagnostic only.

Every tier keeps the reference semantics of decision 0001 (accumulation error,
float64 error of the centre, directed rounding onto the output grid). For row j
in state s < n:

  centre  c_j = Σ_{l≤s} fl(L_l[j]·h)
  radius  r_j = miss_j + γ_acc·A_j + γ₆₄·M_j

M_j ≥ Σ_{l≤s} Σ_k |L_l[j,k] h_k| bounds the centre's float64 error, and
A_j ≥ Σ_k |W[j,k] h_k| bounds the reference accumulator's error. realistic uses
A_j = M_j + miss_j, which a runtime can compute. abs_mass and ideal use the exact
absolute mass of W[j]. In the exact state, c_j = fl(W[j]·h), miss_j = 0 and
A_j = M_j = Σ_k |W[j,k] h_k|. A row's intervals from successive states are
intersected, since each one encloses the reference logit.

Modes:

  global         every row advances one state per round
  row_selective  only contenders advance (rows not provably beaten; see
                 `awpmi.certificate.contenders`)

Both modes certify at the same state, because an eliminated row can never block
a certificate. row_selective just reads fewer rows. The certificate's winner is
the row with the largest lower bound, the only row that can certify.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from awpmi.bounds.linear import absolute_mass_upper
from awpmi.bounds.remainder import input_norms
from awpmi.bounds.residual import ReferenceNumerics, exact_radius, reference_logit_interval, remainder_radius
from awpmi.certificate import TieBreak, certify_columns, contenders
from awpmi.decomposition import RefinementDecomposition
from awpmi.decomposition.accounting import IO_BLOCK_BYTES, blocks_touched

__all__ = [
    "IO_BLOCK_BYTES",
    "BoundTier",
    "Mode",
    "RefinementBatch",
    "Simulation",
    "StateIntervals",
    "blocks_touched",
    "simulate",
]


class BoundTier(enum.Enum):
    REALISTIC = "realistic"
    ABS_MASS = "abs_mass"
    IDEAL = "ideal"


class Mode(enum.Enum):
    GLOBAL = "global"
    ROW_SELECTIVE = "row_selective"


@dataclass(frozen=True)
class StateIntervals:
    """Bounds on the reference logits for every state, intersected over earlier states: [S, V, B]."""

    lower: torch.Tensor
    upper: torch.Tensor

    def violations(self, reference_logits: torch.Tensor) -> torch.Tensor:
        """Per column: (state, row) pairs whose interval misses the reference logit. Must be 0."""
        reference = reference_logits.to(torch.float64)[None]
        outside = (reference < self.lower) | (reference > self.upper)
        return outside.sum(dim=(0, 1))


class RefinementBatch:
    """State centres, masses and missing-remainder bounds for one decomposition and a batch of inputs."""

    def __init__(self, decomposition: RefinementDecomposition, hidden: torch.Tensor, numerics: ReferenceNumerics):
        if hidden.dim() != 2 or hidden.shape[1] != decomposition.in_features:
            raise ValueError("hidden must be [B, in_features]")
        if hidden.dtype != decomposition.weight.dtype:
            raise TypeError("hidden dtype must match the weight dtype")
        if numerics.reduction_length != decomposition.in_features:
            raise ValueError("numerics.reduction_length must equal in_features")
        # γ₆₄ = γ_{2K+2} covers K-term dot products plus a sum over at most K + 2 levels.
        if decomposition.num_states > decomposition.in_features + 2:
            raise ValueError("too many levels for the float64 error model")
        self.decomposition = decomposition
        self.numerics = numerics
        columns = hidden.to(torch.float64).T
        inputs = input_norms(hidden)
        weight = decomposition.weight.to(torch.float64)

        self.centers: list[torch.Tensor] = []
        self.level_masses: list[torch.Tensor] = []
        self.missing: dict[BoundTier, list[torch.Tensor]] = {tier: [] for tier in BoundTier}
        center = torch.zeros(weight.shape[0], columns.shape[1], dtype=torch.float64, device=weight.device)
        mass = torch.zeros_like(center)
        remainder = weight
        for state, level in enumerate(decomposition.levels):
            if level.bits:
                values = level.values()
                center = center + values @ columns
                mass = mass + absolute_mass_upper(values, columns)
                remainder = remainder - values
            self.centers.append(center)
            self.level_masses.append(mass)
            remainder_mass = absolute_mass_upper(remainder, columns)
            self.missing[BoundTier.REALISTIC].append(decomposition.residual_bound(state, inputs))
            self.missing[BoundTier.ABS_MASS].append(remainder_mass)
            self.missing[BoundTier.IDEAL].append((remainder @ columns).abs() + numerics.float64_gamma * remainder_mass)
        self.exact_center = weight @ columns
        self.weight_mass = absolute_mass_upper(weight, columns)

    def intervals(self, tier: BoundTier) -> StateIntervals:
        lowers, uppers = [], []
        for state in range(self.decomposition.exact_state):
            miss = self.missing[tier][state]
            if tier is BoundTier.REALISTIC:
                reference_mass = center_mass = self.level_masses[state] + miss
            else:
                reference_mass, center_mass = self.weight_mass, self.level_masses[state]
            radius = remainder_radius(miss, reference_mass, center_mass, self.numerics)
            lower, upper = reference_logit_interval(self.centers[state], radius, self.numerics.output_dtype)
            lowers.append(lower)
            uppers.append(upper)
        radius = exact_radius(self.weight_mass, self.numerics)
        lower, upper = reference_logit_interval(self.exact_center, radius, self.numerics.output_dtype)
        lowers.append(lower)
        uppers.append(upper)
        return StateIntervals(
            lower=torch.stack(lowers).cummax(dim=0).values,
            upper=torch.stack(uppers).cummin(dim=0).values,
        )


@dataclass(frozen=True)
class Simulation:
    """Per column (input): the decision and what was read. [S, B] fields are per state."""

    certified: torch.Tensor
    decision_state: torch.Tensor
    winner: torch.Tensor
    competitor: torch.Tensor
    margin: torch.Tensor
    contenders: torch.Tensor
    rows_loaded: torch.Tensor
    io_blocks: torch.Tensor
    final_contenders: torch.Tensor


def simulate(
    intervals: StateIntervals,
    mode: Mode,
    tie_break: TieBreak,
    row_bytes: list[int],
    block_bytes: int = IO_BLOCK_BYTES,
) -> Simulation:
    """Advance rows state by state until each column certifies or reaches the exact state."""
    states, rows, count = intervals.lower.shape
    device = intervals.lower.device
    active = torch.ones(count, dtype=torch.bool, device=device)
    certified = torch.zeros(count, dtype=torch.bool, device=device)
    decision = torch.full((count,), -1, dtype=torch.int64, device=device)
    winner = torch.zeros(count, dtype=torch.int64, device=device)
    competitor = torch.zeros(count, dtype=torch.int64, device=device)
    margin = torch.zeros(count, dtype=torch.float64, device=device)
    contender_counts = torch.full((states, count), -1, dtype=torch.int64, device=device)
    rows_loaded = torch.zeros(states, count, dtype=torch.int64, device=device)
    io_blocks = torch.zeros(states, count, dtype=torch.int64, device=device)

    lower, upper = intervals.lower[0].clone(), intervals.upper[0].clone()
    advance = torch.ones(rows, count, dtype=torch.bool, device=device)
    alive = advance
    for state in range(states):
        if state > 0:
            eligible = alive if mode is Mode.ROW_SELECTIVE else torch.ones_like(alive)
            advance = eligible & active[None, :]
            lower = torch.where(advance, intervals.lower[state], lower)
            upper = torch.where(advance, intervals.upper[state], upper)
        rows_loaded[state] = advance.sum(dim=0)
        io_blocks[state] = blocks_touched(advance, row_bytes[state], block_bytes)
        result = certify_columns(lower, upper, tie_break)
        alive = contenders(lower, upper, tie_break)
        contender_counts[state] = torch.where(active, alive.sum(dim=0), contender_counts[state])
        winner = torch.where(active, result.winner, winner)
        competitor = torch.where(active, result.competitor, competitor)
        margin = torch.where(active, result.margin, margin)
        newly = active & result.certified
        certified |= newly
        decision = torch.where(newly, state, decision)
        active &= ~result.certified
        if not bool(active.any()):
            break
    return Simulation(
        certified=certified,
        decision_state=decision,
        winner=winner,
        competitor=competitor,
        margin=margin,
        contenders=contender_counts,
        rows_loaded=rows_loaded,
        io_blocks=io_blocks,
        final_contenders=alive,
    )
