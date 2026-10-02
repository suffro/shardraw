"""Phase 1C runtime: progressive precision refinement of the LM head, row-selective, certified.

For one exact LM-head input h (decision 0004):

  state 0      read the packed base level of every row; S_j = Σ_k q_jk·h_k in binary32,
               c_j = s_j·fl(S_j); interval from `awpmi.bounds.coarse`
  certify      tie-aware certificate on every row's interval (`certify_columns`)
  eliminate    keep only the rows that can still be the reference argmax (`contenders`)
  state s ≥ 1  read only those rows of the next level and recompute them in float64
               exactly as the oracle does (`awpmi.oracle.refinement`, realistic tier)
  exact state  read only the surviving original rows; centre fl₆₄(W_j·h)
  fallback     still UNKNOWN after the exact state: recompute the reference operation

A row's intervals from successive states are intersected. The token is the
certified winner or the fallback's argmax, never an estimate. Weights are read only
through the `PackedRefinementStore`, which counts every byte.

Fallbacks (`FallbackMode`):

  FULL    read every original row not read yet and apply F.linear(h, W): the
          reference operation, bitwise (decision 0002).
  MASKED  apply F.linear(h, W') with the full shape, where W' keeps the surviving rows
          (already read in the exact state) and zeros elsewhere, and take the argmax
          over the survivors. It assumes that each output row of the reference GEMM
          depends only on its own weight row: verified by a self-test on the platform
          when the head is built, and guarded per call by checking that every
          surviving logit lies in its certified interval (otherwise FULL runs).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from awpmi.bounds.coarse import CoarseArithmetic, coarse_error_bound, coarse_radius, coded_mass_bound
from awpmi.bounds.linear import absolute_mass_upper
from awpmi.bounds.remainder import RowNormBounds, input_norms, remainder_mass_bound
from awpmi.bounds.residual import ReferenceNumerics, exact_radius, reference_logit_interval, remainder_radius
from awpmi.certificate import TieBreak, certify_columns, contenders
from awpmi.decomposition.packing import CodeLayout
from awpmi.decomposition.quantization import dequantize_rows
from awpmi.stores.refinement import PackedRefinementStore, Read
from awpmi.tracing import NullTimer, StageTimer

DEFAULT_CHUNK_ROWS = 8192


class FallbackMode(enum.Enum):
    FULL = "full"
    MASKED = "masked"


@dataclass(frozen=True)
class StateBounds:
    """The interval one state produced for the rows it computed (before intersection with earlier states)."""

    state: int
    rows: torch.Tensor | None  # None: every row
    center: torch.Tensor
    radius: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class RunTrace:
    """Everything needed to validate a run against the reference; collected only on request."""

    coarse_sum: torch.Tensor  # fl₃₂(S_j) for every row
    coarse_error: torch.Tensor  # its bound γ·B_j + s_j·τ
    states: tuple[StateBounds, ...]
    lower: torch.Tensor  # final intersected intervals
    upper: torch.Tensor
    final_contenders: torch.Tensor | None  # rows alive after the last certificate check


@dataclass(frozen=True)
class RefinementRunResult:
    token_id: int
    certified: bool
    fallback: FallbackMode | None
    fallback_reason: str | None  # "uncertified", "nonfinite_input" or "coarse_overflow"
    masked_guard_tripped: bool
    decision_state: int | None
    winner: int
    competitor: int
    certificate_margin: float
    contenders: tuple[int, ...]  # alive rows after each certificate check
    rows_loaded: tuple[int, ...]  # rows that entered each state
    bytes: dict[str, int]
    reads: tuple[Read, ...]
    fallback_logits: torch.Tensor | None
    timings_ms: dict[str, float]
    trace: RunTrace | None


def coarse_matvec(payload: torch.Tensor, layout: CodeLayout, vector: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """fl₃₂(Σ_k q_jk·v_k) for every packed row, decoding `chunk_rows` rows at a time (float32 [V]).

    The error model is `awpmi.bounds.coarse`; only binary32 arithmetic is used.
    """
    if vector.dtype != torch.float32:
        raise TypeError("the coarse pass runs in float32")
    out = torch.empty(payload.shape[0], dtype=torch.float32, device=payload.device)
    for start in range(0, payload.shape[0], chunk_rows):
        codes = layout.unpack(payload[start : start + chunk_rows])
        out[start : start + chunk_rows] = torch.mv(codes.to(torch.float32), vector)
    return out


def masked_fallback_self_test(store: PackedRefinementStore, trials: int = 8, seed: int = 0) -> dict[str, int]:
    """Does F.linear(h, W') reproduce the rows kept in W' bitwise on this platform?

    Random inputs and random row subsets, from one row to half of them. The reads go
    through the store and are logged as a throwaway run.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = store.out_features
    store.begin_run()
    weight = store.read_fallback(torch.arange(rows, device=store.device))
    checks = mismatches = 0
    for trial in range(trials):
        scale = 10.0 ** float(torch.empty(()).uniform_(-1.0, 2.0, generator=generator))
        hidden = (torch.randn(store.in_features, generator=generator) * scale).to(store.dtype).to(store.device)
        density = 0.5 ** (trial % 16 + 1)
        keep = (torch.rand(rows, generator=generator) < density).to(store.device)
        keep[int(torch.randint(rows, (1,), generator=generator))] = True
        masked = torch.where(keep[:, None], weight, torch.zeros((), dtype=weight.dtype, device=weight.device))
        full = F.linear(hidden.view(1, 1, -1), weight).reshape(-1)
        partial = F.linear(hidden.view(1, 1, -1), masked).reshape(-1)
        checks += 1
        mismatches += int(not torch.equal(full[keep], partial[keep]))
    store.begin_run()
    return {"trials": checks, "mismatches": mismatches}


class RefinementLMHead:
    """Certified next token from the exact LM-head input, reading as few weight bytes as the certificate allows."""

    def __init__(
        self,
        store: PackedRefinementStore,
        numerics: ReferenceNumerics,
        coarse: CoarseArithmetic,
        tie_break: TieBreak = TieBreak.LOWEST_INDEX,
        fallback: FallbackMode = FallbackMode.FULL,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        self_test_trials: int = 8,
    ) -> None:
        if numerics.reduction_length != store.in_features or coarse.reduction_length != store.in_features:
            raise ValueError("reduction lengths must equal the store's in_features")
        if numerics.output_dtype != store.dtype:
            raise ValueError("the reference output grid is the weight dtype")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        self.store = store
        self.numerics = numerics
        self.coarse = coarse
        self.tie_break = tie_break
        self.fallback = fallback
        self.chunk_rows = chunk_rows
        self._layouts = [
            CodeLayout(store.level_bits(level), store.in_features, store.device) for level in range(store.num_levels)
        ]
        self.self_test: dict[str, int] | None = None
        if fallback is FallbackMode.MASKED:
            self.self_test = masked_fallback_self_test(store, self_test_trials)
            if self.self_test["mismatches"]:
                raise RuntimeError(f"masked fallback failed its platform self-test: {self.self_test}")

    def _vector(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dtype != self.store.dtype:
            raise TypeError(f"hidden dtype {hidden.dtype} does not match weight dtype {self.store.dtype}")
        if hidden.numel() != self.store.in_features:
            raise ValueError("hidden state must be a single position")
        return hidden.reshape(-1).contiguous()

    def run(self, hidden: torch.Tensor, timer: StageTimer | None = None, trace: bool = False) -> RefinementRunResult:
        """`hidden` is the exact LM-head input of one position (any shape with in_features elements)."""
        return _Run(self, hidden, timer or NullTimer(), trace).execute()


class _Run:
    """State of one `RefinementLMHead.run` call."""

    def __init__(self, head: RefinementLMHead, hidden: torch.Tensor, timer, trace: bool) -> None:
        self.head = head
        self.store = head.store
        self.numerics = head.numerics
        self.timer = timer
        self.trace = trace
        self.vector = head._vector(hidden)
        rows = self.store.out_features
        self.rows_loaded = [0] * self.store.num_states
        self.contender_counts: list[int] = []
        self.states: list[StateBounds] = []
        self.lower = self.upper = None
        self.alive: torch.Tensor | None = None
        self.exact_rows: torch.Tensor | None = None
        self.exact_weight: torch.Tensor | None = None
        self.coarse_sum = self.coarse_error = None
        self.running_center = torch.zeros(rows, dtype=torch.float64, device=self.store.device)
        self.running_mass = torch.zeros_like(self.running_center)

    def execute(self) -> RefinementRunResult:
        self.store.begin_run()
        self.timer.start()
        if not bool(torch.isfinite(self.vector).all()):
            return self._fallback("nonfinite_input", certificate=None)
        self.inputs = input_norms(self.vector)
        self.vector64 = self.vector.to(torch.float64)
        self.timer.mark("input")
        if not self._coarse_state():
            return self._fallback("coarse_overflow", certificate=None)
        state = 0
        while True:
            certificate = certify_columns(self.lower, self.upper, self.head.tie_break)
            self.alive = contenders(self.lower, self.upper, self.head.tie_break)
            certified = bool(certificate.certified)
            self.contender_counts.append(int(self.alive.sum()))
            self.timer.mark(f"{state}:certify")
            if certified:
                return self._result(certificate, decision_state=state)
            if state == self.store.exact_state:
                return self._fallback("uncertified", certificate=certificate)
            state += 1
            rows = self.alive.nonzero().squeeze(1)
            if state == self.store.exact_state:
                center, radius = self._exact_state(rows)
            else:
                center, radius = self._refinement_state(state, rows)
            self._tighten(state, rows, center, radius)
            self.timer.mark(f"{state}:{'exact' if state == self.store.exact_state else 'refine'}")

    # States

    def _coarse_state(self) -> bool:
        payload, scales = self.store.read_level(0)
        layout = self.head._layouts[0]
        coarse_sum = coarse_matvec(payload, layout, self.vector.to(torch.float32), self.head.chunk_rows)
        self.timer.mark("0:matvec")
        if not bool(torch.isfinite(coarse_sum).all()):
            return False
        self.coarse_payload, self.coarse_scales = payload, scales
        center = coarse_sum.to(torch.float64) * scales.to(torch.float64)  # exact: 24 × 24 bits
        mass = coded_mass_bound(scales, layout.limit, self.inputs.l1)
        error = coarse_error_bound(scales, mass, layout.limit, self.head.coarse)
        miss = remainder_mass_bound(self.store.remainder_norms(0), self.inputs)
        radius = coarse_radius(mass, miss, error, self.numerics)
        self.lower, self.upper = reference_logit_interval(center, radius, self.numerics.output_dtype)
        self.rows_loaded[0] = self.store.out_features
        if self.trace:
            self.coarse_sum, self.coarse_error = coarse_sum, error
            # Later states tighten self.lower/upper in place; keep this state's own interval.
            self.states.append(StateBounds(0, None, center, radius, self.lower.clone(), self.upper.clone()))
        self.timer.mark("0:bounds")
        return True

    def _level_values(self, level: int, payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return dequantize_rows(self.head._layouts[level].unpack(payload), scales)

    def _refinement_state(self, state: int, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Levels 0..state for `rows`, accumulated in float64 as in the oracle."""
        if state == 1:
            # The base level of every row is already in hand: recompute these rows exactly.
            base = self._level_values(
                0, self.coarse_payload.index_select(0, rows), self.coarse_scales.index_select(0, rows)
            )
            self.running_center[rows] = base @ self.vector64
            self.running_mass[rows] = absolute_mass_upper(base, self.vector64)
        payload, scales = self.store.read_level(state, rows)
        values = self._level_values(state, payload, scales)
        center = self.running_center[rows] + values @ self.vector64
        mass = self.running_mass[rows] + absolute_mass_upper(values, self.vector64)
        self.running_center[rows], self.running_mass[rows] = center, mass
        norms = self.store.remainder_norms(state)
        miss = remainder_mass_bound(RowNormBounds(norms.l2[rows], norms.linf[rows]), self.inputs)
        reference_mass = mass + miss
        return center, remainder_radius(miss, reference_mass, reference_mass, self.numerics)

    def _exact_state(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.store.read_exact(rows)
        self.exact_rows, self.exact_weight = rows, weight
        values = weight.to(torch.float64)
        return values @ self.vector64, exact_radius(absolute_mass_upper(values, self.vector64), self.numerics)

    def _tighten(self, state: int, rows: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> None:
        lower, upper = reference_logit_interval(center, radius, self.numerics.output_dtype)
        self.lower[rows] = torch.maximum(self.lower[rows], lower)
        self.upper[rows] = torch.minimum(self.upper[rows], upper)
        self.rows_loaded[state] = rows.numel()
        if self.trace:
            self.states.append(StateBounds(state, rows, center, radius, lower, upper))

    # Outcomes

    def _fallback(self, reason: str, certificate) -> RefinementRunResult:
        reference_input = self.vector.view(1, 1, -1)
        guard_tripped = False
        if self.head.fallback is FallbackMode.MASKED and reason == "uncertified":
            token, logits = self._masked(reference_input)
            if token is not None:
                self.timer.mark("fallback")
                return self._result(certificate, None, FallbackMode.MASKED, reason, False, token, logits)
            guard_tripped = True
        token, logits = self._full(reference_input)
        self.timer.mark("fallback")
        return self._result(certificate, None, FallbackMode.FULL, reason, guard_tripped, token, logits)

    def _masked(self, reference_input: torch.Tensor) -> tuple[int | None, torch.Tensor | None]:
        """Full-shape GEMM on the surviving rows only; None if a survivor leaves its certified interval."""
        keep = self.alive[self.exact_rows]
        if int(keep.sum()) != int(self.alive.sum()):
            return None, None  # a survivor without its original row: impossible, but never guess
        rows, weight = self.exact_rows[keep], self.exact_weight[keep]
        masked = torch.zeros(
            self.store.out_features, self.store.in_features, dtype=weight.dtype, device=weight.device
        )
        masked.index_copy_(0, rows, weight)
        logits = F.linear(reference_input, masked).reshape(-1)
        survivors = logits[rows].to(torch.float64)
        if not bool(((survivors >= self.lower[rows]) & (survivors <= self.upper[rows])).all()):
            return None, logits
        candidates = torch.where(self.alive, logits, torch.full_like(logits, -math.inf))
        return int(torch.argmax(candidates)), logits

    def _full(self, reference_input: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Read every original row not read yet and apply the reference operation (decision 0002)."""
        rows = self.store.out_features
        unread = torch.ones(rows, dtype=torch.bool, device=self.store.device)
        if self.exact_rows is not None:
            unread[self.exact_rows] = False
        missing = unread.nonzero().squeeze(1)
        weight = torch.empty(rows, self.store.in_features, dtype=self.store.dtype, device=self.store.device)
        weight.index_copy_(0, missing, self.store.read_fallback(missing))
        if self.exact_rows is not None:
            weight.index_copy_(0, self.exact_rows, self.exact_weight)
        logits = F.linear(reference_input, weight).reshape(-1)
        return int(torch.argmax(logits)), logits

    def _result(
        self,
        certificate,
        decision_state: int | None,
        fallback: FallbackMode | None = None,
        reason: str | None = None,
        guard_tripped: bool = False,
        token: int | None = None,
        logits: torch.Tensor | None = None,
    ) -> RefinementRunResult:
        if certificate is not None:
            winner, competitor = int(certificate.winner), int(certificate.competitor)
            margin = float(certificate.margin)
        else:
            winner, competitor, margin = -1, -1, -math.inf
        run_trace = None
        if self.trace and self.lower is not None:
            run_trace = RunTrace(
                self.coarse_sum, self.coarse_error, tuple(self.states), self.lower, self.upper, self.alive
            )
        return RefinementRunResult(
            token_id=winner if fallback is None else token,
            certified=fallback is None,
            fallback=fallback,
            fallback_reason=reason,
            masked_guard_tripped=guard_tripped,
            decision_state=decision_state,
            winner=winner,
            competitor=competitor,
            certificate_margin=margin,
            contenders=tuple(self.contender_counts),
            rows_loaded=tuple(self.rows_loaded),
            bytes=self.store.bytes_read(),
            reads=self.store.reads,
            fallback_logits=logits,
            timings_ms=dict(self.timer.stages),
            trace=run_trace,
        )
