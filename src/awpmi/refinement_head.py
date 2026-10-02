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

  MASKED  the default (decision 0005). Apply F.linear(h, W') with the full shape, where
          W' keeps the surviving rows (already read in the exact state) and zeros
          elsewhere, and take the argmax over the survivors. It assumes that each
          output row of the reference GEMM depends only on its own weight row:
          verified by a mandatory self-test on the platform when the head is built,
          and guarded per call by checking that every surviving logit lies in its
          certified interval. If the self-test fails, the head runs FULL for every
          fallback; if the guard trips, that call runs FULL.
  FULL    read every original row not read yet and apply F.linear(h, W): the
          reference operation, bitwise (decision 0002). The canonical correctness
          fallback, always available.

Phase 2 (decision 0005) adds two things, without changing a run on an exact input:

  * `run_enclosure`: the same states for an *enclosure* of h (an adaptive suffix only
    bounds it). With centre c and radius ρ, every state adds the input's spread |L_j|·ρ
    to its radius and bounds absolute masses with |c| + ρ. The coarse pass computes
    S = q·c and T = |q|·ρ in binary32 with the same error model. A pass that stays
    UNKNOWN after the exact state ends there: no LM-head fallback runs on an enclosure.
  * `HeldReads`: one token's passes share what they read, so a row is read (and
    counted) once per token even when a later pass on the exact h needs it again.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from awpmi.bounds.coarse import CoarseArithmetic, coarse_error_bound, coarse_radius, coded_mass_bound
from awpmi.bounds.enclosure import Enclosure
from awpmi.bounds.floating import next_up, round_up_to_grid
from awpmi.bounds.linear import absolute_mass_upper, l1_norm_upper
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


@dataclass(frozen=True)
class EnclosureRunResult:
    """One pass on an enclosure of h: certified, or what is left after the exact state."""

    certified: bool
    token_id: int  # the certified winner, or -1
    reason: str | None  # None, "uncertified" (after the exact state), "too_wide" (a row limit) or "coarse_overflow"
    decision_state: int | None
    winner: int
    competitor: int
    certificate_margin: float
    contenders: tuple[int, ...]
    rows_loaded: tuple[int, ...]
    alive: torch.Tensor | None  # rows that may still be the reference argmax
    lower: torch.Tensor | None
    upper: torch.Tensor | None
    exact_rows: torch.Tensor | None  # rows read into the exact state by this pass, and their original values
    exact_weight: torch.Tensor | None
    trace: RunTrace | None


class _RowCache:
    """Rows of one state that this token has already read, with their data."""

    def __init__(self, rows: int, device: torch.device) -> None:
        self.position = torch.full((rows,), -1, dtype=torch.long, device=device)
        self.data: tuple[torch.Tensor, ...] | None = None

    @property
    def held(self) -> torch.Tensor:
        return self.position >= 0

    def missing(self, rows: torch.Tensor) -> torch.Tensor:
        return rows[self.position[rows] < 0]

    def add(self, rows: torch.Tensor, *tensors: torch.Tensor) -> None:
        start = 0 if self.data is None else self.data[0].shape[0]
        self.position[rows] = torch.arange(start, start + rows.numel(), device=rows.device)
        self.data = tensors if self.data is None else tuple(torch.cat([old, new]) for old, new in zip(self.data, tensors))

    def gather(self, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        index = self.position[rows]
        return tuple(tensor.index_select(0, index) for tensor in self.data)


class HeldReads:
    """What one token has read from the store so far, shared by its passes."""

    def __init__(self, store: PackedRefinementStore) -> None:
        self.base: tuple[torch.Tensor, torch.Tensor] | None = None
        self.levels = {level: _RowCache(store.out_features, store.device) for level in range(1, store.num_levels)}
        self.exact = _RowCache(store.out_features, store.device)


def coarse_matvec(
    payload: torch.Tensor,
    layout: CodeLayout,
    vector: torch.Tensor,
    chunk_rows: int,
    absolute_vector: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """fl₃₂(Σ_k q_jk·v_k) for every packed row, decoding `chunk_rows` rows at a time (float32 [V]).

    With `absolute_vector` ρ, also fl₃₂(Σ_k |q_jk|·ρ_k) from the same decoded chunk.
    The error model is `awpmi.bounds.coarse`; only binary32 arithmetic is used.
    """
    if vector.dtype != torch.float32 or (absolute_vector is not None and absolute_vector.dtype != torch.float32):
        raise TypeError("the coarse pass runs in float32")
    out = torch.empty(payload.shape[0], dtype=torch.float32, device=payload.device)
    spread = None if absolute_vector is None else torch.empty_like(out)
    for start in range(0, payload.shape[0], chunk_rows):
        codes = layout.unpack(payload[start : start + chunk_rows]).to(torch.float32)
        out[start : start + chunk_rows] = torch.mv(codes, vector)
        if spread is not None:
            spread[start : start + chunk_rows] = torch.mv(codes.abs(), absolute_vector)
    return out if spread is None else (out, spread)


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
        fallback: FallbackMode = FallbackMode.MASKED,
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
        # Whether a fallback may try MASKED first: requested, and the platform passed the self-test.
        self.masked_enabled = False
        if fallback is FallbackMode.MASKED:
            if self_test_trials <= 0:
                raise ValueError("the masked fallback needs its platform self-test")
            self.self_test = masked_fallback_self_test(store, self_test_trials)
            self.masked_enabled = self.self_test["mismatches"] == 0

    def _vector(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dtype != self.store.dtype:
            raise TypeError(f"hidden dtype {hidden.dtype} does not match weight dtype {self.store.dtype}")
        if hidden.numel() != self.store.in_features:
            raise ValueError("hidden state must be a single position")
        return hidden.reshape(-1).contiguous()

    def begin_token(self) -> HeldReads:
        """Start one token's read log, for passes that share it (`run(..., held=)`, `run_enclosure`)."""
        self.store.begin_run()
        return HeldReads(self.store)

    def run(
        self,
        hidden: torch.Tensor,
        timer: StageTimer | None = None,
        trace: bool = False,
        held: HeldReads | None = None,
    ) -> RefinementRunResult:
        """`hidden` is the exact LM-head input of one position (any shape with in_features elements).

        Without `held` this is one complete token (Phase 1C). With it, the pass continues a
        token started by `begin_token`: rows already held are not read again, and the
        store's log and the timer are not reset.
        """
        return _Run(self, hidden, timer or NullTimer(), trace, held).execute()

    def run_enclosure(
        self,
        enclosure: Enclosure,
        held: HeldReads,
        timer: StageTimer | None = None,
        trace: bool = False,
        row_limits: tuple[int, ...] | None = None,
    ) -> EnclosureRunResult:
        """The same states for an enclosure of h; stops UNKNOWN after the exact state (no fallback).

        `row_limits[s - 1]` caps the rows that may enter state s ≥ 1: a wider pass stops
        before reading them ("too_wide"). The limits save reads: they can turn a
        certificate into a fallback, never into a wrong token.
        """
        if enclosure.lower.numel() != self.store.in_features:
            raise ValueError("the enclosure must cover one position of the hidden dimension")
        if row_limits is not None and len(row_limits) != self.store.num_states - 1:
            raise ValueError("one row limit per state after the coarse one")
        run = _Run(self, enclosure, timer or NullTimer(), trace, held)
        run.row_limits = row_limits
        return run.execute_enclosure()


class _Run:
    """State of one pass (`RefinementLMHead.run` or `run_enclosure`)."""

    def __init__(self, head: RefinementLMHead, source, timer, trace: bool, held: HeldReads | None) -> None:
        self.head = head
        self.store = head.store
        self.numerics = head.numerics
        self.timer = timer
        self.trace = trace
        self.fresh = held is None
        self.held = held if held is not None else HeldReads(head.store)
        if isinstance(source, Enclosure):
            self.enclosure, self.vector = source, None
        else:
            self.enclosure, self.vector = None, head._vector(source)
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
        self.running_spread = torch.zeros_like(self.running_center) if self.enclosure is not None else None
        self.row_limits: tuple[int, ...] | None = None
        self.too_wide = False

    # Passes

    def execute(self) -> RefinementRunResult:
        if self.fresh:
            self.store.begin_run()
            self.timer.start()
        if not bool(torch.isfinite(self.vector).all()):
            return self._fallback("nonfinite_input", certificate=None)
        self.inputs = input_norms(self.vector)
        self.vector64 = self.vector.to(torch.float64)
        self.timer.mark("input")
        if not self._coarse_state():
            return self._fallback("coarse_overflow", certificate=None)
        certificate, state = self._refine()
        if certificate is not None:
            return self._result(certificate, decision_state=state)
        return self._fallback("uncertified", certificate=self._last_certificate)

    def execute_enclosure(self) -> EnclosureRunResult:
        enclosure = self.enclosure
        if not enclosure.finite:
            return self._enclosure_result(None, None, "coarse_overflow")
        self.center32 = enclosure.center.to(torch.float32)
        self.vector64 = self.center32.to(torch.float64)
        self.rho = enclosure.radius_about(self.vector64)
        self.magnitude = next_up(self.vector64.abs() + self.rho)
        self.inputs = input_norms(self.magnitude)
        self.timer.mark("input")
        if not self._coarse_state():
            return self._enclosure_result(None, None, "coarse_overflow")
        certificate, state = self._refine()
        if certificate is not None:
            return self._enclosure_result(certificate, state, None)
        return self._enclosure_result(self._last_certificate, None, "too_wide" if self.too_wide else "uncertified")

    def _refine(self):
        """Certificate → elimination → next state, up to the exact state. Returns (certificate, state) if certified."""
        state = 0
        while True:
            certificate = certify_columns(self.lower, self.upper, self.head.tie_break)
            self.alive = contenders(self.lower, self.upper, self.head.tie_break)
            self.contender_counts.append(int(self.alive.sum()))
            self.timer.mark(f"{state}:certify")
            if bool(certificate.certified):
                return certificate, state
            if state == self.store.exact_state:
                self._last_certificate = certificate
                return None, state
            state += 1
            rows = self.alive.nonzero().squeeze(1)
            if self.row_limits is not None and rows.numel() > self.row_limits[state - 1]:
                self._last_certificate, self.too_wide = certificate, True
                return None, state - 1
            if state == self.store.exact_state:
                center, radius = self._exact_state(rows)
            else:
                center, radius = self._refinement_state(state, rows)
            self._tighten(state, rows, center, radius)
            self.timer.mark(f"{state}:{'exact' if state == self.store.exact_state else 'refine'}")

    # Reads (each row of each state at most once per token)

    def _base(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.held.base is None:
            self.held.base = self.store.read_level(0)
        return self.held.base

    def _level(self, level: int, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.held.levels[level]
        missing = cache.missing(rows)
        if missing.numel():
            cache.add(missing, *self.store.read_level(level, missing))
        return cache.gather(rows)

    def _exact(self, rows: torch.Tensor) -> torch.Tensor:
        cache = self.held.exact
        missing = cache.missing(rows)
        if missing.numel():
            cache.add(missing, self.store.read_exact(missing))
        return cache.gather(rows)[0]

    # States

    def _coarse_state(self) -> bool:
        payload, scales = self._base()
        self.timer.mark("0:read")
        layout = self.head._layouts[0]
        if self.enclosure is None:
            coarse_sum = coarse_matvec(payload, layout, self.vector.to(torch.float32), self.head.chunk_rows)
            spread_sum = None
        else:
            rho32 = round_up_to_grid(self.rho, torch.float32).to(torch.float32)
            coarse_sum, spread_sum = coarse_matvec(payload, layout, self.center32, self.head.chunk_rows, rho32)
        self.timer.mark("0:matvec")
        if not bool(torch.isfinite(coarse_sum).all()) or (spread_sum is not None and not bool(torch.isfinite(spread_sum).all())):
            return False
        self.coarse_payload, self.coarse_scales = payload, scales
        center = coarse_sum.to(torch.float64) * scales.to(torch.float64)  # exact: 24 × 24 bits
        mass = coded_mass_bound(scales, layout.limit, self.inputs.l1)
        if spread_sum is None:
            error = coarse_error_bound(scales, mass, layout.limit, self.head.coarse)
        else:
            # The binary32 error of S = q·c uses the centre's mass; T = |q|·ρ errs by γ·limit·‖ρ‖₁ + τ.
            center_mass = coded_mass_bound(scales, layout.limit, l1_norm_upper(self.center32))
            error = coarse_error_bound(scales, center_mass, layout.limit, self.head.coarse)
            slack = self.head.coarse.gamma * layout.limit * float(l1_norm_upper(rho32)) + self.head.coarse.underflow(
                layout.limit
            )
            spread = next_up(scales.to(torch.float64) * next_up(spread_sum.to(torch.float64) + slack))
            error = error + spread
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
        weights = self.vector64 if self.enclosure is None else self.magnitude
        if state == 1:
            # The base level of every row is already in hand: recompute these rows exactly.
            base = self._level_values(
                0, self.coarse_payload.index_select(0, rows), self.coarse_scales.index_select(0, rows)
            )
            self.running_center[rows] = base @ self.vector64
            self.running_mass[rows] = absolute_mass_upper(base, weights)
            if self.enclosure is not None:
                self.running_spread[rows] = absolute_mass_upper(base, self.rho)
        self.timer.mark(f"{state}:refine")
        payload, scales = self._level(state, rows)
        self.timer.mark(f"{state}:read")
        values = self._level_values(state, payload, scales)
        center = self.running_center[rows] + values @ self.vector64
        mass = self.running_mass[rows] + absolute_mass_upper(values, weights)
        self.running_center[rows], self.running_mass[rows] = center, mass
        norms = self.store.remainder_norms(state)
        miss = remainder_mass_bound(RowNormBounds(norms.l2[rows], norms.linf[rows]), self.inputs)
        reference_mass = mass + miss
        radius = remainder_radius(miss, reference_mass, reference_mass, self.numerics)
        if self.enclosure is not None:
            spread = self.running_spread[rows] + absolute_mass_upper(values, self.rho)
            self.running_spread[rows] = spread
            radius = radius + spread
        return center, radius

    def _exact_state(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.timer.mark(f"{self.store.exact_state}:exact")
        weight = self._exact(rows)
        self.timer.mark(f"{self.store.exact_state}:read")
        self.exact_rows, self.exact_weight = rows, weight
        values = weight.to(torch.float64)
        if self.enclosure is None:
            return values @ self.vector64, exact_radius(absolute_mass_upper(values, self.vector64), self.numerics)
        radius = absolute_mass_upper(values, self.rho) + exact_radius(absolute_mass_upper(values, self.magnitude), self.numerics)
        return values @ self.vector64, radius

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
        if self.head.masked_enabled and reason == "uncertified":
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
        cache = self.held.exact
        held = cache.held
        missing = (~held).nonzero().squeeze(1)
        weight = torch.empty(
            self.store.out_features, self.store.in_features, dtype=self.store.dtype, device=self.store.device
        )
        weight.index_copy_(0, missing, self.store.read_fallback(missing))
        self.timer.mark("fallback:read")
        if bool(held.any()):
            rows = held.nonzero().squeeze(1)
            weight.index_copy_(0, rows, cache.gather(rows)[0])
        logits = F.linear(reference_input, weight).reshape(-1)
        return int(torch.argmax(logits)), logits

    def _trace(self) -> RunTrace | None:
        if self.trace and self.lower is not None:
            return RunTrace(self.coarse_sum, self.coarse_error, tuple(self.states), self.lower, self.upper, self.alive)
        return None

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
            trace=self._trace(),
        )

    def _enclosure_result(self, certificate, decision_state: int | None, reason: str | None) -> EnclosureRunResult:
        certified = reason is None
        if certificate is not None:
            winner, competitor, margin = int(certificate.winner), int(certificate.competitor), float(certificate.margin)
        else:
            winner, competitor, margin = -1, -1, -math.inf
        return EnclosureRunResult(
            certified=certified,
            token_id=winner if certified else -1,
            reason=reason,
            decision_state=decision_state,
            winner=winner,
            competitor=competitor,
            certificate_margin=margin,
            contenders=tuple(self.contender_counts),
            rows_loaded=tuple(self.rows_loaded),
            alive=self.alive,
            lower=self.lower,
            upper=self.upper,
            exact_rows=self.exact_rows,
            exact_weight=self.exact_weight,
            trace=self._trace(),
        )
