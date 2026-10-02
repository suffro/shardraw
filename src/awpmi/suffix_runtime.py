"""Phase 2 runtime: an adaptive suffix inside the last MLP, ahead of the adaptive LM head (decision 0005).

Stages (adaptive suffix depth, roadmap §2.1):

  lm_head  depth 0: exact prefix through the final norm; the Phase 1C LM head
  down     depth 1, the final projection: the last layer's down_proj is adaptive
  mlp      depth 2, the last MLP: gate_proj, up_proj and down_proj are adaptive

For one token (stages down and mlp):

  prefix     the model's own forward, interrupted where the suffix begins
             (`awpmi.models.smollm2.suffix_prefix`): exact, bitwise the reference's
  suffix     read a budgeted share of the suffix's neuron pages (`MLPStore`), in a fixed
             order of decreasing bound contribution; propagate enclosures through the
             reference operations (`awpmi.bounds.operators`): [gate, up, SiLU, product,]
             down projection, residual addition, final RMSNorm → an enclosure of h
  LM head    the Phase 1C runtime on that enclosure (`RefinementLMHead.run_enclosure`); a
             pass that would read more rows into a state than `enclosure_row_limits`
             allows stops there (an efficiency rule: it can turn a certificate into a
             fallback, never into a wrong token)
  pairwise   if still UNKNOWN: the scale-free pairwise certificate (`awpmi.bounds.pairwise`)
             on the remaining contenders, whose exact LM-head rows are in hand
  fallback   still UNKNOWN: read the rest of the suffix and recompute it exactly, with the
             reference's own operations and shapes (all positions), which gives the
             reference's h bitwise; then the Phase 1C LM head on that exact h, sharing the
             reads of the first pass (its own fallback is MASKED, guarded, else FULL).

The certified runtime uses the faithful rounding model. With an experimental model
(round-to-nearest-even, decision 0005) the runtime only reports whether the certificate
*would* hold and with which token; it never sets `certified` and runs no fallback.

The last layer's attention is part of the exact prefix, so every layer's keys and values
are the reference's: the KV cache is exact by construction (roadmap §2.8, Mode A).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from awpmi.bounds.enclosure import Enclosure, UnboundedValue
from awpmi.bounds.floating import next_up
from awpmi.bounds.operators import RMSNormBounds, linear, multiply, residual_add, rms_norm, silu
from awpmi.bounds.pairwise import DownProjection, PairwiseResult, pairwise_certificate
from awpmi.bounds.residual import ReferenceNumerics
from awpmi.bounds.rounding import CERTIFIED, RoundingAssumptions, relative_rounding_error, subnormal_floor
from awpmi.models.smollm2 import SuffixPrefix
from awpmi.refinement_head import EnclosureRunResult, RefinementLMHead, RefinementRunResult
from awpmi.stores.suffix import MLPStore
from awpmi.tracing import NullTimer, StageTimer


class SuffixStage(enum.Enum):
    LM_HEAD = "lm_head"
    DOWN = "down"
    MLP = "mlp"

    @property
    def depth(self) -> int:
        return {"lm_head": 0, "down": 1, "mlp": 2}[self.value]

    @property
    def adaptive_roles(self) -> tuple[str, ...]:
        return {"lm_head": (), "down": ("down",), "mlp": ("gate", "up", "down")}[self.value]

    @property
    def boundary(self) -> str | None:
        return {"lm_head": None, "down": "down_proj", "mlp": "mlp"}[self.value]


@dataclass(frozen=True)
class SuffixBounds:
    """Every enclosure the suffix propagated, plus what the pairwise certificate needs."""

    read: torch.Tensor  # bool [I]: neurons whose adaptive pages were read
    gate: Enclosure | None
    activation_function: Enclosure | None  # SiLU(gate)
    up: Enclosure | None  # read neurons only
    activation: Enclosure  # a over [I]; unread neurons hold 0 (their columns are unread too)
    unread_activation: torch.Tensor  # |a_i| bound of unread neurons
    output: Enclosure  # o = down projection
    residual_sum: Enclosure  # y = r + o
    norm: RMSNormBounds  # final norm: q, n and h
    down: DownProjection


@dataclass(frozen=True)
class SuffixRunResult:
    stage: str
    budget: float
    assumptions: str
    token_id: int  # the decision (for an experimental run: the would-be token, or -1)
    certified: bool
    would_certify: bool  # what the certificate says under `assumptions`
    path: str  # "lm_head", "interval", "pairwise", "exact_suffix" or "none" (experimental, uncertified)
    reason: str | None  # why the suffix certificate did not hold
    enclosure: EnclosureRunResult | None
    pairwise: PairwiseResult | None
    head: RefinementRunResult | None  # the LM-head pass on the exact h (lm_head stage, or after the exact suffix)
    neurons_read: dict[str, int]  # adaptive pages read before any fallback
    suffix_bytes: dict[str, int]
    head_bytes: dict[str, int]
    bytes_total: int
    region_bytes: int  # BF16 bytes of the adaptive region (suffix roles + LM head)
    bounds: SuffixBounds | None
    exact_suffix: dict[str, torch.Tensor] | None  # o, y and h of the exact recomputation (all positions)
    timings_ms: dict[str, float]

    @property
    def fraction(self) -> float:
        return self.bytes_total / self.region_bytes


class AdaptiveSuffixRuntime:
    def __init__(
        self,
        stage: SuffixStage,
        head: RefinementLMHead,
        final_norm: torch.nn.Module,
        mlp_store: MLPStore | None = None,
        activation_function: torch.nn.Module | None = None,
        accumulation_unit_roundoff: float = 2.0**-22,
        budget: float = 1.0,
        assumptions: RoundingAssumptions = CERTIFIED,
        experimental: bool = False,
        enclosure_row_limits: tuple[int, ...] | None = None,
    ) -> None:
        if not assumptions.certified and not experimental:
            raise ValueError("only the faithful rounding model may certify; pass experimental=True for a what-if")
        if stage is not SuffixStage.LM_HEAD:
            if mlp_store is None or mlp_store.adaptive != stage.adaptive_roles:
                raise ValueError(f"stage {stage.value} needs an MLPStore with adaptive roles {stage.adaptive_roles}")
            if stage is SuffixStage.MLP and activation_function is None:
                raise ValueError("the mlp stage recomputes the activation: pass the model's act_fn")
        if not 0.0 < budget <= 1.0:
            raise ValueError("budget must be in (0, 1]")
        self.stage = stage
        self.head = head
        self.final_norm = final_norm
        self.store = mlp_store
        self.activation_function = activation_function
        self.unit_roundoff = accumulation_unit_roundoff
        self.budget = budget
        self.assumptions = assumptions
        self.experimental = experimental
        self.enclosure_row_limits = enclosure_row_limits
        self.dtype = head.store.dtype
        if mlp_store is not None:
            self.numerics_in = ReferenceNumerics(self.dtype, accumulation_unit_roundoff, mlp_store.hidden)
            self.numerics_down = ReferenceNumerics(self.dtype, accumulation_unit_roundoff, mlp_store.neurons)
        self.region_bytes = head.store.weight_bytes + (0 if mlp_store is None else mlp_store.weight_bytes)
        self.resident_bytes = 0 if mlp_store is None else final_norm.weight.numel() * final_norm.weight.element_size()

    # One token

    def run(self, prefix: SuffixPrefix | torch.Tensor, timer: StageTimer | None = None, trace: bool = False) -> SuffixRunResult:
        """`prefix`: the exact LM-head input (lm_head stage) or the `SuffixPrefix` of the stage's boundary."""
        timer = timer or NullTimer()
        if self.stage is SuffixStage.LM_HEAD:
            head = self.head.run(prefix, timer=timer, trace=trace)
            return self._result(head.token_id, head.certified, "lm_head", None, None, None, head, None, None, timer)
        timer.start()
        held = self.head.begin_token()
        self.store.begin_run()
        self._pages: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._neurons_before = None
        enclosure = pairwise = bounds = None
        reason = None
        try:
            bounds = self._bound_suffix(prefix)
            timer.mark("suffix:bounds")
        except UnboundedValue:
            reason = "unbounded"
        if bounds is not None:
            enclosure = self.head.run_enclosure(
                bounds.norm.output, held, timer=timer, trace=trace, row_limits=self.enclosure_row_limits
            )
            if enclosure.certified:
                return self._finish(enclosure.token_id, "interval", None, enclosure, None, None, bounds, None, timer)
            reason = enclosure.reason
            if enclosure.reason == "uncertified":
                pairwise = self._pairwise(enclosure, bounds)
                timer.mark("suffix:pairwise")
                if pairwise.certified:
                    return self._finish(pairwise.winner, "pairwise", None, enclosure, pairwise, None, bounds, None, timer)
        if self.experimental:
            return self._finish(-1, "none", reason, enclosure, pairwise, None, bounds, None, timer)
        self._neurons_before = self.store.neurons_read()
        exact = self._exact_suffix(prefix)
        timer.mark("suffix:exact")
        head = self.head.run(exact["h"][:, -1:, :], timer=timer, trace=trace, held=held)
        return self._finish(head.token_id, "exact_suffix", reason, enclosure, pairwise, head, bounds, exact, timer)

    # The suffix's bounds

    def _read(self, role: str, neurons: torch.Tensor) -> torch.Tensor:
        pages = self.store.read(role, neurons)
        if role in self._pages:
            held, data = self._pages[role]
            self._pages[role] = (torch.cat([held, neurons]), torch.cat([data, pages]))
        else:
            self._pages[role] = (neurons, pages)
        return pages

    def _schedule(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The budgeted neurons, largest bound contribution first (stable, deterministic); and the read mask."""
        count = min(score.numel(), math.ceil(self.budget * score.numel() - 1e-9))
        order = torch.argsort(score, descending=True, stable=True)
        chosen = torch.sort(order[:count]).values
        read = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        read[chosen] = True
        return chosen, read

    def _bound_suffix(self, prefix: SuffixPrefix) -> SuffixBounds:
        store, model = self.store, self.assumptions
        column_norms = store.down_column_norms.to(torch.float64)
        gate = activation_function = up = None
        if self.stage is SuffixStage.DOWN:
            a = prefix.activation[0, -1].to(torch.float64)
            activation_magnitude = a.abs()
            chosen, read = self._schedule(activation_magnitude * column_norms)
            activation = Enclosure.exact(a)
            unread_activation = activation_magnitude
        else:
            x = Enclosure.exact(prefix.mlp_input[0, -1])
            gate_rows = self._read("gate", torch.arange(store.neurons, device=store.device))
            gate = linear(x, gate_rows.to(torch.float64), self.numerics_in, model.gemm, label="gate")
            activation_function = silu(gate, self.dtype, model.elementwise)
            # |u_i| of an unread up row: Cauchy–Schwarz, the accumulation error, the GEMM's rounding;
            # then |a_i| = |rnd(s_i·u_i)|, an elementwise rounding.
            floor16 = subnormal_floor(self.dtype)
            gemm16 = relative_rounding_error(self.dtype, model.gemm)
            elementwise16 = relative_rounding_error(self.dtype, model.elementwise)
            x_norm = float(next_up(torch.linalg.vector_norm(x.upper) * (1.0 + 2.0**-40)))
            up_magnitude = store.up_row_norms.to(torch.float64) * x_norm * (1.0 + self.numerics_in.accumulation_gamma)
            up_magnitude = next_up(up_magnitude * (1.0 + gemm16) + floor16)
            unread_activation = next_up(activation_function.magnitude * up_magnitude * (1.0 + elementwise16) + floor16)
            chosen, read = self._schedule(unread_activation * column_norms)
            up_rows = self._read("up", chosen)
            up = linear(x, up_rows.to(torch.float64), self.numerics_in, model.gemm, label="up")
            product = multiply(activation_function.select(chosen), up, self.dtype, model.elementwise, label="activation")
            lower = torch.zeros(store.neurons, dtype=torch.float64, device=store.device)
            upper = torch.zeros_like(lower)
            lower[chosen], upper[chosen] = product.lower, product.upper
            activation = Enclosure(lower, upper, product.provenance)
        columns = self._read("down", chosen)
        read_weight = torch.zeros(store.hidden, store.neurons, dtype=torch.float64, device=store.device)
        read_weight[:, chosen] = columns.to(torch.float64).t()
        unread = torch.where(read, 0.0, unread_activation)
        row_l2, row_linf = store.down_row_l2.to(torch.float64), store.down_row_linf.to(torch.float64)
        unread_l2 = next_up(torch.linalg.vector_norm(unread) * (1.0 + 2.0**-40))
        unread_l1 = next_up(unread.sum() * (1.0 + 2.0**-40))
        unread_mass = next_up(torch.minimum(row_l2 * unread_l2, row_linf * unread_l1))
        has_unread = not bool(read.all())
        output = linear(
            activation, read_weight, self.numerics_down, model.gemm, unread_mass if has_unread else None, label="down"
        )
        if has_unread:
            output = output.tagged(*(f"missing:{store.parameter_name(role)}" for role in self.stage.adaptive_roles if role != "gate"))
        activation_magnitude = torch.where(read, activation.magnitude, 0.0)
        mass = next_up(read_weight.abs() @ activation_magnitude * (1.0 + 2.0**-40)) + (unread_mass if has_unread else 0.0)
        residual = prefix.residual[0, -1].to(torch.float64)
        residual_sum = residual_add(Enclosure.exact(residual), output, self.dtype, model.elementwise, label="residual")
        norm = rms_norm(
            residual_sum, self.final_norm.weight.detach(), self.final_norm.variance_epsilon, model.elementwise, self.unit_roundoff
        )
        down = DownProjection(
            residual=residual,
            read_weight=read_weight,
            read=read,
            activation=activation,
            unread_activation=unread_activation,
            column_norms=column_norms,
            accumulation_error=next_up(self.numerics_down.accumulation_gamma * mass),
            output=output,
        )
        return SuffixBounds(read, gate, activation_function, up, activation, unread_activation, output, residual_sum, norm, down)

    def _pairwise(self, enclosure: EnclosureRunResult, bounds: SuffixBounds) -> PairwiseResult:
        rows = enclosure.alive.nonzero().squeeze(1)
        positions = torch.searchsorted(enclosure.exact_rows, rows)
        return pairwise_certificate(
            winner=enclosure.winner,
            rows=rows,
            lm_weight=enclosure.exact_weight.index_select(0, positions),
            logit_lower=enclosure.lower[rows],
            logit_upper=enclosure.upper[rows],
            norm_weight=self.final_norm.weight.detach(),
            y=bounds.residual_sum,
            scale_lower=bounds.norm.scale_lower,
            lm_gamma=self.head.numerics.accumulation_gamma,
            assumptions=self.assumptions,
            down=bounds.down,
        )

    # The exact suffix (fallback)

    def _matrix(self, role: str) -> torch.Tensor:
        """The role's weight in its original layout, from every page read (reading the missing ones)."""
        store = self.store
        neurons = torch.arange(store.neurons, device=store.device)
        held = torch.zeros(store.neurons, dtype=torch.bool, device=store.device)
        if role in self._pages:
            held[self._pages[role][0]] = True
        missing = neurons[~held]
        if missing.numel():
            self._read(role, missing)
        index, data = self._pages[role]
        pages = torch.empty(store.neurons, store.hidden, dtype=store.dtype, device=store.device)
        pages.index_copy_(0, index, data)
        return pages.t().contiguous() if role == "down" else pages

    @torch.inference_mode()
    def _exact_suffix(self, prefix: SuffixPrefix) -> dict[str, torch.Tensor]:
        """The reference's own operations on all positions, as `LlamaMLP`, the decoder layer and `LlamaModel` run them."""
        if self.stage is SuffixStage.DOWN:
            activation = prefix.activation
        else:
            x = prefix.mlp_input
            activation = self.activation_function(F.linear(x, self._matrix("gate"))) * F.linear(x, self._matrix("up"))
        output = F.linear(activation, self._matrix("down"))
        residual_sum = prefix.residual + output
        return {"o": output, "y": residual_sum, "h": self.final_norm(residual_sum)}

    # Results

    def _finish(self, token, path, reason, enclosure, pairwise, head, bounds, exact, timer) -> SuffixRunResult:
        would_certify = path in ("interval", "pairwise")
        certified = would_certify and not self.experimental
        if head is not None:
            certified = head.certified
        return self._result(token, certified, path, reason, enclosure, pairwise, head, bounds, exact, timer, would_certify)

    def _result(
        self, token, certified, path, reason, enclosure, pairwise, head, bounds, exact, timer, would_certify=None
    ) -> SuffixRunResult:
        head_bytes = self.head.store.bytes_read()
        if self.store is None:
            suffix_bytes = {"total": 0}
            neurons = {}
        else:
            suffix_bytes = self.store.bytes_read()
            suffix_bytes["resident"] = self.resident_bytes
            suffix_bytes["total"] += self.resident_bytes
            neurons = self._neurons_before if self._neurons_before is not None else self.store.neurons_read()
            neurons = {role: neurons[role] for role in self.stage.adaptive_roles}
        return SuffixRunResult(
            stage=self.stage.value,
            budget=self.budget,
            assumptions=self.assumptions.name,
            token_id=token,
            certified=certified,
            would_certify=certified if would_certify is None else would_certify,
            path=path,
            reason=reason,
            enclosure=enclosure,
            pairwise=pairwise,
            head=head,
            neurons_read=neurons,
            suffix_bytes=suffix_bytes,
            head_bytes=head_bytes,
            bytes_total=suffix_bytes["total"] + head_bytes["total"],
            region_bytes=self.region_bytes,
            bounds=bounds,
            exact_suffix=exact,
            timings_ms=dict(timer.stages),
        )
