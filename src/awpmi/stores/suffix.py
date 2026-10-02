"""Neuron-major store of an MLP in the adaptive suffix (Phase 2): every read counted.

Neuron i of a SwiGLU MLP owns gate_proj row i, up_proj row i and down_proj column i,
each `hidden` values. The store keeps the down projection transposed, so all three are
one contiguous run of hidden·itemsize bytes: the page of roadmap §2.7 is one role of one
neuron. It also keeps the resident bound metadata of decision 0005, all float32 and
rounded up:

  down column norms   C_i ≥ ‖W_d[:, i]‖₂      unread neurons in the pairwise certificate
  down row norms      ‖W_d[k, :]‖₂, ‖W_d[k, :]‖_∞   unread neurons in each output's interval
  up row norms        ‖W_u[i, :]‖₂            |u_i| of a neuron whose up row is unread

Only the roles listed as adaptive are served (and counted); the others belong to the
exact prefix. A run's effective bytes are the read log plus the resident metadata, as
for `PackedRefinementStore`. Everything is resident: a read is a gather (Phase 3 makes
it physical).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from awpmi.bounds.floating import round_up_to_grid
from awpmi.bounds.linear import block_l2_norm_upper
from awpmi.paging.page import WeightPage
from awpmi.tracing import tensor_digest

ROLES = ("gate", "up", "down")
METADATA_DTYPE = torch.float32


@dataclass(frozen=True)
class SuffixRead:
    role: str
    neurons: torch.Tensor
    count: int
    bytes: int


def _row_l2(matrix: torch.Tensor) -> torch.Tensor:
    return block_l2_norm_upper(matrix, [slice(0, matrix.shape[1])], storage_dtype=METADATA_DTYPE)[:, 0].to(METADATA_DTYPE)


class MLPStore:
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        adaptive: Sequence[str],
        layer: int,
        prefix: str,
    ) -> None:
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError("expected gate/up [intermediate, hidden] and down [hidden, intermediate]")
        if not adaptive or any(role not in ROLES for role in adaptive):
            raise ValueError(f"adaptive roles must be among {ROLES}")
        self.adaptive = tuple(role for role in ROLES if role in adaptive)
        self.layer = layer
        self.prefix = prefix
        self._pages = {"gate": gate, "up": up, "down": down.t().contiguous()}
        self._down = down
        down64 = down.to(torch.float64)
        self.down_column_norms = _row_l2(down64.t())
        self.down_row_l2 = _row_l2(down64)
        self.down_row_linf = round_up_to_grid(down64.abs().amax(dim=1), METADATA_DTYPE).to(METADATA_DTYPE)
        self.up_row_norms = _row_l2(up.to(torch.float64))
        self._log: list[SuffixRead] = []

    @classmethod
    def from_mlp(cls, mlp: torch.nn.Module, adaptive: Sequence[str], layer: int, prefix: str) -> MLPStore:
        return cls(
            mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach(), adaptive, layer, prefix
        )

    # Structure

    @property
    def neurons(self) -> int:
        return self._pages["gate"].shape[0]

    @property
    def hidden(self) -> int:
        return self._pages["gate"].shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self._pages["gate"].dtype

    @property
    def device(self) -> torch.device:
        return self._pages["gate"].device

    @property
    def page_bytes(self) -> int:
        return self.hidden * self._pages["gate"].element_size()

    def parameter_name(self, role: str) -> str:
        return f"{self.prefix}.{role}_proj.weight"

    @property
    def metadata_bytes(self) -> int:
        """Resident metadata of the adaptive roles (down norms if down is adaptive, up norms if up is)."""
        nbytes = 0
        if "down" in self.adaptive:
            nbytes += sum(t.numel() * t.element_size() for t in (self.down_column_norms, self.down_row_l2, self.down_row_linf))
        if "up" in self.adaptive:
            nbytes += self.up_row_norms.numel() * self.up_row_norms.element_size()
        return nbytes

    @property
    def weight_bytes(self) -> int:
        """BF16 bytes of the adaptive roles: the reference bytes the suffix may skip."""
        return len(self.adaptive) * self.neurons * self.page_bytes

    def pages(self) -> list[WeightPage]:
        """The page index of the adaptive roles: one page per role and neuron."""
        pages = []
        for role in self.adaptive:
            norms = {"down": self.down_column_norms, "up": self.up_row_norms}.get(role)
            for neuron in range(self.neurons):
                metadata = {} if norms is None else {"l2_norm_upper": float(norms[neuron])}
                shape = (self.hidden, 1) if role == "down" else (1, self.hidden)
                pages.append(
                    WeightPage(len(pages), self.parameter_name(role), neuron, shape, self.dtype, self.page_bytes, metadata, self.layer, f"{role}_proj")
                )
        return pages

    def content_digest(self) -> str:
        return tensor_digest(
            *[self._pages[role] for role in ROLES],
            self.down_column_norms,
            self.down_row_l2,
            self.down_row_linf,
            self.up_row_norms,
        )

    # Reads

    def begin_run(self) -> None:
        self._log = []

    @property
    def reads(self) -> tuple[SuffixRead, ...]:
        return tuple(self._log)

    def read(self, role: str, neurons: torch.Tensor) -> torch.Tensor:
        """Pages of `role` for `neurons`: gate/up rows, or down columns (as rows). Read-only."""
        if role not in self.adaptive:
            raise KeyError(f"{role} is not an adaptive role of this store")
        self._log.append(SuffixRead(role, neurons, neurons.numel(), neurons.numel() * self.page_bytes))
        return self._pages[role].index_select(0, neurons)

    def bytes_read(self) -> dict[str, int]:
        breakdown = {"metadata": self.metadata_bytes, **{role: 0 for role in ROLES}}
        for read in self._log:
            breakdown[read.role] += read.bytes
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    def neurons_read(self) -> dict[str, int]:
        counts = dict.fromkeys(ROLES, 0)
        for read in self._log:
            counts[read.role] += read.count
        return counts
