"""OLMoE adapter (Phase 4A, decision 0007): allenai/OLMoE-1B-7B served with its experts out of device memory.

Only what is OLMoE-specific lives here; storage, caching, streaming and the compact experts
call are the generic MoE path (`awpmi.models.moe`, `awpmi.models.checkpoint`).

  * Checkpoint layout. The published checkpoint stores each expert's projections as separate
    tensors, model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight. The transformers loader
    stacks them and fuses gate with up: gate_up_proj [64, 2·1024, 2048] (gate rows first) and
    down_proj [64, 2048, 1024]. `EXPERT_LAYOUT` states that layout; `expert_sources` derives it
    from the transformers conversion mapping and requires the two to agree, so a change on
    either side is caught instead of silently streaming the wrong bytes.
  * Routers. Layer L's router is model.layers.L.mlp.gate: a softmax over the 64 experts in
    float32, the top 8, no renormalization, scores cast back to the activations' dtype. It is
    called before the experts and returns (router_logits, scores, indices); `routers` exposes
    it so that routing can be recorded where it is decided.
  * Reference profile. BF16 weights exactly as published (the 0924 checkpoint is stored in
    BF16; the 0125 base model is stored in FP32 and would need conversion), grouped_mm experts
    and SDPA attention, transformers' defaults.
"""

from __future__ import annotations

from torch import nn

from awpmi.models import checkpoint
from awpmi.models.moe import ExpertModule, find_expert_modules
from awpmi.profiles import BF16_REFERENCE

EXPERT_LAYOUT = {
    "gate_up_proj": ("mlp.experts.{expert}.gate_proj.weight", "mlp.experts.{expert}.up_proj.weight"),
    "down_proj": ("mlp.experts.{expert}.down_proj.weight",),
}
EXPERTS_SUFFIX = "mlp.experts"
ROUTER_SUFFIX = "mlp.gate"
REFERENCE_PROFILE = BF16_REFERENCE.with_kernels(experts="grouped_mm", attention="sdpa")


def expert_sources(model: nn.Module, modules: list[ExpertModule] | None = None) -> dict[str, dict[str, tuple[str, ...]]]:
    """The checkpoint tensors of every expert-sliced parameter, as transformers declares them, checked against OLMoE's layout."""
    return checkpoint.checked_expert_sources(model, EXPERT_LAYOUT, EXPERTS_SUFFIX, modules)


def routers(model: nn.Module) -> dict[str, nn.Module]:
    """Experts module name → the router that chooses for it."""
    modules = find_expert_modules(model)
    found = checkpoint.neighbours(model, EXPERTS_SUFFIX, ROUTER_SUFFIX, modules)
    for entry in modules:
        router = found[entry.name]
        if getattr(router, "num_experts", None) != entry.num_experts or not hasattr(router, "top_k"):
            raise ValueError(f"{entry.name}: no router next to it")
    return found
