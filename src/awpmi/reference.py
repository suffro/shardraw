"""The fully materialized reference: the unmodified Hugging Face forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel

# Intermediates of the last MLP and the final norm (all positions), for validating Phase 2 bounds.
INTERMEDIATES = ("residual", "x", "g", "s", "u", "a", "o", "y", "h")


@dataclass(frozen=True)
class ReferenceResult:
    token_id: int
    logits: torch.Tensor
    hidden_state: torch.Tensor
    # With `intermediates=True`: the tensors named in INTERMEDIATES, plus "q" (the final norm's
    # rsqrt scale) and "n" (its normalized input), recomputed from "y" by the norm's own operations.
    intermediates: dict[str, torch.Tensor] | None = None


class ReferenceRunner:
    """Greedy next-token reference: token_id = argmax(reference_logits)."""

    def __init__(self, model: PreTrainedModel) -> None:
        if model.training:
            raise ValueError("reference model must be in eval() mode")
        self.model = model

    def _intermediate_hooks(self, captured: dict[str, torch.Tensor]) -> list:
        layer = self.model.model.layers[-1]
        mlp = layer.mlp

        def save(name: str, use_input: bool):
            def hook(module, inputs, output):
                captured[name] = (inputs[0] if use_input else output).detach().clone()

            return hook

        modules = [
            (layer.post_attention_layernorm, "residual", True),
            (mlp, "x", True),
            (mlp.gate_proj, "g", False),
            (mlp.act_fn, "s", False),
            (mlp.up_proj, "u", False),
            (mlp.down_proj, "a", True),
            (mlp.down_proj, "o", False),
            (self.model.model.norm, "y", True),
            (self.model.model.norm, "h", False),
        ]
        return [module.register_forward_hook(save(name, use_input)) for module, name, use_input in modules]

    @torch.inference_mode()
    def next_token(self, input_ids: torch.Tensor, intermediates: bool = False) -> ReferenceResult:
        """Run the model's own forward (as generate() does for one step) and capture the LM-head input.

        Hooks only observe: the forward and its arithmetic are the model's own.
        """
        captured: list[torch.Tensor] = []
        values: dict[str, torch.Tensor] = {}
        hooks = [
            self.model.get_output_embeddings().register_forward_hook(
                lambda module, inputs, output: captured.append(inputs[0].detach().clone())
            )
        ]
        if intermediates:
            hooks += self._intermediate_hooks(values)
        try:
            output = self.model(input_ids=input_ids, logits_to_keep=1, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        logits = output.logits[0, -1]
        if intermediates:
            norm = self.model.model.norm
            hidden = values["y"].to(torch.float32)
            scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
            values["q"], values["n"] = scale, (hidden * scale).to(values["y"].dtype)
        return ReferenceResult(
            token_id=int(torch.argmax(logits, dim=-1).item()),
            logits=logits,
            hidden_state=captured[0],
            intermediates=values if intermediates else None,
        )
