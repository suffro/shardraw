"""The fully materialized reference: the unmodified Hugging Face forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel


@dataclass(frozen=True)
class ReferenceResult:
    token_id: int
    logits: torch.Tensor
    hidden_state: torch.Tensor


class ReferenceRunner:
    """Greedy next-token reference: token_id = argmax(reference_logits)."""

    def __init__(self, model: PreTrainedModel) -> None:
        if model.training:
            raise ValueError("reference model must be in eval() mode")
        self.model = model

    @torch.inference_mode()
    def next_token(self, input_ids: torch.Tensor) -> ReferenceResult:
        """Run the model's own forward (as generate() does for one step) and capture the LM-head input."""
        captured: list[torch.Tensor] = []
        hook = self.model.get_output_embeddings().register_forward_hook(
            lambda module, inputs, output: captured.append(inputs[0].detach().clone())
        )
        try:
            output = self.model(input_ids=input_ids, logits_to_keep=1, use_cache=False)
        finally:
            hook.remove()
        logits = output.logits[0, -1]
        return ReferenceResult(
            token_id=int(torch.argmax(logits, dim=-1).item()),
            logits=logits,
            hidden_state=captured[0],
        )
