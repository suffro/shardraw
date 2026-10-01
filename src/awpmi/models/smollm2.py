"""SmolLM2 (Llama architecture) integration: loading, LM-head access, exact prefix."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

LM_HEAD_PARAMETER = "lm_head.weight"


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    revision: str
    dtype: torch.dtype
    device: torch.device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """`auto` means BF16 when the device supports it, otherwise FP32."""
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_model(spec: ModelSpec) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(spec.repository, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(spec.repository, revision=spec.revision, dtype=spec.dtype)
    model = model.to(spec.device).eval()
    lm_head = model.get_output_embeddings()
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        raise TypeError("AWPMI Phase 1 expects a bias-free nn.Linear LM head")
    if lm_head.weight.dtype != spec.dtype:
        raise TypeError(f"LM head dtype {lm_head.weight.dtype} != requested {spec.dtype}")
    return model, tokenizer


def lm_head_weight(model: PreTrainedModel) -> torch.Tensor:
    return model.get_output_embeddings().weight.detach()


@torch.inference_mode()
def final_hidden_state(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Exact transformer prefix: body + final norm, last position, shaped [1, 1, hidden]."""
    hidden_states = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    return hidden_states[:, -1:, :]
