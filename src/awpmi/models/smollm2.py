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


def last_layer(model: PreTrainedModel) -> torch.nn.Module:
    return model.model.layers[-1]


class _Boundary(Exception):
    """Raised by a pre-hook to stop the reference forward where the adaptive suffix begins."""


@dataclass(frozen=True)
class SuffixPrefix:
    """The exact prefix's tensors at the start of an adaptive suffix in the last layer's MLP (all positions).

    `residual` is the input of the last post-attention RMSNorm (it is added back after the
    MLP); `mlp_input` is that norm's output, the MLP's input; `activation` is
    act(gate(x))·up(x), the down projection's input (None when the gate and up
    projections are adaptive too).
    """

    residual: torch.Tensor
    mlp_input: torch.Tensor | None
    activation: torch.Tensor | None


@torch.inference_mode()
def suffix_prefix(model: PreTrainedModel, input_ids: torch.Tensor, boundary: str, past_key_values=None) -> SuffixPrefix:
    """Run the model's own forward up to `boundary` ("down_proj" or "mlp") of the last layer, and stop there.

    The forward is the reference's (same modules, shapes and kernels), interrupted by a
    forward pre-hook before the boundary module runs, so no weight of the adaptive
    suffix is touched and every captured tensor equals the reference's bitwise.

    With `past_key_values` (a cache), the step uses and extends it as the reference's
    cached forward does. The boundary is after the last layer's attention, so every
    layer's keys and values are written by this exact prefix: the cache stays the
    reference's (roadmap §2.8, Mode A).
    """
    layer = last_layer(model)
    target = {"down_proj": layer.mlp.down_proj, "mlp": layer.mlp}[boundary]
    captured: dict[str, torch.Tensor] = {}

    def residual_hook(module, inputs):
        captured["residual"] = inputs[0]

    def mlp_input_hook(module, inputs):
        captured["mlp_input"] = inputs[0]

    def boundary_hook(module, inputs):
        captured["boundary"] = inputs[0]
        raise _Boundary

    hooks = [
        layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
        layer.mlp.register_forward_pre_hook(mlp_input_hook),
        target.register_forward_pre_hook(boundary_hook),
    ]
    try:
        model.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=past_key_values is not None)
    except _Boundary:
        pass
    else:
        raise RuntimeError("the forward did not reach the suffix boundary")
    finally:
        for hook in hooks:
            hook.remove()
    return SuffixPrefix(
        residual=captured["residual"],
        mlp_input=captured["mlp_input"],
        activation=captured["boundary"] if boundary == "down_proj" else None,
    )
