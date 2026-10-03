"""Phase 4B (decision 0008): the Moonlight adapter (layout, routers, shared experts, routing configuration)."""

from __future__ import annotations

import pytest
import torch
import transformers

from awpmi.models import moonlight
from awpmi.models.moe import find_expert_modules

MOONLIGHT = ("moonshotai/Moonlight-16B-A3B", "476b36a473d4467f94469414bef6cee75c9c8172")


def tiny_moonlight(seed: int = 0, **overrides) -> transformers.PreTrainedModel:
    """DeepSeek-V3 at a test size, with Moonlight's routing: one dense layer first, one group, top 6, two shared experts."""
    settings = dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        intermediate_size=96, moe_intermediate_size=32, n_routed_experts=16, num_experts_per_tok=6, n_shared_experts=2, n_group=1,
        topk_group=1, first_k_dense_replace=1, norm_topk_prob=True, routed_scaling_factor=2.446, q_lora_rank=None, kv_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, rope_scaling=None,
    )
    config = transformers.DeepseekV3Config(**{**settings, **overrides})
    torch.manual_seed(seed)
    return transformers.DeepseekV3ForCausalLM(config).to(torch.bfloat16).eval()


def moonlight_config():
    try:
        return transformers.AutoConfig.from_pretrained(MOONLIGHT[0], revision=MOONLIGHT[1], local_files_only=True)
    except Exception:  # not in the local cache
        pytest.skip("Moonlight's configuration is not in the local Hugging Face cache")


def test_the_adapter_checks_its_layout_against_transformers(monkeypatch):
    model = tiny_moonlight()
    modules = find_expert_modules(model)
    assert [m.name for m in modules] == ["model.layers.1.mlp.experts", "model.layers.2.mlp.experts"]  # layer 0 is dense
    sources = moonlight.expert_sources(model)
    assert sources["model.layers.2.mlp.experts"] == {
        "gate_up_proj": ("model.layers.2.mlp.experts.{expert}.gate_proj.weight", "model.layers.2.mlp.experts.{expert}.up_proj.weight"),
        "down_proj": ("model.layers.2.mlp.experts.{expert}.down_proj.weight",),
    }
    swapped = {**moonlight.EXPERT_LAYOUT, "gate_up_proj": moonlight.EXPERT_LAYOUT["gate_up_proj"][::-1]}
    monkeypatch.setattr(moonlight, "EXPERT_LAYOUT", swapped)
    with pytest.raises(ValueError):
        moonlight.expert_sources(model)


def test_routers_shared_experts_and_blocks_are_found_per_layer():
    model = tiny_moonlight(seed=1)
    routers, shared, blocks = moonlight.routers(model), moonlight.shared_experts(model), moonlight.moe_blocks(model)
    assert set(routers) == set(shared) == set(blocks) == {m.name for m in find_expert_modules(model)}
    for name, router in routers.items():
        assert router.e_score_correction_bias.dtype == torch.float32 or router.e_score_correction_bias.dtype == torch.bfloat16
        assert blocks[name].gate is router and blocks[name].shared_experts is shared[name]
        # The shared experts are a dense MLP of n_shared × moe_intermediate neurons, never an experts module.
        assert shared[name].gate_proj.weight.shape == (2 * 32, 64)
    assert not any("shared" in m.name for m in find_expert_modules(model))


def test_the_router_computes_deepseek_v3_routing():
    """transformers' router against the declared semantics, recomputed: sigmoid scores, choice on scores + bias, renormalized unbiased weights × scale."""
    model = tiny_moonlight(seed=2)
    router = moonlight.routers(model)["model.layers.1.mlp.experts"]
    generator = torch.Generator().manual_seed(2)
    router.e_score_correction_bias.copy_(torch.randn(16, generator=generator))
    hidden = torch.randn(40, 64, generator=generator).to(torch.bfloat16)
    with torch.inference_mode():
        logits, weights, indices = router(hidden)
    expected_logits = torch.nn.functional.linear(hidden.float(), router.weight.float())
    scores = expected_logits.sigmoid()
    chosen = torch.topk(scores + router.e_score_correction_bias, 6, dim=-1).indices
    expected = scores.gather(1, chosen)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 2.446
    assert logits.dtype == weights.dtype == torch.float32 and torch.equal(logits, expected_logits)
    order = indices.argsort(-1)
    assert torch.equal(indices.gather(1, order), chosen.sort(-1).values)
    assert torch.allclose(weights.gather(1, order), expected.gather(1, chosen.argsort(-1)), rtol=0, atol=1e-6)
    # The bias changes the choice, never the weights: weights are the unbiased scores of the chosen experts.
    assert torch.equal(weights.sum(-1).round(decimals=4), torch.full((40,), 2.446).round(decimals=4))


def test_the_routing_configuration_is_checked():
    config = moonlight_config()
    moonlight.check_config(config)
    assert config.model_type == "deepseek_v3" and config.tie_word_embeddings is False
    with pytest.raises(ValueError):
        moonlight.check_config(tiny_moonlight().config)  # 16 experts, not Moonlight's 64


def test_the_reference_profile_is_bf16_grouped_mm_sdpa():
    model = transformers.DeepseekV3ForCausalLM(tiny_moonlight().config).to(torch.bfloat16)
    moonlight.REFERENCE_PROFILE.check_model(model)
    assert moonlight.REFERENCE_PROFILE.weight_dtype == "BF16"
    with pytest.raises(ValueError):
        moonlight.REFERENCE_PROFILE.check_model(model.to(torch.float16))
