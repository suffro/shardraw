from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from awpmi.models.smollm2 import final_hidden_state, lm_head_weight
from awpmi.reference import ReferenceRunner

pytestmark = pytest.mark.model


def test_reference_definition(model, prompt_ids):
    runner = ReferenceRunner(model)
    assert not model.training
    for ids in prompt_ids:
        result = runner.next_token(ids)
        assert result.logits.shape == (model.config.vocab_size,)
        assert result.logits.dtype == lm_head_weight(model).dtype
        assert result.token_id == int(torch.argmax(result.logits))
        assert result.hidden_state.shape == (1, 1, model.config.hidden_size)


def test_reference_matches_model_forward_and_generate(model, prompt_ids):
    """The hook-captured reference equals the model's own forward and a greedy generate() step."""
    runner = ReferenceRunner(model)
    for ids in prompt_ids:
        result = runner.next_token(ids)
        with torch.inference_mode():
            full = model(input_ids=ids).logits[0, -1]
            generated = model.generate(ids, max_new_tokens=1, do_sample=False, attention_mask=torch.ones_like(ids))
        assert int(torch.argmax(full)) == result.token_id
        assert int(generated[0, -1]) == result.token_id


def test_reference_is_deterministic(model, prompt_ids):
    runner = ReferenceRunner(model)
    for ids in prompt_ids:
        first, second = runner.next_token(ids), runner.next_token(ids)
        assert torch.equal(first.logits, second.logits)
        assert torch.equal(first.hidden_state, second.hidden_state)


def test_exact_prefix_reproduces_reference_hidden_state(model, prompt_ids):
    runner = ReferenceRunner(model)
    weight = lm_head_weight(model)
    for ids in prompt_ids:
        result = runner.next_token(ids)
        hidden = final_hidden_state(model, ids)
        assert torch.equal(hidden, result.hidden_state)
        # The reference LM head is exactly a bias-free linear map of that hidden state.
        assert torch.equal(F.linear(hidden, weight).reshape(-1), result.logits)


def test_reference_rejects_training_mode(model):
    model.train()
    try:
        with pytest.raises(ValueError):
            ReferenceRunner(model)
    finally:
        model.eval()
