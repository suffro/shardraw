"""Phase 2 runtime: an adaptive suffix in the last MLP, checked end to end against the reference.

A tiny random Llama (tied embeddings with near-duplicate rows, so that tokens are certified
by intervals, by the pairwise certificate, or not at all) runs the reference forward with
hooks on every intermediate of the last MLP and the final norm. For every stage, budget and
rounding model the runtime must give the reference token, enclose every intermediate, read
each page once, account for every byte, and reproduce the reference bitwise when it
recomputes the suffix.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import awpmi.bounds.pairwise as pairwise_module
from awpmi.bounds.coarse import CoarseArithmetic
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics
from awpmi.bounds.rounding import ASSUMPTIONS, CERTIFIED, EXPERIMENTAL
from awpmi.decomposition import RefinementDecomposition
from awpmi.models.smollm2 import last_layer, suffix_prefix
from awpmi.reference import ReferenceRunner
from awpmi.refinement_head import RefinementLMHead
from awpmi.stores.refinement import PackedRefinementStore
from awpmi.stores.suffix import MLPStore
from awpmi.suffix_runtime import AdaptiveSuffixRuntime, SuffixStage
from tests.conftest import DEVICES

STAGES = [SuffixStage.DOWN, SuffixStage.MLP]
ROW_LIMITS = (32768, 256)


def tiny_model(device: str, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).to(device).eval()
    generator = torch.Generator().manual_seed(seed + 5)
    with torch.no_grad():
        embedding = model.model.embed_tokens.weight
        embedding.mul_(8.0)
        twin_distance = 10 ** (-3 * torch.rand(256, 1, generator=generator))
        noise = torch.randn(256, 64, generator=generator) * twin_distance * float(embedding.float().abs().mean())
        embedding[1::2] = (embedding[0::2].float().cpu() + noise).to(embedding.dtype).to(device)
        for layer in model.model.layers:
            layer.mlp.down_proj.weight.mul_(30.0)
    return model


class Reference:
    """The reference forward (`ReferenceRunner`) with every intermediate of the last MLP and the final norm."""

    def __init__(self, model) -> None:
        self.runner = ReferenceRunner(model)

    def __call__(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.runner.next_token(ids, intermediates=True)
        values = dict(result.intermediates)
        values["q"], values["logits"] = values["q"][0, -1, 0], result.logits
        return values

    def __enter__(self) -> Reference:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def build(model, stage: SuffixStage, budget: float, assumptions=CERTIFIED, row_limits=ROW_LIMITS):
    weight = model.lm_head.weight.detach()
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    head = RefinementLMHead(
        PackedRefinementStore.from_decomposition(decomposition),
        ReferenceNumerics(weight.dtype, FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1]),
        CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1]),
    )
    layer = last_layer(model)
    index = len(model.model.layers) - 1
    store = MLPStore.from_mlp(layer.mlp, stage.adaptive_roles, index, f"model.layers.{index}.mlp")
    runtime = AdaptiveSuffixRuntime(
        stage,
        head,
        model.model.norm,
        store,
        layer.mlp.act_fn,
        budget=budget,
        assumptions=assumptions,
        experimental=not assumptions.certified,
        enclosure_row_limits=row_limits,
    )
    return runtime, decomposition


def last(values: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return values[name][0, -1].to(torch.float64)


def check_enclosures(result, reference) -> None:
    bounds = result.bounds
    read = bounds.read
    assert bounds.output.violations(last(reference, "o")) == 0
    assert bounds.residual_sum.violations(last(reference, "y")) == 0
    assert bounds.norm.output.violations(last(reference, "h")) == 0
    assert bounds.norm.normalized.violations(last(reference, "n")) == 0
    assert bounds.norm.scale_lower <= float(reference["q"]) <= bounds.norm.scale_upper
    a = last(reference, "a")
    assert bounds.activation.select(read).violations(a[read]) == 0
    assert bool((a[~read].abs() <= bounds.unread_activation[~read]).all())
    if bounds.gate is not None:
        assert bounds.gate.violations(last(reference, "g")) == 0
        assert bounds.activation_function.violations(last(reference, "s")) == 0
        assert bounds.up.violations(last(reference, "u")[read]) == 0


def check_pairwise(result, reference, model) -> None:
    """Both lower bounds on Σ_k Δ_k·y_k hold for the reference's y, for every pair checked; and a
    positive margin is a lower bound on the reference's own logit difference ℓ_w − ℓ_j."""
    pairwise = result.pairwise
    weight = model.lm_head.weight.detach().to(torch.float64)
    gain = model.model.norm.weight.detach().to(torch.float64)
    delta = (weight[pairwise.winner][None, :] - weight[pairwise.contenders]) * gain[None, :]
    actual = delta @ last(reference, "y")
    tolerance = 1e-9 * (delta.abs() @ last(reference, "y").abs())
    assert bool((pairwise.box_bound <= actual + tolerance).all())
    assert bool((pairwise.decomposed_bound <= actual + tolerance).all())
    logits = reference["logits"].to(torch.float64)
    difference = logits[pairwise.winner] - logits[pairwise.contenders]
    positive = pairwise.margin > 0
    assert bool((difference[positive] >= pairwise.margin[positive]).all())


def check_bytes(result, runtime, decomposition) -> None:
    store, head_store = runtime.store, runtime.head.store
    # The suffix: each page once per token; the budget before any fallback; metadata and the norm counted.
    seen = {role: torch.zeros(store.neurons, dtype=torch.bool, device=store.device) for role in store.adaptive}
    for read in store.reads:
        assert not bool(seen[read.role][read.neurons].any())
        seen[read.role][read.neurons] = True
    expected = math.ceil(runtime.budget * store.neurons - 1e-9)
    for role, count in result.neurons_read.items():
        assert count == (store.neurons if role == "gate" else expected)
    pages = sum(read.bytes for read in store.reads)
    assert result.suffix_bytes["total"] == store.metadata_bytes + runtime.resident_bytes + pages
    # The LM head: each row of each state once; the store's log equals the decomposition's accounting.
    per_state = [0] * decomposition.num_states
    marks: dict[tuple[str, int], torch.Tensor] = {}
    for read in head_store.reads:
        rows = torch.arange(head_store.out_features, device=head_store.device) if read.rows is None else read.rows
        key = ("weight" if read.kind in ("exact", "fallback") else read.kind, read.state)
        mask = marks.setdefault(key, torch.zeros(head_store.out_features, dtype=torch.bool, device=head_store.device))
        assert not bool(mask[rows].any())
        mask[rows] = True
        if read.kind in ("level", "exact"):
            per_state[read.state] += read.row_count
    fallback_rows = result.head_bytes["fallback_rows"] // decomposition.exact_row_bytes
    assert result.head_bytes == decomposition.materialized_bytes(per_state, fallback_rows)
    assert result.bytes_total == result.head_bytes["total"] + result.suffix_bytes["total"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_prefix_is_the_reference_bitwise(device, stage):
    model = tiny_model(device)
    generator = torch.Generator().manual_seed(1)
    with Reference(model) as reference:
        check_prefix(model, reference, stage, device, generator)


def check_prefix(model, reference, stage, device, generator) -> None:
    for _ in range(4):
        ids = torch.randint(512, (1, 9), generator=generator).to(device)
        values = reference(ids)
        prefix = suffix_prefix(model, ids, stage.boundary)
        assert torch.equal(prefix.residual, values["residual"]) and torch.equal(prefix.mlp_input, values["x"])
        if stage is SuffixStage.DOWN:
            assert torch.equal(prefix.activation, values["a"])
        else:
            assert prefix.activation is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("budget", [1.0, 0.9, 0.5])
def test_suffix_runtime_against_the_reference(device, stage, budget):
    model = tiny_model(device)
    runtimes = {name: build(model, stage, budget, assumptions) for name, assumptions in ASSUMPTIONS.items()}
    with Reference(model) as reference:
        paths = run_and_check(model, reference, runtimes, stage, device)
    assert paths.get("exact_suffix", 0) > 0
    if stage is SuffixStage.DOWN and budget == 1.0:
        assert paths.get("interval", 0) > 0 and paths.get("pairwise", 0) > 0


def run_and_check(model, reference, runtimes, stage, device) -> dict[str, int]:
    generator = torch.Generator().manual_seed(2)
    paths: dict[str, int] = {}
    for _ in range(24):
        ids = torch.randint(512, (1, 11), generator=generator).to(device)
        values = reference(ids)
        token = int(torch.argmax(values["logits"]))
        prefix = suffix_prefix(model, ids, stage.boundary)
        for name, (runtime, decomposition) in runtimes.items():
            result = runtime.run(prefix, trace=True)
            if result.bounds is not None:
                check_enclosures(result, values)
            if result.pairwise is not None:
                check_pairwise(result, values, model)
            if result.enclosure is not None and result.enclosure.trace is not None:
                logits = values["logits"].to(torch.float64)
                for bounds in result.enclosure.trace.states:
                    covered = logits if bounds.rows is None else logits[bounds.rows]
                    assert bool(((covered >= bounds.lower) & (covered <= bounds.upper)).all())
            if name == "faithful":
                assert result.token_id == token
                assert result.certified == (result.path in ("interval", "pairwise") or result.head.certified)
                paths[result.path] = paths.get(result.path, 0) + 1
                check_bytes(result, runtime, decomposition)
                if result.path == "exact_suffix":
                    for key in ("o", "y", "h"):
                        assert torch.equal(result.exact_suffix[key], values[key]), key
            else:
                assert not result.certified and result.head is None
                if result.would_certify:
                    assert result.token_id == token  # the probed platform rounds to nearest even
    return paths


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_kv_cache_stays_the_references_over_a_generation(device, stage):
    """Roadmap §2.8, Mode A: the suffix starts after the last attention, so the cache the prefix writes
    is the reference's, bitwise, at every step of a greedy generation with KV reuse."""
    from transformers import DynamicCache

    model = tiny_model(device)
    runtime, _ = build(model, stage, 0.9)
    generator = torch.Generator().manual_seed(4)
    for _ in range(3):
        reference_cache, cache = DynamicCache(config=model.config), DynamicCache(config=model.config)
        reference_ids = ids = torch.randint(512, (1, 7), generator=generator).to(device)
        for _ in range(6):
            with torch.inference_mode():
                logits = model(input_ids=reference_ids, past_key_values=reference_cache, use_cache=True, logits_to_keep=1).logits
            reference_token = int(torch.argmax(logits[0, -1]))
            result = runtime.run(suffix_prefix(model, ids, stage.boundary, past_key_values=cache))
            assert result.token_id == reference_token
            for ours, theirs in zip(cache.layers, reference_cache.layers):
                assert torch.equal(ours.keys, theirs.keys) and torch.equal(ours.values, theirs.values)
            if result.exact_suffix is not None:
                assert result.exact_suffix["h"].shape[1] == ids.shape[1]  # the step's own shape, as the reference
            reference_ids = ids = torch.tensor([[reference_token]], device=device)


def test_experimental_rounding_cannot_certify():
    model = tiny_model("cpu")
    runtime, _ = build(model, SuffixStage.DOWN, 1.0)
    with pytest.raises(ValueError):
        AdaptiveSuffixRuntime(SuffixStage.DOWN, runtime.head, model.model.norm, runtime.store, assumptions=EXPERIMENTAL)


def test_disabling_a_rounding_term_breaks_the_pairwise_bound(monkeypatch):
    """Guard check: without the roundings of o and y, the decomposed bound exceeds the reference's value."""
    model = tiny_model("cpu")
    runtime, _ = build(model, SuffixStage.DOWN, 1.0, row_limits=None)
    monkeypatch.setattr(pairwise_module, "rounding_error_upper", lambda magnitude, dtype, model: torch.zeros_like(magnitude))
    generator = torch.Generator().manual_seed(3)
    violations = checked = 0
    with Reference(model) as reference:
        for _ in range(40):
            ids = torch.randint(512, (1, 11), generator=generator)
            values = reference(ids)
            result = runtime.run(suffix_prefix(model, ids, "down_proj"))
            if result.pairwise is None:
                continue
            checked += 1
            try:
                check_pairwise(result, values, model)
            except AssertionError:
                violations += 1
    assert checked > 0 and violations > 0


@pytest.mark.model
@pytest.mark.parametrize("stage", STAGES)
def test_suffix_runtime_on_the_real_model(model, prompt_ids, reference_numerics, stage):
    runtime, decomposition = build(model, stage, 1.0)
    with Reference(model) as reference:
        for ids in prompt_ids:
            values = reference(ids)
            result = runtime.run(suffix_prefix(model, ids, stage.boundary), trace=True)
            assert result.token_id == int(torch.argmax(values["logits"]))
            check_enclosures(result, values)
            check_bytes(result, runtime, decomposition)
            if result.pairwise is not None:
                check_pairwise(result, values, model)
            if result.exact_suffix is not None:
                assert all(torch.equal(result.exact_suffix[key], values[key]) for key in ("o", "y", "h"))
