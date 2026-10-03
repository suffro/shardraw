"""Phase 4B (decision 0008): the streaming reference is from_pretrained's model, with one experts layer existing at a time."""

from __future__ import annotations

import pytest
import torch
import transformers

from awpmi.streaming_reference import ReferenceCall, StreamingReference, experts_modules
from tests.conftest import DEVICES
from tests.test_moe import tiny_model
from tests.test_moe_compact import kv_tensors, prompts

# Checkpoints as save_pretrained writes them: DeepSeek-V3 and Qwen3-MoE split each expert, Mixtral also renames
# (block_sparse_moe), which the reference resolves with transformers' own renaming of each key.
ARCHITECTURES = ("deepseek_v3", "qwen3_moe", "mixtral")


class Observer:
    """Per forward: every attention output, every experts call (inputs, output, per-assignment outputs), every MoE block output."""

    def __init__(self, model) -> None:
        self.records: list = []
        self.handles = []
        experts = experts_modules(model)
        for name, module in model.named_modules():
            if name.endswith("self_attn") or name in {e.rpartition(".")[0] for e in experts}:
                self.handles.append(module.register_forward_hook(self._record(name)))

    def _record(self, name):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.records.append((name, tensor.detach().clone()))

        return hook

    def on_call(self, call) -> None:
        for kind in ("hidden_states", "top_k_index", "top_k_weights", "output"):
            self.records.append((f"{call.module}:{kind}", getattr(call, kind).detach().clone()))
        self.records.append((f"{call.module}:per_assignment", call.per_assignment_outputs().detach().clone()))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def run(model, ids, steps: int):
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    logits = [output.logits[:, -1].clone()]
    for _ in range(steps):
        output = model(input_ids=logits[-1].argmax(-1, keepdim=True), past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        logits.append(output.logits[:, -1].clone())
    return logits, [t.clone() for t in kv_tensors(output.past_key_values)]


def offload_observer(model, observer: Observer):
    """Experts calls of a resident model, as `ReferenceCall`s (the same observation as the streaming reference's)."""
    handles = []
    names = ("hidden_states", "top_k_index", "top_k_weights")
    for name, module in experts_modules(model).items():

        def hook(module_, args, kwargs, output, name=name):
            values = [args[i] if len(args) > i else kwargs[names[i]] for i in range(3)]
            observer.on_call(ReferenceCall(name, module_, *values, output))

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    return handles


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_streaming_reference_equals_from_pretrained(tmp_path, device, architecture):
    source = tiny_model(architecture, "cpu", seed=21)
    source.save_pretrained(tmp_path / "checkpoint")
    resident = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint", dtype=torch.bfloat16).to(device).eval()
    reference = StreamingReference(str(tmp_path / "checkpoint"), None, torch.bfloat16, device).load()
    try:
        # The same weights, dtypes and buffers (the router bias's float32 dtype plan, the rotary frequencies), experts aside.
        expected = {**dict(resident.named_parameters()), **dict(resident.named_buffers())}
        streamed = {**dict(reference.model.named_parameters()), **dict(reference.model.named_buffers())}
        experts = {f"{name}.{p}" for name, module in experts_modules(resident).items() for p, _ in module.named_parameters(recurse=False)}
        assert set(streamed) == set(expected) - experts
        for name, tensor in streamed.items():
            assert tensor.dtype == expected[name].dtype and tensor.device == expected[name].device and torch.equal(tensor, expected[name]), name
        assert reference.model.config._experts_implementation == resident.config._experts_implementation
        assert reference.model.config._attn_implementation == resident.config._attn_implementation
        # Every observed tensor equal, prefill and decode with the KV cache.
        for ids in prompts(device):
            observers = []
            for model, kind in ((resident, "resident"), (reference.model, "streamed")):
                observer = Observer(model)
                handles = offload_observer(model, observer) if kind == "resident" else []
                reference.on_call = observer.on_call if kind == "streamed" else None
                try:
                    observer.result = run(model, ids, steps=3)
                finally:
                    observer.close()
                    for handle in handles:
                        handle.remove()
                observers.append(observer)
            a, b = observers
            assert [name for name, _ in a.records] == [name for name, _ in b.records]
            for (name, x), (_, y) in zip(a.records, b.records, strict=True):
                assert x.dtype == y.dtype and torch.equal(x, y), name
            for x, y in zip(a.result[0] + a.result[1], b.result[0] + b.result[1], strict=True):
                assert torch.equal(x, y)
    finally:
        reference.close()


def test_the_reference_holds_one_experts_layer_at_a_time(tmp_path):
    device = DEVICES[-1]
    source = tiny_model("deepseek_v3", "cpu", seed=22)
    source.save_pretrained(tmp_path / "checkpoint")
    layer_bytes = {
        name: sum(t.numel() * t.element_size() for t in module.parameters(recurse=False)) for name, module in experts_modules(source).items()
    }
    reference = StreamingReference(str(tmp_path / "checkpoint"), None, torch.bfloat16, device).load()
    try:
        modules = experts_modules(reference.model)
        assert all(getattr(module, p) is None for module in modules.values() for p in ("gate_up_proj", "down_proj"))
        module = next(iter(modules.values()))
        hidden = torch.randn(3, 64, device=device).to(torch.bfloat16)
        index = torch.tensor([[0, 1], [2, 3], [1, 5]], device=device)
        with pytest.raises((TypeError, AttributeError, RuntimeError)):
            module.forward(hidden, index, torch.full((3, 2), 0.5, device=device))  # no hook: no weights exist
        run(reference.model, prompts(device)[1], steps=2)
        assert reference.peak_expert_device_bytes == max(layer_bytes.values()) and reference.expert_device_bytes == 0
        assert reference.loads == 3 * len(modules)
        assert all(getattr(module, p) is None for module in modules.values() for p in ("gate_up_proj", "down_proj"))
    finally:
        reference.close()


def test_residency_changes_no_reference_result(tmp_path):
    device = DEVICES[-1]
    source = tiny_model("qwen3_moe", "cpu", seed=23)
    source.save_pretrained(tmp_path / "checkpoint")
    results = []
    for resident in ((), ("model.layers.1.mlp.experts",)):
        reference = StreamingReference(str(tmp_path / "checkpoint"), None, torch.bfloat16, device, resident=resident).load()
        try:
            results.append([run(reference.model, ids, steps=3) for ids in prompts(device)])
            results[-1].append(reference.loads)
        finally:
            reference.close()
    loads_all, loads_resident = results[0].pop(), results[1].pop()
    assert loads_resident < loads_all
    for (logits_a, kv_a), (logits_b, kv_b) in zip(results[0], results[1], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(logits_a + kv_a, logits_b + kv_b, strict=True))
