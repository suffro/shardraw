"""Phase 4B (decision 0008): dense parameters (a MoE layer's shared experts) served from storage at every call, exactly."""

from __future__ import annotations

import pytest
import torch
import transformers

from awpmi.materialization.backend import MaterializationBackend
from awpmi.materialization.weights import WeightStore
from awpmi.models import checkpoint, moonlight
from awpmi.models.moe import StreamedExperts
from awpmi.models.streamed import StreamedParameters
from awpmi.storage.pack import SourceFile, open_pack
from awpmi.storage.store import FileBackedPageStore
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import DEVICES
from tests.test_moe_chunked import budget
from tests.test_moe_compact import assert_same_run, expert_store, greedy, prompts
from tests.test_moonlight import tiny_moonlight


@pytest.mark.parametrize("device", DEVICES)
def test_streamed_shared_experts_reproduce_the_resident_model(tmp_path, device):
    source = tiny_moonlight(seed=31)
    source.save_pretrained(tmp_path / "checkpoint")
    files = {p.name: p for p in sorted((tmp_path / "checkpoint").glob("*.safetensors"))}
    resident = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint", dtype=torch.bfloat16).to(device).eval()
    inputs = prompts(device)
    reference = [greedy(resident, ids, steps=3) for ids in inputs]
    model, _ = checkpoint.load_model_without_experts(str(tmp_path / "checkpoint"), None, torch.bfloat16, device, files)
    shared = moonlight.shared_experts(model)
    owners = sorted(name for name, module in model.named_modules() if any(module is m for m in shared.values()))
    names = [f"{owner}.{p}" for owner in owners for p, _ in model.get_submodule(owner).named_parameters()]
    shared_bytes = sum(p.numel() * p.element_size() for module in shared.values() for p in module.parameters())
    segments = checkpoint.parameter_segments(model, names, files)
    store = FileBackedPageStore(dict(files), segments, direct=True)
    weights = WeightStore(MaterializationBackend(store, device, PageStreamer(device)))
    files_info = {k: checkpoint.CheckpointFile(k, SourceFile("example/split", "0" * 40, k), p, None) for k, p in files.items()}
    checkpoint.write_expert_index(model, tmp_path / "index", files_info, sources=moonlight.expert_sources(model))
    pack = open_pack(tmp_path / "index", verify="size", resolve=lambda requested: files[requested.filename])
    experts = expert_store(pack, device)
    routed = StreamedExperts(model, experts, compact=True, max_call_bytes=budget(model, 2)).install()
    dense = StreamedParameters(model, weights, {name: name for name in names}).install()
    try:
        assert all(module.gate_proj.weight is None for module in shared.values())  # nothing resident between calls
        for ids, expected in zip(inputs, reference):
            assert_same_run(greedy(model, ids, steps=3), expected)
        forwards = 4 * len(inputs)  # a prefill and 3 decode steps per prompt; each call of a projection serves its weight
        assert dense.calls == forwards * len(names) and dense.served_bytes == forwards * shared_bytes
        report = weights.backend.report()
        assert report["materialization"]["fetched_bytes"] == dense.served_bytes == report["storage"]["logical_bytes"]
    finally:
        dense.remove(restore=True)
        routed.remove()
        store.close()
        experts.weights.backend.store.close()
    assert all(module.gate_proj.weight is not None for module in shared.values())  # resident again
    with pytest.raises(KeyError):
        StreamedParameters(model, weights, {f"{owners[0]}.missing": names[0]}).install()
