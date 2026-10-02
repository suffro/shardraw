"""Phase 3: the Phase 1C LM head on physical storage reproduces the resident runtime bitwise."""

from __future__ import annotations

import pytest
import torch

from awpmi.bounds.coarse import CoarseArithmetic
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF
from awpmi.decomposition import RefinementDecomposition
from awpmi.materialization.backend import MaterializationBackend
from awpmi.refinement_head import FallbackMode, RefinementLMHead, RefinementRunResult
from awpmi.storage.cache import LRUPolicy, PageCache
from awpmi.storage.fileio import os_read_counters
from awpmi.storage.pack import open_pack
from awpmi.stores.refinement import EXACT_SEGMENT, PackedRefinementStore, level_segment, write_refinement_pack
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import DEVICES
from tests.test_refinement_oracle import numerics_for, reference_logits
from tests.test_refinement_runtime import check_run, system_with_a_winning_tie


def head_on(store: PackedRefinementStore, fallback: FallbackMode, chunk_rows: int = 64) -> RefinementLMHead:
    return RefinementLMHead(
        store,
        numerics_for_store(store),
        CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, store.in_features),
        fallback=fallback,
        chunk_rows=chunk_rows,
    )


def numerics_for_store(store: PackedRefinementStore):
    return numerics_for(torch.empty(store.out_features, store.in_features, dtype=store.dtype))


def file_store(pack_dir, device: str, direct: bool | None, cache_base: bool = False, verify: str = "segments") -> PackedRefinementStore:
    """A store over the pack: direct or buffered file reads, or (direct=None) the pack loaded in host memory."""
    pack = open_pack(pack_dir, verify=verify)
    cache = PageCache(64 << 20, LRUPolicy()) if cache_base else None
    pages = pack.load() if direct is None else pack.store(direct=direct)
    backend = MaterializationBackend(pages, device, PageStreamer(device), cache)
    if cache_base and not backend.resident:  # a host-memory store on a CPU device is already resident
        backend.pin(level_segment(0))
    return PackedRefinementStore.from_pack(pack, backend)


def assert_same_run(a: RefinementRunResult, b: RefinementRunResult) -> None:
    """Two runs on the same input: same decision, same reads, same bounds, same fallback logits, bit for bit."""
    for name in (
        "token_id", "certified", "fallback", "fallback_reason", "masked_guard_tripped", "decision_state",
        "winner", "competitor", "certificate_margin", "contenders", "rows_loaded", "bytes",
    ):
        assert getattr(a, name) == getattr(b, name), name
    assert [(r.kind, r.state, r.row_count) for r in a.reads] == [(r.kind, r.state, r.row_count) for r in b.reads]
    for ra, rb in zip(a.reads, b.reads):
        assert (ra.rows is None) == (rb.rows is None) and (ra.rows is None or torch.equal(ra.rows.cpu(), rb.rows.cpu()))
    assert (a.fallback_logits is None) == (b.fallback_logits is None)
    if a.fallback_logits is not None:
        assert torch.equal(a.fallback_logits.cpu(), b.fallback_logits.cpu())
    if a.trace is not None:
        assert torch.equal(a.trace.coarse_sum.cpu(), b.trace.coarse_sum.cpu())
        assert torch.equal(a.trace.lower.cpu(), b.trace.lower.cpu()) and torch.equal(a.trace.upper.cpu(), b.trace.upper.cpu())
        for sa, sb in zip(a.trace.states, b.trace.states, strict=True):
            for field in ("center", "radius", "lower", "upper"):
                assert torch.equal(getattr(sa, field).cpu(), getattr(sb, field).cpu()), field


def read_bytes(result: RefinementRunResult) -> int:
    """Logical bytes of a run's weight reads (the store's log, without the resident metadata)."""
    return result.bytes["total"] - result.bytes["metadata"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("direct", [True, False, None])
@pytest.mark.parametrize("spec", ["q6+q4", "q2+q2+q4"])
def test_file_backed_runtime_reproduces_the_resident_runtime(tmp_path, device, direct, spec):
    weight, hidden = system_with_a_winning_tie(device, torch.bfloat16, seed=4)
    reference = reference_logits(weight, hidden)
    decomposition = RefinementDecomposition.build(weight, spec)
    write_refinement_pack(decomposition, tmp_path / "pack")
    resident = PackedRefinementStore.from_decomposition(decomposition)
    for cache_base in (False, True):
        stored = file_store(tmp_path / "pack", device, direct, cache_base)
        assert stored.content_digest() == resident.content_digest()
        backend = stored.backend
        for mode in FallbackMode:
            heads = head_on(resident, mode), head_on(stored, mode)
            for b in range(hidden.shape[0]):
                backend.reset_stats()
                ours, theirs = heads[1].run(hidden[b], trace=True), heads[0].run(hidden[b], trace=True)
                check_run(ours, heads[1], decomposition, reference[:, b])
                assert_same_run(ours, theirs)
                report = backend.report()
                # The backend was asked for exactly what the store logged; what the cache did not hold,
                # storage served, and nothing else.
                served = report["materialization"]
                assert served["requested_bytes"] == read_bytes(ours)
                assert served["cache_hit_bytes"] + served["fetched_bytes"] == served["requested_bytes"]
                assert report["storage"]["logical_bytes"] == served["fetched_bytes"]
                assert cache_base or served["cache_hit_bytes"] == 0
                if device == "cuda":
                    assert report["transfer"]["h2d_bytes"] == report["storage"]["logical_bytes"]
                if direct and os_read_counters() is not None:
                    assert report["storage"]["os_read_bytes"] == report["storage"]["physical_bytes"]
                if direct is None:
                    assert report["storage"]["physical_bytes"] == report["storage"]["logical_bytes"]
        stored.backend.store.close()


@pytest.mark.parametrize("device", DEVICES)
def test_rows_the_runtime_did_not_read_cannot_change_its_result(tmp_path, device):
    """Overwrite every level-1 and exact row a run did not read with garbage: the run is unchanged."""
    weight, hidden = system_with_a_winning_tie(device, torch.bfloat16, seed=6)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    write_refinement_pack(decomposition, tmp_path / "pack")
    checked = 0
    for b in range(hidden.shape[0]):
        stored = file_store(tmp_path / "pack", device, direct=True, verify="size")
        before = head_on(stored, FallbackMode.MASKED).run(hidden[b], trace=True)
        stored.backend.store.close()
        if before.fallback is FallbackMode.FULL:
            continue  # the full fallback reads every row
        read = {segment: set() for segment in (level_segment(1), EXACT_SEGMENT)}
        for entry in before.reads:
            segment = EXACT_SEGMENT if entry.kind != "level" else level_segment(entry.state)
            if segment in read:
                read[segment].update(entry.rows.tolist())
        pack = open_pack(tmp_path / "pack", verify="size")
        raw = bytearray(pack.files["pack"].read_bytes())
        for segment, rows in read.items():
            info = pack.segments[segment]
            for row in set(range(info.rows)) - rows:
                start = info.offset + row * info.row_bytes
                # Finite garbage (BF16 0x3F3F ≈ 0.75): the masked fallback's start-up self-test reads
                # every exact row, and must still see a platform that keeps rows independent.
                raw[start : start + info.row_bytes] = b"\x3f" * info.row_bytes
        original = pack.files["pack"].read_bytes()
        pack.files["pack"].write_bytes(bytes(raw))
        try:
            poisoned = file_store(tmp_path / "pack", device, direct=True, verify="size")
            head = RefinementLMHead(
                poisoned,
                numerics_for_store(poisoned),
                CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, poisoned.in_features),
                fallback=FallbackMode.MASKED,
                chunk_rows=64,
                self_test_trials=1,
            )
            assert head.masked_enabled
            assert_same_run(head.run(hidden[b], trace=True), before)
            poisoned.backend.store.close()
        finally:
            pack.files["pack"].write_bytes(original)
        checked += 1
    assert checked > 0


@pytest.mark.model
def test_file_backed_runtime_on_the_real_lm_head(tmp_path, model, prompt_ids, reference_numerics, experiment_config):
    """SmolLM2: levels from a pack, exact rows straight from the published checkpoint's embedding tensor."""
    from awpmi.models.smollm2 import final_hidden_state, lm_head_weight
    from awpmi.reference import ReferenceRunner
    from awpmi.storage.pack import SourceFile

    weight = lm_head_weight(model)
    decomposition = RefinementDecomposition.build(weight, "q6+q4")
    source = SourceFile(experiment_config.model.repository, experiment_config.model.revision, "model.safetensors")
    pack = write_refinement_pack(decomposition, tmp_path / "pack", source=(source, "model.embed_tokens.weight"))
    assert pack.segments[EXACT_SEGMENT].file == "source"  # the checkpoint's own bytes, not a copy
    device = str(weight.device)
    stored = file_store(tmp_path / "pack", device, direct=True)
    resident = PackedRefinementStore.from_decomposition(decomposition)
    assert stored.content_digest() == resident.content_digest()
    coarse = CoarseArithmetic(FP32_ACCUMULATION_UNIT_ROUNDOFF, weight.shape[1])
    runner = ReferenceRunner(model)
    for mode in FallbackMode:
        ours = RefinementLMHead(stored, reference_numerics, coarse, fallback=mode)
        theirs = RefinementLMHead(resident, reference_numerics, coarse, fallback=mode)
        for ids in prompt_ids:
            reference = runner.next_token(ids)
            hidden = final_hidden_state(model, ids).reshape(-1).contiguous()
            stored.backend.reset_stats()
            result = ours.run(hidden, trace=True)
            check_run(result, ours, decomposition, reference.logits)
            assert_same_run(result, theirs.run(hidden, trace=True))
            report = stored.backend.report()
            assert report["storage"]["logical_bytes"] == read_bytes(result)
            assert report["storage"]["physical_bytes"] < weight.numel() * weight.element_size() or result.fallback is FallbackMode.FULL
    stored.backend.store.close()
