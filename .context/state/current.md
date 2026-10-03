# Current State

## Current focus

**Phase 4A (the first real out-of-VRAM mixture of experts) is complete (2026-10-03).
Correctness and gates A, B, C and D pass.** The full report is
`history/2026-10-03-awpmi-phase4a-report.md`; the decisions are in decision 0007.

- **Answer to the phase's question: yes.** Shardraw runs OLMoE-1B-7B, whose 12.9 GB of experts
  exceed this 8 GB GPU (and the 6 GB cap the streamed process runs under). Every step reproduces
  the fully materialized reference bit for bit:
  - tokens, logits and the whole KV cache;
  - per layer, the router logits, routed experts, their weights, every (token, expert) output
    and the experts output.

  That holds for 2,089 of 2,089 streamed steps per run, in two runs with different hash seeds.
- **What was built** (decision 0007):
  - a compact experts call: buffers for the routed experts only, in ascending expert order,
    `num_experts` and `top_k_index` remapped for the call, parameters `None` between calls;
  - composed segments: a row made of byte spans of several files, so a checkpoint that stores
    each expert as separate tensors is read in place. The index is built from headers alone
    (`awpmi pack expert-index`), its layout derived from transformers' conversion mapping;
  - an expert-free loader (meta skeleton, direct reads, transformers' dtype plan);
  - reference profiles (`awpmi.profiles`: BF16, FP16, native quantized declared only);
  - `FullLayerOffload`, the reference for a model that does not fit: transformers' own model,
    one experts layer materialized at a time;
  - a CUDA memory pool for cache entries; direct copies of long runs; 32 MiB staging slots;
  - the OLMoE adapter (layout check, routers, profile).
- **Results** (run1, per decode token, as fractions of all expert bytes):

  | | No cache | LRU 1/8 | LRU 1/4 | Hotness 1/4 | Every expert |
  | --- | --- | --- | --- | --- | --- |
  | Read from the drive | 0.1251 | 0.0819 | 0.0672 | 0.0658 | 1.0006 |
  | Cache hit rate | – | 0.345 | 0.463 | 0.475 | – |
  | Peak GPU memory | 1.14 GB | 2.74 GB | 4.35 GB | 4.35 GB | 1.82 GB |

  - Compact buffers hold |R| × 12 MiB: 101 MB for a decode call, against Phase 3's 805 MB.
  - Prefill routes 51.5 of 64 experts per layer, and needs up to one layer of buffers (805 MB).
  - The streamed process loads 0.95 GB of non-expert weights in 1.2 s and peaks at 2.7 GB of
    host memory; the reference needs 15.1 GB.
  - Time (no digests): a decode step takes 759 ms without a cache and 525 ms with LRU at 1/4. The drive is the first bottleneck, at 94% of its
    sequential rate; then Python and kernel launches, then the transfer path's per-request
    overhead.
- **Findings:**
  - `meta` weights give CUDA garbage, not errors;
  - cache pages fragment the allocator under a cap (fixed by a separate pool);
  - transformers' dtype plans must be replicated by any loader that bypasses `from_pretrained`;
  - OLMoE's `0125` base checkpoint is stored in FP32.

Phases 1A, 1B, 1C, 2 and 3 are complete. Their reports are in `history/`.

## Recent relevant changes

- New modules:
  - `src/awpmi/profiles.py`;
  - `src/awpmi/models/checkpoint.py`;
  - `src/awpmi/models/olmoe.py`.
- Changed modules:
  - `storage/layout.py` (`ComposedSegment`, `byte_runs`);
  - `storage/store.py` (multi-file plans with output offsets and `positions`);
  - `storage/pack.py` (manifest v2, publisher sha256, direct-read hashing);
  - `streaming/streamer.py` (`out`/`positions`, direct copies);
  - `materialization/backend.py` (`out=`, hits first, cache memory pool);
  - `materialization/weights.py` (`assemble`);
  - `models/moe.py` (compact mode, `FullLayerOffload`, `ExpertCall`);
  - `cli.py` (`pack expert-index`).
- Phase 3 regression check (its first prompts rerun against the committed run1 records):
  - Granite: 249 of 249 records identical.
  - LM head: 159 of 160. The other one differs only in the transfer counters
    (`gathered_bytes`, `h2d_copies`), the effect of direct copies.
- Benchmarks:
  - `benchmarks/olmoe_runtime.py`, `olmoe_profile.py` and `olmoe_report.py`;
  - `configs/phase4a-olmoe.yaml`, with gates fixed before the runs;
  - raw results in `experiments/phase4a/olmoe-run{1,2}`;
  - the expert index under `packs/` (gitignored).
- Decision 0007 is new.
- 452 tests, 53 of them new: composed storage, compact calls (7 architectures, CPU and CUDA,
  caches, hash seeds, sabotage), split checkpoints of 5 architectures, the expert-free loader,
  the offload reference, the OLMoE adapter.

## Next

The next phase is **not started**. It needs the user's go-ahead.

1. **Recommended next model: `moonshotai/Moonlight-16B-A3B`** (report §13).
   - It is DeepSeek-V3's architecture at 1/40 of the size: MLA, 64 + 2 experts, top-6, sigmoid
     `noaux_tc` router; 28.8 GB of BF16 experts, MIT license.
   - It forces what DeepSeek-V3 will need, at a checkable size:
     - a reference that does not fit in host memory, streamed per layer through an independent
       reader;
     - bounded prefill buffers: an experts call split over groups of experts, with an exact
       replica of the implementation's combine. That also enables ds4's compute-level hits
       first.
   - DeepSeek-V2-Lite is subsumed (same sizes, older router).
2. **Open decisions for the user:**
   - the reference for FP8 experts (DeepSeek-V3). `NATIVE_QUANTIZED_REFERENCE` is declared and
     refused for now;
   - still open from Phase 2: RN-even for elementwise kernels, and Phase 2 on a larger model.
3. **Open levers** (profile, report §9):
   - native read submission and overlap across layers;
   - fewer, larger reads per layer (one plan per expert for all its tensors);
   - an admission freeze during prefill;
   - fused kernels for the launch-bound transformer.

## Blockers

None technical. The next phase needs the user's decision to proceed.
