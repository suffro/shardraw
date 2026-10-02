# Current State

## Current focus

**Phase 3 (real selective materialization and a general storage substrate) is complete
(2026-10-02). Gates A, B and C pass.** The full report is
`history/2026-10-02-awpmi-phase3-report.md`; the decisions are in decision 0006.

- **What was built** (decision 0006):
  - a storage core that knows no model: `awpmi.storage`, `awpmi.streaming` and
    `awpmi.materialization`;
  - segments of rows in safetensors files, read plans aligned to 4 KiB, and positioned direct
    reads with 8 threads;
  - a pinned, double-buffered streamer with a copy stream and events, and prefetch tickets;
  - a budgeted device page cache (LRU or hotness);
  - packs with a hashed manifest that refer to published checkpoint tensors (`awpmi pack`);
  - the Phase 1C store reading through it, its mathematics unchanged;
  - a model-agnostic MoE adapter for transformers' experts modules.
- **Correctness:**
  - LM head: 0 mismatches. The drive-backed runtime is bit for bit the resident one on
    6,000 runs per benchmark run, and the resident one reproduces the Phase 1C records field
    by field.
  - Every storage audit holds, and the OS's own read counters equal the store's.
  - MoE: 3,275 of 3,275 streamed steps are bit for bit the resident Granite model's.
  - Both benchmarks were run twice, with identical digests.
- **LM head on the drive** (SmolLM2, masked fallback), per token, as fractions of the BF16 head:

  | | Logical | Read from the drive | Moved to the GPU |
  | --- | --- | --- | --- |
  | Phase 1C on the drive | 0.404 | **0.468** | 0.390 |
  | Full BF16 head | 1.0 | 1.0 | 1.0 |
  | Base level cached on the device | 0.404 | 0.090 | 0.011 |

  - Amplification is 1.20×, all of it from the 4 KiB blocks around the int4 refinement rows.
  - Phase 1C's 4 KiB model had predicted 0.482 including resident metadata; measured, it is
    0.468 + 0.014.
  - Time is not a win yet: 43.3 ms against 22.2 ms for the full head from the drive. The eager
    PyTorch runtime is still about 20 ms.
- **MoE** (Granite 3.1 1B-A400M, 24 × 32 experts, top-8):
  - device memory falls from 2.70 GB to 0.39 GB;
  - a decode token reads 0.251 of the expert bytes without a cache (605 MB, 451 ms), 0.080–0.096
    with a cache of half the experts, and 1.0 with dense streaming (1,074 ms);
  - the experts are read from the published checkpoint, with nothing copied.
- **Finding:** on NTFS, direct reads of a file with an active OS cache map are 4–6× slower.
  Packs are verified with direct reads, and benchmarks never mix buffered and direct access to
  one file.

Phases 1A, 1B, 1C and 2 are complete. Their reports are in `history/`.

## Recent relevant changes

- New packages:
  - `src/awpmi/storage/` (`layout`, `fileio`, `store`, `cache`, `pack`);
  - `src/awpmi/streaming/streamer.py`;
  - `src/awpmi/materialization/` (`backend`, `weights`).
- New modules: `src/awpmi/models/moe.py` and `src/awpmi/cli.py`. `pyproject.toml` gains the
  `awpmi` script; `uv.lock` is unchanged.
- Changed modules:
  - `stores/refinement.py`: reads through a `MaterializationBackend`; `from_pack`,
    `write_refinement_pack`; level records hold the payload and then the scale.
  - `refinement_head.py`: `*:read` timer stages, for timing only.
- Benchmarks:
  - `benchmarks/storage_runtime.py`, `storage_report.py`, `moe_runtime.py` and `moe_report.py`;
  - `configs/phase3-storage.yaml` and `configs/phase3-moe.yaml`, with gates fixed before the
    runs;
  - raw results in `experiments/phase3/{storage,moe}-run{1,2}`;
  - packs under `packs/` (gitignored, rebuilt by `awpmi pack`).
- Decision 0006 is new. Decision 0004 points to it.
- 399 tests, 76 of them new: storage, streaming, refinement on storage, MoE (7 architectures),
  layering, hotness determinism across hash seeds.

## Next

Phase 4 or the DeepSeek-class path is **not started**. It needs the user's go-ahead.

1. **Recommended next step (report §10):** a MoE whose experts exceed GPU memory with the same
   code: OLMoE-1B-7B-0125, with only non-expert weights loaded and dense streaming as the
   reference. That needs:
   - a compact expert call: slot buffers for the routed experts only, `top_k_index` remapped,
     exactness re-verified;
   - zero-copy packs for checkpoints that split gate and up.

   Then DeepSeek-V2-Lite or Moonlight-16B (DeepSeek architecture, more than this GPU, close to
   the RAM).
2. **Open decisions for the user:**
   - Certify against a quantized reference? DeepSeek-V3 ships FP8 experts, and DwarfStar runs
     2-bit ones. That would revise decision 0001's BF16 reference.
   - Still open from Phase 2: RN-even for elementwise kernels, and Phase 2 on a larger model.
3. **Open levers:**
   - fused kernels, now justified by the profile (Phase 4);
   - native read submission (`io_uring`/`IoRing` or a small extension) and per-layer batching;
   - 512-byte reads or row clustering for the int4 level (0.078 of the head of amplification);
   - a prefetch predictor better than the previous step (which would waste half);
   - a cache admission freeze during long prefills.

## Blockers

None technical. The next phase needs the user's decision to proceed.
