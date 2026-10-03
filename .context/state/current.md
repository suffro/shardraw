# Current State

## Current focus

**Phase 4B (Moonlight-16B-A3B out of VRAM and out of host RAM) is complete (2026-10-03).
Correctness and gates A–F pass.** The full report is `history/2026-10-03-awpmi-phase4b-report.md`;
the decisions are in decision 0008.

- **Answer to the phase's question: yes.** Shardraw runs Moonlight-16B-A3B, DeepSeek-V3's
  architecture at 16 B parameters, on this machine:
  - its 28.8 GB of routed experts are 4.8× the 6 GB device cap and 3.4× the GPU;
  - its 31.9 GB checkpoint does not fit the 32 GB of RAM next to the OS.

  Every step reproduces an independent, fully materialized reference bit for bit: tokens,
  logits, the whole KV cache, and per layer attention, router logits/scores/indices/weights,
  every (token, expert) output where checked, the shared experts' and the MoE block's outputs.
  That holds for 726 of 726 streamed steps per run, in two runs with different hash seeds and
  identical digests.
- **What was built** (decision 0008):
  - bounded experts calls (`StreamedExperts(max_call_bytes=…)`): a call that would exceed the
    budget runs in chunks of experts. Each expert matrix is a `ChunkedExpertWeight` stand-in
    reachable only by `weight[slot]` or `torch._grouped_mm`; transformers' own combine runs once
    per call;
  - the independent streaming reference (`awpmi.streaming_reference`): transformers' model and
    loader, one experts layer at a time from the checkpoint, no Shardraw import. Checked against
    `from_pretrained` on Moonlight truncated to 4 layers: all equal;
  - the Moonlight adapter (layout, routers with float32 bias, shared experts, routing
    configuration, profile);
  - `StreamedParameters`: dense parameters served from storage at every call (the shared-experts
    measurement).
- **Results** (run1):

  | | No cache | Hotness, 80 experts | Shared experts streamed |
  | --- | --- | --- | --- |
  | Drive per decode token (of all experts) | 0.0938 (2.70 GB) | 0.0796 | 0.0938 + 0.90 GB |
  | Decode step, profile | 1.27 s | 1.19 s | +0.44 s |
  | Prefill expert buffers (max) | 173 MB (budget 256 MiB) | 173 MB | 173 MB |

  - Without the budget, a prefill holds a whole layer (1,107 MB); chunks of 15 experts cost 8.5%
    of a prefill's host time.
  - Host memory peaks at a 3.25 GB working set (streamed) and 2.1 GB per step (reference).
  - LRU caches of 40 and 80 experts never hit: a decode token loads 156 experts. A replay of the
    routing, checked against the measured hits, predicts 67–83% fewer decode reads with an
    11–17 GB host-RAM tier.
  - Time: the drive is 66.5% of decode and 78.6% of prefill; the transformer's Python and launches
    22%; per-request transfer overhead 11%; GEMM about 2%.
- **Findings:**
  - under Windows' WDDM driver model, device memory is charged to a process's private bytes (the
    host-memory gate uses the working set and the host commit);
  - the cache pool's fragmentation caps the device cache at about 80 experts under the cap;
  - LRU is useless below one token's working set;
  - transformers' loader reads at 1.06 GB/s, so the reference takes 29 s per step.

Phases 1A, 1B, 1C, 2, 3 and 4A are complete. Their reports are in `history/`.

## Recent relevant changes

- New modules:
  - `src/awpmi/streaming_reference.py`;
  - `src/awpmi/models/moonlight.py`;
  - `src/awpmi/models/streamed.py`.
- Changed modules:
  - `models/moe.py` (chunked calls: `max_call_bytes`, `ChunkedExpertWeight`, `_ChunkedCall`,
    `ExpertCall.chunks` and `per_assignment_outputs(weights=…)`);
  - `models/checkpoint.py` (`checked_expert_sources`, `neighbours`, `parameter_segments`);
  - `models/olmoe.py` (uses the shared layout check; behavior unchanged);
  - `materialization/weights.py` (`assemble` of some parameters);
  - `materialization/backend.py` (`largest_request_bytes`).
- Dependency: `tiktoken` 0.14.0, for Moonlight's official tokenizer (remote code, read before use,
  pinned revision).
- Benchmarks:
  - `benchmarks/moonlight_runtime.py` (stages prepare, reference, stream, digest);
  - `moonlight_reference_check.py`, `moonlight_profile.py`, `moonlight_report.py`;
  - `configs/phase4b-moonlight.yaml`, with gates fixed before the full runs;
  - raw results in `experiments/phase4b/moonlight-run{1,2}` and `experiments/phase4b/reference-check`;
  - the expert index under `packs/` (gitignored, `awpmi pack expert-index --config configs/phase4b-moonlight.yaml`).
- Decision 0008 is new.
- 510 tests, 58 of them new:
  - chunked calls on 7 architectures × CPU/CUDA × `grouped_mm`/eager, the adversarial
    accumulation test, misuse, budget and allocator guards, caches, hash seeds;
  - the streaming reference against `from_pretrained`;
  - the Moonlight adapter, streamed shared experts, layering.

## Next

The next phase is **not started**. It needs the user's go-ahead. The report (§19) recommends, from
the measured bottleneck (bytes from the drive):

1. **AWPMI inside routed experts, first as an oracle measurement**: how many neuron pages of the
   last MoE layer's routed experts a certificate needs, before any runtime.
   - The hook is the chunked call (`_ChunkedCall.materialize`, `_chunked_grouped_mm`).
   - The down projection's layout must be decided first: down columns are strided in the
     checkpoint.
2. **Native runtime and a host-RAM expert tier**, if usable speed on this machine comes first:
   - an 11–17 GB host tier cuts 67–83% of decode reads (replay);
   - Rust for planning, submission and cache bookkeeping (11% of decode, 18% of prefill);
   - CUDA graphs for the launch-bound decode path (22%).
3. **DeepSeek-V3-class scaling last.** Its architecture is ready, but it needs:
   - the FP8 native reference decision and FP8 layouts;
   - a host for the reference's 11 GB layers;
   - about 700 GB of storage;
   - fewer bytes per token.

Open decisions for the user:

- the order above;
- the reference for FP8 experts;
- still open from Phase 2: RN-even for elementwise kernels, and Phase 2 on a larger model.

## Blockers

None technical. The next phase needs the user's decision to proceed.
