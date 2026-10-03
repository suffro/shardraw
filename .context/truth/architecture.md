# Architecture

## Overview

This repository hosts **AWPMI — Adaptive Weight-Page Materialization for Inference**, a
research prototype (Python package `awpmi`, at the repository root). AWPMI materializes
independently fetchable weight pages one at a time. It stops only when a conservative,
deterministic certificate proves that the next-token argmax equals that of the fully
materialized reference model. The plan is in `state/AWPMI_Implementation_Roadmap.md`.

Implemented so far:

- **Phase 1A**: a runtime that materializes input-column pages. Only the LM head is
  adaptive. The transformer body runs exactly ("exact prefix", roadmap Mode A), and all
  weights stay resident, so materialization is logical rather than physical.
- **Phase 1B**: an *oracle* for precision-refinement decompositions of the LM head, with
  row-selective refinement. It is a simulator, not a runtime: nothing in `awpmi.oracle`
  is called by a runtime.
- **Phase 1C**: the runtime of the decomposition Phase 1B chose (q6+q4, row-selective,
  tie-aware), `RefinementLMHead`. It reads a bit-packed store whose every read is
  counted. The store is still resident in memory: materialization is logical, but the
  bytes are those of real packed buffers.
- **Phase 2**: certification extended into the transformer, as an adaptive suffix inside
  the last layer's MLP (`AdaptiveSuffixRuntime`): the final projection (`down_proj`,
  depth 1) or the whole last MLP (depth 2), ahead of the Phase 1C LM head. Enclosures
  are propagated through the reference's own operations under the faithful rounding
  model; a scale-free pairwise certificate works through the final RMSNorm. The masked
  LM-head fallback is the default (decision 0005).
- **Phase 3**: physical selective materialization (decision 0006). A storage core that knows no
  model (`awpmi.storage`, `awpmi.streaming`, `awpmi.materialization`) reads only the requested
  rows of segments from files (direct I/O) or host memory, moves only their bytes to the
  device, caches pages under a budget, and counts every byte, cross-checked against the OS.
  The Phase 1C LM head runs on it unchanged (its segments in a pack, its exact rows read from
  the published checkpoint), bitwise equal to the resident runtime. A model-agnostic
  mixture-of-experts adapter serves transformers' experts modules from the same backend,
  bitwise equal to the resident model (7 architectures in tests, Granite 3.1 1B-A400M in the
  benchmark).
- **Phase 4A**: a real MoE whose experts exceed the GPU (decision 0007). OLMoE-1B-7B (12.9 GB of
  experts, an 8 GB GPU, a 6 GB allocator cap) runs with its experts read from the published,
  split checkpoint through composed segments and executed by a compact experts call (buffers for
  the routed experts only), bit for bit equal to the fully materialized reference executed one
  layer at a time. Reference profiles declare what "exact" is relative to.

## Major components (`src/awpmi/`)

| Module | Role |
| --- | --- |
| `runtime.py` | Declared numerical environment (determinism, FP32 accumulation, no TF32 or reduced-precision reductions). Call it before any CUDA matmul. |
| `models/smollm2.py` | Loads the pinned model, gives access to the LM-head weight, runs the exact prefix (`final_hidden_state`). Phase 2: `suffix_prefix`, the model's own forward interrupted by a pre-hook where the adaptive suffix begins (`down_proj` or the last `mlp`), optionally with a KV cache. |
| `reference.py` | `ReferenceRunner.next_token`: the unmodified HF forward (`logits_to_keep=1`), with the LM-head input captured by a hook; with `intermediates=True`, also every intermediate of the last MLP and the final norm (hooks only observe). |
| `paging/` | `WeightPage` (column block of a weight; Phase 2 adds `layer` and `tensor_role`), `PageIndex` (pages plus bound metadata), `PageBoundMetadata` (row-page L2 norms, rounded up to float32), `InMemoryPageSource` (fetch-counting views). |
| `bounds/floating.py` | γ_n error bounds; directed rounding of float64 values onto a dtype's grid. |
| `bounds/linear.py` | Cauchy–Schwarz block-norm upper bounds; exact float64 page contributions. |
| `bounds/residual.py` | `ReferenceNumerics` (accumulation model of the reference GEMM); `ResidualBounder` (partial logits and remaining pages → intervals on the reference logits); `reference_logit_interval` (centre and radius → output-grid interval, shared by Phase 1A and 1B). |
| `bounds/remainder.py` | Phase 1B: per-row remainder norm metadata (`RowNormBounds`); bound min(‖x‖₂‖h‖₂, ‖x‖_∞‖h‖₁) on a missing remainder row. |
| `bounds/coarse.py` | Phase 1C: error model of the runtime's binary32 coarse pass (`CoarseArithmetic`, γ_{K+2}(2⁻²²) plus a flush-to-zero term) and the free Hölder bound s·limit·‖h‖₁ on a level's absolute mass. |
| `bounds/rounding.py` | Phase 2: rounding models of the reference's outputs. `FAITHFUL` (certified, decision 0001) and `NEAREST_EVEN` (experimental, decision 0005); `RoundingAssumptions` pairs a model for GEMM epilogues and one for elementwise kernels (`CERTIFIED` is faithful for both); `round_enclosure`, `spacing_upper`, rounding-error bounds. |
| `bounds/enclosure.py` | Phase 2: `Enclosure` (lower, upper, provenance), the residual state generalized to any internal tensor (roadmap §2.2); `UnboundedValue` makes the caller fall back. |
| `bounds/operators.py` | Phase 2: verified propagation through the reference operations of the last MLP and the final norm: `linear` (exact or partly unread weights, interval input), `residual_add`, `multiply`, `silu`, `rms_norm` (returns q, n and h). Documented fp32 assumptions: 2⁻¹⁸ relative error per elementwise fp32 operation, flushing below the normal range, γ_{n+2}(2⁻²²) per reduction. |
| `bounds/pairwise.py` | Phase 2: scale-free pairwise certificate through the final RMSNorm. For a winner w and each contender j it lower-bounds (acc_w − acc_j)/q with a box bound and a decomposed bound that keeps the down projection's cancellation (M = W_dᵀΔ, unread columns by Cauchy–Schwarz), then requires a 2-ulp separation of the faithful logits. |
| `state.py` | `ResidualState`: partial logits, lower/upper bounds, materialized and remaining pages. |
| `certificate.py` | `Certificate.check`: CERTIFIED iff lower[w] > max_{j≠w} upper[j] with w = argmax(partial), otherwise UNKNOWN. `TieBreak.LOWEST_INDEX` (decision 0003) also accepts equality against higher-index rows. `certify_columns` and `contenders` are batched forms used by the oracle (row elimination). |
| `decomposition/` | Phase 1B: `RefinementDecomposition` (W = base + refinement levels + exact remainder, per-row int-b levels with float32 scales, exactness checked with TwoSum, byte accounting). Phase 1C adds `packing.py` (per-row little-endian bitstream of biased codes, `CodeLayout.unpack`) and `accounting.py` (4 KiB block accounting, shared with the oracle). |
| `oracle/refinement.py` | Phase 1B diagnostic simulator: `RefinementBatch` (per-state intervals for the `realistic`, `abs_mass` and `ideal` bound tiers), `simulate` (global or row-selective refinement). |
| `stores/refinement.py` | Phase 1C: `PackedRefinementStore`, packed levels, scales, resident remainder norms and the original rows. `read_level` / `read_exact` / `read_fallback` are the only way to weight values; each read is logged with its rows and bytes (`bytes_read`, `reads`). Phase 3: it reads through a `MaterializationBackend`: `from_decomposition` (segments resident on the weight's device, Phase 1C) or `from_pack` (a refinement pack on storage); a level row is one record, the packed codes then the float32 scale (`level_records`); `write_refinement_pack` writes the pack, its exact rows referring to the checkpoint tensor. |
| `storage/` | Phase 3 storage core (no model, no certificate). `layout`: `Segment` (fixed-size rows at an offset of a file; a safetensors tensor is one as it is), Phase 4A `ComposedSegment` (each row the concatenation of byte spans of one or more files: a logical tensor a checkpoint stores as several tensors), `byte_runs`, safetensors headers, typed row views. `fileio`: `PositionedFile` (thread-safe positioned reads; direct = `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), the OS's per-process read counters, process memory, aligned (pinned) host buffers. `store`: `plan_reads` (runs of requested bytes, sorted by file and offset → 4 KiB-aligned extents per file, merged when their blocks touch; each run keeps its output offset; `positions` writes rows into chosen rows of an output), `IOStats`, `PageStore` with `InMemoryPageStore` and `FileBackedPageStore` (reads the planned extents only, of any of its files, with 8 worker threads). `cache`: `PageCache` (device pages under a byte budget, pinned entries, admission freeze) with `LRUPolicy` and `HotnessPolicy`. `pack`: `PackWriter` (v2 adds composed segments and publisher-declared file sha256), `open_pack` (manifest with files, segments and sha256; segments or files re-hashed with direct reads), `Pack.store` / `Pack.load` (the host-memory tier), `SourceFile` (a published checkpoint file in the Hugging Face cache; `published_sha256`), `sha256_file_direct`. |
| `streaming/streamer.py` | Phase 3 transfer layer: `PageStreamer`. Pinned, page-aligned staging slots (two: double buffering), a CUDA copy stream, per-slot events, and the compute stream waiting on a ready event. A file-backed fetch reads its plan in slot-sized pieces and copies only the requested rows' bytes to the device; a host-memory fetch gathers into a slot. Phase 4A: `fetch(…, out, positions)` writes into a caller's buffer (the copy stream first waits for the caller's stream); runs of at least 256 KiB go straight from staging, shorter ones are gathered; scattered destinations get one copy per contiguous range. `prefetch` returns a `Ticket` (consumed and wasted bytes counted). |
| `materialization/` | Phase 3: `MaterializationBackend` (`materialize(segment, rows, out=None)` → device rows: a store in memory on the compute device answers directly; otherwise the cache, then the streamer; with `out`, written into a caller's buffer, cached rows copied first ("hits first"); cache entries in their own CUDA memory pool; every request counted; `report` gathers the storage, transfer and cache counters), `WeightStore` (typed rows of named weights), `ExpertStore` / `ExpertGroup` (a layer's expert-sliced segments; `load`, `fill` into full-shape buffers, Phase 4A `assemble` into compact buffers). |
| `models/moe.py` | MoE adapter, model-agnostic over the experts convention of transformers 5: `find_expert_modules` (a module with `num_experts` and a ≥3-D parameter whose first dimension is that count), `write_expert_pack` (refers to checkpoint tensors with the same bytes, copies the rest), `StreamedExperts` (Phase 3 full-shape slot buffers; Phase 4A `compact=True`: parameters `None` between calls, buffers for the routed experts in ascending order, `num_experts` and `top_k_index` remapped for the call; poison mode; `all_experts` for dense layer streaming; `on_call` with an `ExpertCall`, whose `per_assignment_outputs` re-runs the call per (token, expert) on the same weights), `FullLayerOffload` (the reference for a model whose experts do not fit: host parameters copied whole to the device before each experts call), `move_except_experts`, `routed_experts`, `record_routing`. |
| `models/checkpoint.py` | Phase 4A: a published checkpoint served in place. `checkpoint_sources` (the files of a pinned revision, with their Hub sha256), `expert_sources` (each expert-sliced parameter's checkpoint tensors, derived from the transformers conversion mapping: stacking per expert, optionally concatenation along each expert's first dimension; literal renamings reversed; anything else refused), `expert_index_segments` (composed segments from headers, shapes and dtypes checked), `write_expert_index` (a pack v2 that copies nothing), `load_model_without_experts` (meta skeleton, every non-expert tensor read with direct I/O and cast by transformers' dtype plan, non-persistent buffers from `_init_weights`; experts stay on `meta`). |
| `models/olmoe.py` | Phase 4A OLMoE adapter: `EXPERT_LAYOUT` checked against transformers' mapping, `routers`, `REFERENCE_PROFILE` (BF16, `grouped_mm`, SDPA). |
| `profiles.py` | Phase 4A reference profiles: `ReferenceProfile` (kind, stored weight dtype, compute dtype, kernels, numerics, quantization), `BF16_REFERENCE`, `FP16_REFERENCE`, `native_quantized_reference` (declared, refused by `check_model`); `check_model`, `check_weights`. |
| `cli.py` | `awpmi pack lm-head`, `awpmi pack experts` (roadmap §3.2) and `awpmi pack expert-index` (Phase 4A: the composed-segment index of a split checkpoint, from headers only). |
| `stores/suffix.py` | Phase 2: `MLPStore`, neuron-major pages of the last MLP (gate row, up row, down column of each neuron, one contiguous run each), served only for the stage's adaptive roles, every read logged; resident metadata: down column norms, down row norms (L2, L∞), up row norms. |
| `refinement_head.py` | Phase 1C runtime `RefinementLMHead.run`: binary32 coarse pass over every row → certificate → elimination → float64 refinement of the contenders (the oracle's realistic arithmetic) → exact rows → certificate or fallback (`FallbackMode.MASKED`, the default since decision 0005, self-tested and guarded, else `FULL`). Phase 2: `run_enclosure` runs the same states on an `Enclosure` of h (input spread \|L\|·ρ added to every radius, no fallback, optional row limits), and `HeldReads` lets one token's passes share their reads. Optional `RunTrace` and `StageTimer` (Phase 3 adds `*:read` stages around every store read; timing only). |
| `suffix_runtime.py` | Phase 2 runtime `AdaptiveSuffixRuntime.run` for a `SuffixStage` (lm_head, down, mlp): read a budgeted share of the suffix's neuron pages (largest bound contribution first) → enclosures through the suffix → LM-head pass on the enclosure → pairwise certificate → else exact recomputation of the suffix (the reference's operations and shapes, bitwise) and the Phase 1C LM head on the exact h, reusing the reads. Experimental rounding models report `would_certify` only. |
| `schedulers/` | Deterministic page ordering: `sequential`, `largest_residual`, `bound_per_byte`. They see metadata and state, never page values. |
| `executor.py` | `AdaptiveLMHead.run`: the certify → schedule → materialize → refine loop, plus the exact fallback. |
| `tracing.py`, `config.py` | Environment metadata, source-tree hash, JSONL, digests, `StageTimer`; YAML config. |

Outside the package: `benchmarks/run.py` (experiment driver), `benchmarks/report.py`
(aggregation and gate evaluation), `benchmarks/prompts.py` (deterministic wikitext-2
prompts), `benchmarks/oracle.py` (scheduler-independent ceilings: real-certificate
singleton check plus an ideal magnitude-bound ceiling; diagnostic only),
`configs/smollm2-135m.yaml`, `experiments/phase1/<run>/` (raw results), and `tests/`.
Phase 1B adds `benchmarks/refinement_oracle.py` (driver: reference pass, every decomposition ×
mode × bound tier × tie rule, hard-failure checks), `benchmarks/refinement_report.py`
(aggregation, Phase 1A comparison, decision-gate classification),
`configs/phase1b-refinement.yaml` and `experiments/phase1b/<run>/`.
Phase 1C adds `benchmarks/refinement_runtime.py` (driver: both fallback modes per prompt,
envelope and coarse-arithmetic validation, byte audit, comparison with the Phase 1B records,
stage timing, kernel profile), `benchmarks/refinement_runtime_report.py` (aggregation, oracle
comparison, round-to-nearest what-if, gate), `benchmarks/fallback_study.py` (masked-GEMM
row independence across shapes, dtypes and devices; output-rounding probe),
`configs/phase1c-runtime.yaml` and `experiments/phase1c/<run>/`.
Phase 2 adds `benchmarks/suffix_runtime.py` (driver: reference with every intermediate, the
exact prefix of each stage, the depth-0 LM head, then every stage × budget × rounding model
× row-limit policy, with enclosure, pairwise-bound, exact-suffix, byte and single-read
checks), `benchmarks/suffix_report.py` (curves, bound provenance, gate, per-stage decision),
`configs/phase2-suffix.yaml` and `experiments/phase2/<run>/`.
Phase 3 adds `benchmarks/storage_runtime.py` (the Phase 1C LM head on a tier: the full BF16
head from the drive or from host memory, the resident runtime, the runtime on the drive, on
host memory and with a cached base level; parity with the resident runtime and the Phase 1C
records, storage audits, OS cross-check), `benchmarks/storage_report.py` (gates A and B),
`benchmarks/moe_runtime.py` (Granite MoE: resident reference, then experts from the drive
under several cache budgets and policies, every step compared bit for bit),
`benchmarks/moe_report.py` (gate C, routing statistics), `configs/phase3-storage.yaml`,
`configs/phase3-moe.yaml`, `experiments/phase3/<run>/`, and packs under `packs/` (gitignored,
rebuilt by `awpmi pack`).
Phase 4A adds `benchmarks/olmoe_runtime.py` (two processes: the reference, verified sources,
index audit, residency check and decoding with every layer materialized in turn; then the
streamed model under a device-memory cap, every configuration compared with the reference in
every digest and audited), `benchmarks/olmoe_report.py` (correctness and gates A–D),
`benchmarks/olmoe_profile.py` (un-instrumented timing per stage, kernel trace),
`configs/phase4a-olmoe.yaml` and `experiments/phase4a/<run>/`.

## Data flow (one input)

1. `ReferenceRunner.next_token(ids)` returns the reference token, its BF16 logits, and the
   LM-head input.
2. `final_hidden_state(model, ids)` is the exact prefix. It must be bitwise-equal to the
   captured LM-head input.
3. `AdaptiveLMHead.run(hidden, scheduler)`:
   - computes per-page norms of the hidden state and builds a `ResidualBounder`;
   - loops: `Certificate.check(state)` → if CERTIFIED, stop; else `scheduler.select` →
     `PageSource.get(page)` → `page_contribution` (float64) → `refine_state`;
   - if every page is materialized and the result is still UNKNOWN, the fallback runs
     `F.linear(hidden, cat(pages))`, which is bitwise-equal to the reference.

Phase 1C runtime (`RefinementLMHead.run(h)`), for the same exact LM-head input:

1. `store.read_level(0)` (every row) → decode in chunks → binary32 `torch.mv` → centre
   s·fl(S) and the coarse interval (`bounds/coarse.py`).
2. Loop: `certify_columns` (tie-aware) → if CERTIFIED, stop; else `contenders` → read only
   those rows of the next level (or their original rows in the exact state) → float64
   intervals → intersect with the previous ones.
3. Still UNKNOWN after the exact state: MASKED fallback (the default: full-shape `F.linear`
   on the survivors only, self-tested and guarded) or FULL fallback (read every unread row,
   `F.linear`, bitwise).

Phase 2 runtime (`AdaptiveSuffixRuntime.run`), stages down and mlp:

1. `suffix_prefix(model, ids, boundary)`: the reference forward up to the boundary, all
   positions (residual, MLP input, and for down the MLP activation).
2. Read the budgeted neuron pages from the `MLPStore` (mlp: every gate row first, then up
   rows and down columns; down: down columns), largest bound contribution first.
3. Enclosures at the last position: [gate → SiLU, up → product →] down projection
   (unread columns bounded by row norms) → residual addition → final RMSNorm → h.
4. `RefinementLMHead.run_enclosure(h)`: the Phase 1C states with the input's spread;
   CERTIFIED stops. Else, if the exact state was reached: `pairwise_certificate` on the
   remaining contenders.
5. Still UNKNOWN: read the rest of the suffix, recompute it with the reference's own
   operations on all positions (bitwise the reference's o, y, h), then
   `RefinementLMHead.run(h, held=…)`: Phase 1C on the exact h, sharing the first pass's
   reads.

Phase 3, the Phase 1C runtime on storage (`PackedRefinementStore.from_pack`): every
`read_level` / `read_exact` / `read_fallback` becomes `MaterializationBackend.materialize`:

1. the page cache, if any, serves what it holds (e.g. a pinned base level);
2. the store plans the read: runs of consecutive rows → 4 KiB-aligned extents;
3. the streamer reads the extents into a pinned staging slot (8 threads of positioned reads,
   direct I/O), gathers the requested rows and copies only them to the device on the copy
   stream; the next piece's reads overlap the previous piece's copy;
4. the compute stream waits for the copy; the runtime computes exactly as before.

Phase 3, a MoE model (`StreamedExperts`): the router runs (resident); the experts module's
pre-hook takes `top_k_index`, materializes those experts of that layer (cache, then drive),
writes them into the full-shape slot buffers, and the module's own forward runs.

Phase 4A, compact (`StreamedExperts(compact=True)`), for each experts call:

1. the router runs (resident); the pre-hook takes the routed experts R from `top_k_index`
   (unique, ascending);
2. buffers [|R|, …] are allocated; `ExpertStore.assemble` → `materialize(segment, R, out=…)`
   per parameter: cached experts are copied first, the rest planned as composed spans of the
   checkpoint files, read (direct I/O, 32 MiB pieces) and copied into their rows;
3. the parameters become those buffers, `num_experts` = |R|, and `top_k_index` is replaced by
   each expert's slot; the module's own forward runs;
4. a forward hook releases the buffers: the parameters are `None` again.

## Important constraints

- The certificate bounds the **floating-point reference logits** (after accumulation
  error and output rounding), not just the exact product. See
  `decisions/0001-certificate-targets-floating-point-reference.md`.
- The fallback recomputes the reference operation. See
  `decisions/0002-fallback-recomputes-reference-operation.md`.
- Storage, scheduling, partial execution and certification stay in separate modules.
  Schedulers affect efficiency only.
- Bound metadata is resident and counts against savings. For the LM head
  (49152 × 576, BF16, 56.6 MB) it is 1.8 / 3.5 / 7.1 / 14.2 MB for page widths
  64 / 32 / 16 / 8. Phase 1B remainder norms cost 0.39 MB per non-exact state.
- Decomposition, bounds, certification and simulation are separate modules. A
  decomposition only builds levels and counts bytes. The oracle's `abs_mass` and `ideal`
  tiers use information a runtime cannot have and must never reach a runtime path.
- The Phase 1C runtime touches weights only through its store. Its refined and exact
  states share their radius formulas with the oracle (`remainder_radius`,
  `exact_radius`), so only the coarse state differs from the oracle's arithmetic
  (decision 0004).
- Only the faithful rounding model certifies (decisions 0001 and 0005), for every rounding
  of the suffix: GEMM epilogues and elementwise kernels alike. Round-to-nearest-even is an
  experimental what-if: `AdaptiveSuffixRuntime` refuses it unless `experimental=True`, and
  then never sets `certified`.
- The masked LM-head fallback assumes row independence of the reference GEMM. It is used
  only in the fallback path, behind its self-test and guard, never in a certificate.
- The adaptive suffix starts after the last layer's attention, so the KV cache is written by
  the exact prefix and stays the reference's (roadmap §2.8, Mode A; tested over cached
  greedy generations).
- Layering (decision 0006, enforced by `tests/test_layering.py`): the storage core
  (`storage`, `streaming`, `materialization`) names no model, tensor or router and imports no
  layer above it; the certification layer (`bounds`, `certificate`, `refinement_head`) imports
  no storage. Where bytes come from never changes what is computed: storage-backed runs are
  checked bit for bit against resident ones.
- A file-backed store reads the planned extents and nothing else: a block is read only if it
  holds a requested byte, and only requested rows reach the device. Physical bytes are what
  the reads returned, and they must equal the OS's own per-process counters.
- On NTFS, direct reads of a file that has an active OS cache map are several times slower
  (decision 0006). Packs are verified with direct reads; benchmarks never mix buffered and
  direct access to one file, and wait for cached handles to close after setup.
- The MoE adapter's Phase 3 slot buffers have the experts' full shape (E × expert bytes per
  layer and buffer set). The compact mode (decision 0007) holds only the routed experts, in
  ascending expert order (required for the eager implementation to stay bitwise); a prefill
  that routes every expert still needs one layer of buffers, which a DeepSeek-class layer would
  not fit.
- "Exact" is relative to a declared reference profile (`awpmi.profiles`, decision 0007):
  streamed weights are used in their stored dtype and never converted on the fly.
- A model whose experts do not fit on the device is compared with `FullLayerOffload`, the same
  model with each experts layer materialized whole in turn, in a separate process; residency is
  checked not to change any result.
