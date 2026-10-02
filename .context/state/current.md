# Current State

## Current focus

**Phase 1C (the runtime of the q6+q4 LM head) is complete (2026-10-02), and its gate
passes.** The full report is `history/2026-10-02-awpmi-phase1c-report.md`; the design and
the fallback decisions are in decision 0004.

- **Correctness: PASS.**
  - 0 certified and 0 fallback mismatches in 2,000 runtime runs: 1000 prompts, each
    run with the full and with the masked fallback.
  - 0 envelope violations and 0 coarse-arithmetic violations.
  - The full fallback is bitwise. The masked fallback is bitwise on the surviving rows
    in 96 of 96 cases.
  - On every run, the bytes in the store's log equal the decomposition's accounting.
  - Two runs give identical digests.
- **Bytes**, as counted by the store, as a mean fraction of the BF16 head:
  - **0.500** with the full fallback (oracle: 0.499);
  - **0.404** with the masked fallback;
  - median 0.400.

  The decision state, the coverage (90.4%) and the contenders after the int4 level match
  the Phase 1B oracle on 1000 of 1000 prompts. The runtime's own binary32 coarse pass
  costs +0.001 of the head.
- **Time: 24.5 ms per token, against 0.21 ms for the resident BF16 GEMV.**
  - The runtime is launch-bound: 610–812 kernels and only 3.6–5.9 ms of GPU time.
  - Decoding the int6 level dominates that GPU time.
  - Saved bytes become saved time only with fused kernels (Phase 4) and a storage tier
    where bytes cost time (Phase 3).
- **Evidence on the fallbacks:**
  - The masked GEMM is bitwise in 2,560 of 2,560 checks, across shapes, dtypes and
    devices.
  - The reference GEMM's epilogue rounds to nearest-even in 49,152 of 49,152 probe
    cases. An RN-even output model would certify 85 of the 96 fallbacks.

Phases 1A and 1B are complete. Their reports are in `history/`.

## Recent relevant changes

- Runtime and its supports:
  - `src/awpmi/refinement_head.py` (the runtime, `RefinementLMHead`);
  - `src/awpmi/stores/refinement.py` (packed store with counted reads);
  - `src/awpmi/bounds/coarse.py` (binary32 coarse-pass error model);
  - `src/awpmi/decomposition/packing.py` and `accounting.py` (4 KiB accounting, moved out
    of the oracle).
- Shared code:
  - `bounds/residual.py` gained `remainder_radius` and `exact_radius`, now shared with
    the oracle. Phase 1B was re-checked and is bitwise unchanged.
  - `tracing.py` gained `StageTimer` and `tensor_digest`.
- Benchmarks:
  - `benchmarks/refinement_runtime.py`, `refinement_runtime_report.py` and
    `fallback_study.py`;
  - `configs/phase1c-runtime.yaml`, with the gate fixed before the run;
  - raw results in `experiments/phase1c/runtime-run1`, its twin `runtime-run2`, and
    `fallback-study`.
- Decision 0004 is new. Decisions 0001 (rounding) and 0002 (fallback) point to it.
- 241 tests, 129 of them new.

## Next

Phase 2 is **not started**. It needs the user's go-ahead.

1. **Two decisions for the user before Phase 2.** A fallback will then recompute a whole
   adaptive suffix, so both matter more than they do now.
   - Make the masked fallback the default? It is guarded and evidenced, but its
     property is not documented.
   - Adopt round-to-nearest-even as the output-rounding model? That would revise
     decision 0001, and raise coverage from 90.4% to 98.9%.
2. **Phase 2 (roadmap): extend certification into the transformer**, starting from the
   final norm and the last MLP. The LM-head runtime becomes the last stage.
3. **Open levers:**
   - a fused decode-and-dot coarse kernel and fewer bound kernels (Phase 4, now
     justified by the profile);
   - a row layout that limits 4 KiB read amplification (Phase 3: 0.578 against 0.500
     logical);
   - a tighter sound remainder bound, which would make a base of 5 bits or fewer viable.

## Blockers

None technical. Phase 2 needs the user's decision to proceed.
