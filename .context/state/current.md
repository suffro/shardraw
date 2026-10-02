# Current State

## Current focus

**Phase 2 (certification into the transformer) is complete (2026-10-02). Its gate passes;
the extension is not beneficial on this model.** The full report is
`history/2026-10-02-awpmi-phase2-report.md`; the decisions are in decision 0005.

- **User decisions applied:**
  - The masked LM-head fallback is the default, with its mandatory self-test and per-call
    guard; FULL runs whenever either fails.
  - Only the faithful rounding model certifies. Round-to-nearest-even (RN-even) is an
    experimental what-if that never certifies.
- **What was built:** an adaptive suffix inside the last MLP, ahead of the Phase 1C LM head:
  - depth 1, the final projection (`down_proj`);
  - depth 2, the whole last MLP.

  It has verified operator bounds, a scale-free pairwise certificate through the final
  RMSNorm, neuron pages with counted reads, and an exact suffix recomputation that is bitwise
  the reference.
- **Correctness: PASS** (1000 prompts, 37,000 runs, two runs with identical digests):
  - 0 certified and 0 fallback mismatches;
  - 0 bound violations at any intermediate, pairwise bound or logit;
  - exact prefix, exact suffix and fallbacks bitwise;
  - KV cache exact by construction (Mode A), tested over cached generations.
- **Coverage of the suffix certificate, faithful model.**
  - Depth 1: 27.3% with every page read, 18.8% at 95% of the pages, 4.4% at 90%, 0% at
    ≤ 80%.
  - Depth 2: 0% with row limits; 14.9% without them, but then it reads 1.49× the region.
  - RN-even (what-if): 63.7% at depth 1, 40.4% at depth 2. It held on every intermediate of
    every prompt.
- **Bytes: NOT BENEFICIAL.** No budget saves bytes against reading the suffix in full and
  running Phase 1C (best: −0.0002 of the region).
  - The reference's own BF16 roundings of activations are 88% of the uncertainty.
  - Coverage tracks the top-2 gap: 0% below one logit at depth 1.
- **Depth 0** (the Phase 1C LM head, masked by default): 0.404 of the head's bytes, 90.4%
  certified.

Phases 1A, 1B and 1C are complete. Their reports are in `history/`.

## Recent relevant changes

- New modules:
  - `bounds/rounding.py`, `bounds/enclosure.py`, `bounds/operators.py`,
    `bounds/pairwise.py`;
  - `stores/suffix.py` (`MLPStore`) and `suffix_runtime.py` (`AdaptiveSuffixRuntime`).
- Changed modules:
  - `models/smollm2.py`: `suffix_prefix`, with KV-cache support.
  - `reference.py`: `intermediates=True`.
  - `paging/page.py`: `layer` and `tensor_role`.
  - `refinement_head.py`: masked default; `run_enclosure`; `HeldReads`. Runs on an exact
    input are unchanged: the Phase 1C benchmark gives identical records on 50 prompts.
- Benchmarks:
  - `benchmarks/suffix_runtime.py` and `suffix_report.py`;
  - `configs/phase2-suffix.yaml`, with the gate and decision fixed before the run;
  - raw results in `experiments/phase2/suffix-run1` and `suffix-run2`.
- Decision 0005 is new. Decisions 0001, 0002 and 0004 point to it.
- 323 tests, 82 of them new.

## Next

Phase 3 is **not started**. It needs the user's go-ahead.

1. **Phase 3 (roadmap): real selective storage and streaming.** Recommended adaptive region:
   the LM head (Phase 1C runtime, masked fallback), with the transformer exact. The Phase 2
   suffix runtime remains a tested option, not the default.
2. **Open decisions for the user:**
   - Adopt RN-even for elementwise kernels only? Their BF16 conversion is PyTorch's own
     round-to-nearest-even, a narrower assumption than the cuBLAS epilogue, and it held on
     every prompt. It would revise decisions 0001 and 0005. On this model it would not
     change the byte verdict.
   - Revisit Phase 2 on a model whose last layers weigh more relative to its LM head (e.g.
     a Llama-class model, where one MLP is about 1/3 of the LM head)?
3. **Open levers:**
   - fused kernels (Phase 4);
   - 4 KiB read amplification (Phase 3);
   - for Phase 2, pairwise elimination at coarse precision, and tighter metadata for unread
     columns.

## Blockers

None technical. Phase 3 needs the user's decision to proceed.
