# 0004 — Phase 1C runtime: packed store, binary32 coarse pass, fallback modes

Status: accepted (Phase 1C, 2026-10-02)

## Context

Phase 1B found q6+q4 PROMISING: row-selective refinement with the tie-aware certificate
needs 0.499 of the BF16 LM-head bytes, or 0.403 with a masked fallback (decision 0003,
`history/2026-10-01-awpmi-phase1b-report.md`). Those were oracle numbers: float64 over
every row, no storage, no time.

Phase 1C turns q6+q4 into a runtime. It has to fix four things:

- how the levels are stored and how bytes are counted;
- the arithmetic of the coarse pass and its error term;
- the fallback;
- whether to tighten the output-rounding model.

The last two are the two levers Phase 1B left for its 9.6% of fallbacks.

## Decision

1. **Packed store with counted reads** (`awpmi.stores.refinement.PackedRefinementStore`).
   - Each level row is a little-endian bitstream of biased codes, ceil(K·b/8) bytes. That
     is exactly the payload decision 0003 charges: 432 bytes for int6 and 288 for int4.
   - Each row also has a float32 scale. The remainder norms are resident, and the
     original BF16 rows are kept.
   - Packing is checked to be lossless when the store is built.
   - The runtime reads weight values only through `read_level`, `read_exact` and
     `read_fallback`. Each read is logged with its rows and bytes.
   - A run's effective bytes are the store's log. Every benchmark asserts that they equal
     the decomposition's `materialized_bytes` for the same rows.
   - Phase 1C keeps the store resident, so a read is a gather. Phase 3 makes reads
     physical. (Done in Phase 3, decision 0006: the same reads from a pack on the drive,
     each level row stored as one record, payload then scale. The 4 KiB model of this
     decision matched the measured drive bytes: 0.468 of the head plus 0.014 of resident
     metadata, against 0.482 predicted.)
2. **Binary32 coarse pass** (`awpmi.bounds.coarse`, `coarse_matvec`).
   - The base level of every row is decoded in chunks and multiplied by h with
     `torch.mv` in binary32. The centre c = s·fl(S) is exact in float64.
   - Error model: |c − L·h| ≤ γ_{K+2}(2⁻²²)·B + s·τ, any summation order. The unit
     roundoff is 4u₃₂, as for the reference accumulator. τ covers flush-to-zero.
   - B = s·limit·‖h‖₁ (Hölder, since |q| ≤ limit) bounds the level's absolute mass. It
     needs no metadata and no second pass. B also replaces the exact mass that the oracle
     uses in the accumulation and float64 terms.
   - Cost, measured over 1000 prompts: +11.5% contenders after the base (mean ratio
     1.115), which is +0.001 of the head. The largest binary32 error observed is
     2.1·10⁻⁴ of its bound.
3. **Refined and exact states reproduce the oracle.**
   - The contenders are recomputed in float64 exactly as in the oracle's realistic tier,
     with shared `remainder_radius` and `exact_radius`. The base level of these rows is
     taken from bytes already read.
   - Only the coarse state therefore differs from the oracle. Measured: after the int4
     level and after the exact state, the contenders are identical on 1000 of 1000
     prompts. The decision state and coverage are identical too.
4. **Validation on every prompt.**
   - Envelope check: every state's own interval, including the coarse one, must contain
     the reference logits.
   - Direct check of the binary32 error: |c − fl₆₄(L·h)| ≤ its bound + γ₆₄·M.
   - Both are hard failures.
5. **FULL fallback stays the default.** It reads every unread BF16 row and applies the
   reference `F.linear`, bitwise, as in decision 0002. The gate and the primary numbers
   use it. (Superseded from Phase 2 on by decision 0005: MASKED is the default, FULL the
   canonical fallback behind it.)
6. **MASKED fallback is adopted as an opt-in runtime mode** (`FallbackMode.MASKED`).
   (From Phase 2 on it is the default, decision 0005; a failed self-test now switches the
   head to FULL instead of refusing to build it.)
   - What it does: a full-shape `F.linear` on the surviving rows, with zeros elsewhere,
     and the argmax taken over the survivors. The survivors' rows were already read in
     the exact state, so it reads no extra bytes.
   - What it assumes: each output row of the reference GEMM depends only on its own
     weight row. That property is not documented.
   - Platform self-test: 16 random trials when the head is built. The head refuses to
     build if any trial fails.
   - Per-call guard: every surviving masked logit must lie in its certified interval;
     otherwise FULL runs.
   - Evidence from `benchmarks/fallback_study.py`: 2,560 of 2,560 bitwise, over 6
     LM-head shapes (SmolLM2, Qwen2.5, Llama) and 2 odd shapes, BF16 and FP32, CUDA and
     CPU (CPU up to 64 M weights), including the top rows of real logits.
   - Evidence from the benchmark: 96 of 96 masked fallbacks bitwise on the surviving
     rows, in both runs, with 0 guard trips.
   - Effect: mean 0.404 instead of 0.500, and p95 0.433 instead of 1.397.
7. **Output-rounding model: decision 0001's faithful model is kept.**
   - The probe shows round-to-nearest-even in the reference GEMM's epilogue, in
     49,152 of 49,152 exact-accumulator cases on CUDA and on CPU. That includes 16,384
     exact midpoints, where rounding down or up matches only about half the time.
   - An RN-even model would certify 85 of the 96 fallbacks (none with a wrong token).
     Coverage would go from 90.4% to 98.9%.
   - It is not adopted now, for two reasons. It would put an undocumented library
     property into the certified path, whereas the masked fallback only touches the
     fallback path, behind a guard. And the masked fallback already captures the byte
     gain.
   - Revisit it when coverage itself matters, for example in Phase 2, where a fallback
     recomputes a whole adaptive suffix. Re-run the probe on any new platform.
8. **Time is measured, not gated.**
   - Per-stage wall time, with the device synchronized at stage boundaries.
   - The reference LM-head GEMV, for comparison.
   - A kernel-level profile: device time and kernel launches.
   - Phase 1C has no time threshold, because no custom kernel is allowed before
     profiling (roadmap §0.7). These measurements are that profile.
9. **Gate, fixed in `configs/phase1c-runtime.yaml` before the full run.**
   - Correctness:
     - zero certified and zero fallback mismatches in both modes;
     - a bitwise reference prefix and full fallback;
     - masked fallbacks bitwise on the survivors, a clean self-test and no guard trips;
     - zero envelope violations and zero coarse-arithmetic violations;
     - a store consistent with the Phase 1B decomposition (same level digest);
     - identical digests across two runs.
   - Efficiency: primary mean ≤ 0.60, and at most 0.01 above the oracle's primary mean.

## Rejected

- *Keeping decoded codes resident.* This would make per-token bytes meaningless: the
  decoded int6 level is as large as the BF16 head or larger.
- *A coarse pass on BF16 tensor cores* (`torch.mm(..., out_dtype=float32)`). It is
  0.22 ms instead of 0.43 ms, but it would rest the runtime's own arithmetic on the
  empirical tensor-core accumulation model. The GEMV is not the bottleneck anyway:
  decoding is.
- *An exact absolute-mass pass* (|q|·|h| over every row) or *per-row ‖q‖₂ metadata*.
  The first costs a second pass over the codes. The second costs 0.2 MB per level, more
  than the +0.001 of the head it would save.
- *A compact exact remainder* instead of the original rows. Only 1.4 rows per input
  reach the exact state, so it would save almost nothing.
- *Making MASKED the default.* Its assumption is evidenced and guarded, but not proven.
  The user should make that call. **The user made it for Phase 2 (decision 0005):** MASKED
  is the default, with its self-test and guard, and FULL runs whenever either fails.
