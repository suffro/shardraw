# Current State

## Current focus

**Phase 1B (precision-refinement decomposition search) is complete (2026-10-01).** It
was an oracle only. The full report is `history/2026-10-01-awpmi-phase1b-report.md`.

- **Correctness gate: PASS.** 0 certified mismatches and 0 fallback mismatches in 96,000
  simulated runs. There are 0 envelope violations, every decomposition is exact, and two
  runs produce identical digests.
- **Outcome: PROMISING.** q6+q4 works as follows:
  1. read an int6 per-row coarse copy of the LM head;
  2. refine only the rows that can still win with an int4 remainder level;
  3. read the original BF16 row only for the 1–5 rows left.

  With the tie-aware certificate, it certifies with a mean of **0.499** of the BF16
  LM-head bytes. That includes all metadata and a conservative full fallback. The
  median is 0.399, and the mean is 0.403 with a masked fallback. Phase 1A needed 0.985.
- **What matters:**
  - row-selective refinement, which saves 53–63% over refining every row;
  - a base of at least 6 bits, because the realistic Cauchy–Schwarz/Hölder remainder
    bound is too loose below that.

  The ideal-bound ceiling is about 0.19–0.28 with a 2–3-bit base.
- **Tie-aware certificate** (`TieBreak.LOWEST_INDEX`, decision 0003): coverage goes from
  87.3% to 90.4%. The remaining 9.6% of fallbacks are within 2 BF16 ulps and dominate the
  mean byte cost.

Phase 1A (input-column pages) is complete and superseded as a decomposition. Its
correctness machinery is reused unchanged. See `history/2026-10-01-awpmi-phase1-report.md`.

## Recent relevant changes

- `src/awpmi/decomposition/` (exact refinement decompositions, byte accounting),
  `src/awpmi/bounds/remainder.py` (remainder bounds), `src/awpmi/oracle/refinement.py`
  (simulator). `certificate.py` gained `TieBreak`, `certify_columns` and `contenders`.
  `bounds/residual.py` gained `reference_logit_interval`, a refactor that left Phase 1A
  bitwise unchanged.
- `benchmarks/refinement_oracle.py`, `benchmarks/refinement_report.py`,
  `configs/phase1b-refinement.yaml`. Raw results are in
  `experiments/phase1b/refinement-run1` and its twin
  `experiments/phase1b/refinement-run2`.
- Decision 0003 covers the oracle methodology, byte accounting, tie-aware certificate
  and gate thresholds. Decision 0001 points to it for ties.
- 112 tests, 51 of them new.

## Next

Phase 1C is **not started**. It needs the user's go-ahead.

1. **Phase 1C:** a real runtime for the q6+q4 LM head with row-selective refinement and
   the tie-aware certificate. It needs:
   - packed int6/int4 levels, float32 scales and resident remainder norms;
   - a coarse pass with its own arithmetic error term, re-validated by the envelope
     check;
   - elimination, then refinement, then exact rows, then the certificate or fallback.

   Then measure real bytes and time against the oracle.
2. **Decisions to take with Phase 1C:**
   - adopt the masked fallback (4,980/4,980 bitwise in the oracle, but empirical); and/or
   - tighten the output-rounding model.

   Either one attacks the 9.6% fallbacks.
3. **Open levers:**
   - a tighter sound remainder bound, which would make a 5-bit or lower base viable;
   - a row layout that limits 4 KiB read amplification (0.57 against 0.50 logical).

Phase 2 (transformer suffix) and Phase 3 (streaming) stay blocked until Phase 1C
validates the runtime.

## Blockers

None technical. Phase 1C needs the user's decision to proceed.
