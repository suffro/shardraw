# Current State

## Current focus

**Phase 1 of the AWPMI roadmap is complete (2026-10-01).** The full report is
`history/2026-10-01-awpmi-phase1-report.md`.

- **Correctness gate: PASS.** 0 certified mismatches and 0 fallback mismatches in 12,000
  runs on 1000 wikitext-2 prompts. The fallback is bitwise-equal to the reference, there
  are 0 envelope violations, and two runs produce identical digests.
- **Scientific outcome: negative for CS bounds on input-column pages.** Certification
  happens almost only at full materialization: the best configuration (width 8,
  `bound_per_byte`) materializes 98.5% of the LM head on average.
- **Diagnosis** (oracle): the schedulers are near the best possible ordering. Even a
  perfect per-page bound would leave 83–92% materialized. The column decomposition is
  the binding constraint.
- **Verdict:** do not start Phase 2 or Phase 3. Stay in Phase 1 and change the page
  decomposition.

## Recent relevant changes

- Phase 1 implementation: `src/awpmi/`, `tests/` (61 tests), `benchmarks/` (run,
  report, oracle), `configs/smollm2-135m.yaml`, `pyproject.toml` and `uv.lock`.
- Raw results: `experiments/phase1/cs-column-baseline-run1`, its twin
  `experiments/phase1/cs-column-baseline-run2`, and
  `experiments/phase1/cs-column-baseline-oracle`.
- Decisions 0001 (the certificate targets the floating-point reference: declared
  numerics, an accumulation-error term, directed rounding onto the BF16 grid) and 0002
  (the fallback recomputes the reference operation bitwise).

## Next

This needs a decision, because it changes the roadmap §1.4 page definition:

1. **Precision-refinement pages** (a coarse per-row-scaled int8 or int4 copy of W plus
   residual pages, with residual-norm metadata), combined with **vocabulary-row
   refinement** of only the contending rows.
2. Measure the ceiling of the new decomposition with the oracle methodology before
   implementing it, and count all resident pages and metadata bytes.
3. Optional: a tie-aware certificate to recover part of the 13.5% near-tie fallbacks.

## Blockers

None technical. Advancing past Phase 1 is blocked by the gate's scientific condition,
as designed.
