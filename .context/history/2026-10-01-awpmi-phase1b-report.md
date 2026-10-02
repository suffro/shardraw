# AWPMI Phase 1B report — Precision-refinement decomposition search

Date: 2026-10-01 · Follows: `history/2026-10-01-awpmi-phase1-report.md` (Phase 1A) ·
Decision: `decisions/0003-precision-refinement-oracle-and-byte-accounting.md` ·
Status: **complete. The correctness gate passes. A PROMISING decomposition was found:
q6+q4 with row-selective refinement. Phase 1C (its runtime) is not started; Phase 2
and Phase 3 stay blocked.**

## Outcome in brief

Phase 1A needed 98.5% of the LM head to certify the next token. The best Phase 1B
decomposition needs **49.9% on average, counting every byte**:

- resident metadata;
- scales;
- a conservative fallback that reads the whole BF16 head.

Its median is **39.9%**, and the inputs it certifies need **40.2%**. The decomposition,
q6+q4, works as follows:

1. Read an int6 per-row coarse copy of the LM head.
2. Refine only the rows that can still win with an int4 remainder level.
3. Read the original BF16 row only for the 1–5 rows still in contention.

The certificate is unchanged in its semantics (decision 0001) and still targets
the unmodified BF16 reference. Over 96,000 simulated runs there are 0 certified
mismatches, 0 fallback mismatches and 0 envelope violations.

What makes it work is **row-selective refinement**, not refinement levels as such:

- Refining every row is never competitive. Global q6+q4 needs 1.11 of the head.
- Once a coarse pass has eliminated almost all of the 49,152 vocabulary rows,
  refining the few survivors is nearly free.
- The coarse base must be 6 bits or more, because the realistic remainder bound
  (Cauchy–Schwarz or Hölder on per-row norms) is too loose below that.
- A perfect bound would allow a 2–3-bit base, at about 19–28% of the bytes. The
  realistic bound therefore costs about 2× over the ideal ceiling.

## 1. Question

> Can a progressive precision/refinement decomposition of the original BF16 LM-head
> weights reduce the bytes needed to certify the exact next-token argmax of the fully
> materialized BF16 reference?

Phase 1B is an oracle. There is no storage, streaming or runtime. Every candidate is
an exact decomposition W = L₀ + L₁ + … + X of the original weights (decision 0003).

## 2. Setup

| Item | Value |
| --- | --- |
| Model, reference, numerics | As Phase 1A: SmolLM2-135M-Instruct @ `12fd25f7…`, BF16, unmodified HF forward, deterministic, no TF32 or reduced-precision reductions, RTX 4060 Ti |
| Prompts | The same 1000 wikitext-2 prompts as Phase 1A (`experiments/phase1/cs-column-baseline-run1/prompts.jsonl`) |
| LM head | 49152 × 576 BF16 = 56,623,104 bytes, the denominator of every fraction |
| Decompositions | `none` (row norms only), single base `q8`, `q7`, `q6`, `q5`, `q4`, `q3`, `q2`, and multi-level `q4+q4`, `q4+q2`, `q6+q4`, `q2+q2+q4` |
| Levels | Per-row symmetric round-to-nearest int-b of the current remainder, float32 scale, bit-packed; exact in float64 (TwoSum-checked) |
| Final state | The original BF16 row, 1152 bytes, charged in full |
| Modes | `global` (every row advances) and `row_selective` (only rows that can still win advance) |
| Bound tiers | `realistic` (runtime-computable), `abs_mass` (best sign-agnostic, diagnostic), `ideal` (true remainder contribution, diagnostic) |
| Tie rules | `strict` and `lowest_index` (`torch.argmax` semantics) |
| Runs | 1000 prompts × 12 decompositions × 2 modes × 4 (tier, tie) pairs = 96,000 simulated runs, executed twice |

## 3. Method

- **Interval of row j in state s** (levels 0..s read): the centre is
  Σ_{l≤s} fl(L_l[j]·h). The radius is

  miss + γ_acc·A + γ₆₄·M,

  where miss bounds the remainder's contribution and A, M bound the absolute masses
  behind the reference accumulator error and the float64 error. The interval is then
  directed-rounded onto the BF16 grid (`reference_logit_interval`, shared with
  Phase 1A).
- **The realistic miss** is min(‖X_s[j]‖₂‖h‖₂, ‖X_s[j]‖_∞‖h‖₁). The norms are float32
  metadata rounded up, at 8 bytes per row per non-exact state, and the same
  quantity bounds the remainder's share of A.
- **The exact state** uses A = Σ_k|W_jk h_k|, which is tighter than Phase 1A's
  per-page Cauchy–Schwarz value. It is validated by the envelope check: every
  reference logit lies in every interval of every state, tier and decomposition.
- **Elimination.** A row is dropped once another row is provably at least as large
  and wins the comparison. Intervals of successive states are intersected. The
  winner is the row with the largest lower bound.
- **Effective bytes**:
  - all bound metadata;
  - the base level for every row (codes plus scales);
  - every refinement row and exact row read;
  - for a fallback, every BF16 row not yet read.
- **Secondary measures**: 4 KiB-block I/O, and a *masked fallback* that recomputes
  only the surviving rows in a full-shape GEMM. The masked fallback is checked
  bitwise against the reference logits on every row-selective fallback.
- **Hard failures** stop the run:
  - prefix or fallback not bitwise-equal;
  - an envelope violation;
  - a certified or fallback mismatch;
  - an inexact decomposition.

## 4. Correctness

Source: `experiments/phase1b/refinement-run1/summary.md` (machine-readable:
`experiments/phase1b/refinement-run1/summary.json`; raw:
`experiments/phase1b/refinement-run1/records.jsonl.gz`).

| Criterion | Result |
| --- | --- |
| Certified mismatches = 0 | PASS: 0 in 96,000 runs |
| Fallback mismatches = 0 | PASS: 0 |
| Reference prefix and fallback `F.linear(h, W)` bitwise-equal | PASS: 1000 of 1000 |
| Envelope violations = 0 | PASS: 0 over 1000 prompts × 12 decompositions × 3 tiers × every state × 49,152 logits |
| Decompositions exact | PASS: all 12 (TwoSum, plus a rational-arithmetic test) |
| Reproducible | PASS: run1 and run2 (same source-tree hash `7aadc9ba…`) have identical records, validation, reference and decomposition digests |
| Masked-fallback diagnostic | 4,980 checks, 0 mismatches |

Tests: the full suite passes, 112 tests including 51 new ones, on CPU and CUDA.
The new tests cover:

- reconstruction exactness and metadata soundness in rational arithmetic;
- remainder bounds, including the cases where Cauchy–Schwarz and Hölder are tight;
- `torch.argmax` lowest-index semantics on a vocabulary-sized BF16 vector;
- synthetic BF16 ties won and lost by the lower index;
- tie-rule soundness on 400 random tied systems;
- oracle enclosure and decision parity on random systems;
- a slow per-input simulator that matches the vectorized one;
- 4 KiB accounting against brute force.

Guards were confirmed to fail: weakening the tie-aware certificate or the tie
elimination made the tie tests and the end-to-end oracle test fail, and an inexact
level is refused. After the refactor, the Phase 1A engine reproduces run1 bitwise:
3,200 fields over 25 prompts × 4 widths × 3 schedulers, margins included.

## 5. Results

### 5.1 Primary configuration (realistic bound, `lowest_index`, row-selective)

Coverage is 90.4% for every decomposition. It is decided at the exact state, so it
is a property of the reference, not of the decomposition (§5.4).

| decomposition | mean | median | p90 | p95 | masked-fallback mean | 4 KiB I/O mean | contenders after base (median / p90) | ideal-tier mean | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **q6+q4** | **0.499** | **0.399** | 0.468 | 1.397 | **0.403** | 0.573 | 980 / 5092 | 0.488 | **PROMISING** |
| q6 | 0.517 | 0.411 | 0.682 | 1.385 | 0.426 | 0.628 | 980 / 5092 | 0.481 | PROMISING |
| q7 | 0.545 | 0.448 | 0.459 | 1.448 | 0.449 | 0.548 | 21 / 132 | 0.544 | NOT (decomposition-limited) |
| q8 | 0.607 | 0.511 | 0.511 | 1.510 | 0.511 | 0.607 | 5 / 18 | 0.606 | NOT (decomposition-limited) |
| q4+q4 | 0.617 | 0.521 | 0.523 | 1.521 | 0.521 | 0.618 | all / all | 0.363 | NOT (bound-limited) |
| q2+q2+q4 | 0.689 | 0.579 | 0.953 | 1.531 | 0.599 | 0.856 | all / all | 0.284 | NOT (bound-limited) |
| q5 | 1.124 | 1.180 | 1.322 | 1.323 | 1.098 | 1.380 | 41,068 / 47,555 | 0.419 | NOT (bound-limited) |
| q4, q3, q2 | 1.26 / 1.20 / 1.14 | = mean | = mean | = mean | = mean | ≈ mean | all / all | 0.357 / 0.299 / 0.491 | NOT (bound-limited) |
| q4+q2 | 1.257 | 1.312 | 1.396 | 1.396 | 1.237 | 1.469 | all / all | 0.363 | NOT (bound-limited) |
| none | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | all / all | 1.007 | NOT (decomposition-limited) |

"all" means all 49,152 rows remain. The p95 column of 1.4–1.5 is the conservative
fallback: 9.6% of inputs read the full BF16 head on top of the coarse pass.

Byte breakdown of q6+q4, as a mean fraction of the head:

| metadata | int6 base codes | base scales | int4 refinement codes | refinement scales | exact BF16 rows | fallback rows | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0139 | 0.3750 | 0.0035 | 0.0101 | 0.0001 | 0.0000 | 0.0960 | 0.499 |

The int6 pass leaves a median of 980 contenders, with a mean of 2,000 rows refined
at int4. After the int4 refinement the median is 1–3 contenders, so on average only
one BF16 row is read. 53.2% of inputs certify right after the int4 refinement, and
90.4% at the exact state. Without the int4 level (q6), about 2,000 BF16 rows are
read per input, which is +4% of the head.

### 5.2 Bound tiers: the bound sets the usable base precision

Median contenders after the base pass (row-selective, `lowest_index`):

| base | realistic | abs_mass (sign-agnostic) | ideal |
| --- | --- | --- | --- |
| q8 | 5 | 2 | 1 |
| q7 | 21 | 5 | 1 |
| q6 | 980 | 24 | 1 |
| q5 | 41,068 | 1,617 | 2 |
| q4 | all | 44,124 | 4 |
| q3 | all | all | 207 |
| q2 | all | all | 13,793 |

For a uniform rounding remainder, the realistic (Hölder) bound is about 2× the
sign-agnostic one. Any sign-agnostic bound still ignores that the remainder's terms
partly cancel, and that cancellation is the whole gap to the ideal tier. Each tier
therefore admits a different lowest usable base:

| Tier | Lowest usable base | Mean fraction |
| --- | --- | --- |
| realistic | 6 bits | 0.40 masked, 0.50 conservative |
| abs_mass | 5 bits | 0.380 masked, 0.470 conservative |
| ideal | 2–3 bits | 0.188 (q2+q2+q4) and 0.204 (q3) masked; 0.284 and 0.299 conservative |

At 6 bits the realistic bound keeps 93–98% of the ideal saving. Below 6 bits it
keeps none.

### 5.3 Global versus row-selective refinement

The realistic bound almost never certifies after a global coarse pass. Even q8
certifies 17.3% of inputs at its base, q7 3.7%, and q6 or less 0%. Every other input
then reads the full BF16 head on top of the coarse copy. Global refinement therefore
costs more than the BF16 head:

| Decomposition | Global, realistic | Row-selective, realistic |
| --- | --- | --- |
| q8 | 1.34 | 0.607 |
| q6 | 1.39 | 0.517 |
| q6+q4 | 1.11 | 0.499 |
| q4+q4 | 1.41 | 0.617 |

Row-selective refinement saves 53–63% over global refinement for every decomposition
whose base eliminates rows. Both modes certify at exactly the same state (asserted
by tests), so the saving is pure I/O.

**Multi-level refinement pays only after elimination.** q4+q4 is no better than q8:
its first level eliminates nothing, so the int4 remainder must be read for every row.
It then certifies 11.1% of inputs after that second level, against 17.3% for q8's
single level.
q6+q4 beats q6 only because its second level is applied to about 4% of the rows.

### 5.4 Ties and fallbacks

- **The tie-aware certificate raises coverage from 87.3% to 90.4%** (+31 inputs, 9 of
  them exact BF16 ties). It is sound by `torch.argmax` semantics, and the masked and
  full fallbacks agree with it.
- **Coverage is higher than in Phase 1A even without it** (strict: 87.3% versus
  86.6%), because the exact state now uses the exact absolute mass.
- **The 96 remaining fallbacks** all have a reference top-2 gap ≤ 0.25 (≤ 2 BF16 ulps):
  29 exact ties, 52 at one ulp of 0.125, 8 at 0.0625 and 7 at 0.25. Even exact logits
  cannot separate them under the faithful-rounding model of decision 0001. That
  model is the floor for every decomposition.
- **Fallbacks dominate the mean.** In the conservative accounting a fallback costs
  1.40 for q6+q4, so 9.6% of inputs add 0.096 to the mean.
- **The masked fallback** recomputes only the surviving rows (zeros elsewhere) in a
  full-shape `F.linear`. It reproduced the surviving reference logits bitwise in all
  4,980 checks. It would cut the q6+q4 mean to 0.403, but it relies on cuBLAS
  choosing the same kernel whatever the values are. That is an empirical property,
  not a documented one, so it is reported but not counted.

## 6. Comparison with Phase 1A

Every figure below is a mean fraction of the BF16 LM-head bytes.

| Configuration | Pages or levels only | Effective, all metadata included |
| --- | --- | --- |
| Phase 1A best ordering (width 8, `bound_per_byte`) | **0.985** | 1.235 |
| Phase 1A lowest effective (width 64) | 1.000 | 1.031 |
| **Phase 1B q6+q4, row-selective** | 0.389 (both levels, scales included) | **0.499** (conservative fallback) / **0.403** (masked fallback) |

Against the 0.985 baseline, q6+q4 reads **49% fewer bytes** in the conservative
accounting and 59% fewer with a masked fallback. Against Phase 1A's lowest effective
cost (1.031) the reduction is 52% and 61%.

## 7. Decision gate

Thresholds were fixed in `configs/phase1b-refinement.yaml` before the full run. They
are evaluated on the primary configuration with the conservative fallback.

| Verdict | Rule | Decompositions |
| --- | --- | --- |
| PROMISING | realistic mean ≤ 0.60, and the realistic saving is at least half the ideal saving | **q6+q4** (0.499; 98% of its ideal saving), **q6** (0.517; 93%) |
| NOT PROMISING, decomposition-limited | ideal mean > 0.5 | q8 (0.606), q7 (0.544), none (1.007) |
| NOT PROMISING, bound-limited | otherwise | q5, q4, q3, q2, q4+q4, q4+q2, q2+q2+q4 |

q7 and q8 are classed as decomposition-limited because their coarse copies alone
cost 0.44–0.50, plus the fallback floor. They certify reliably but cannot beat q6.
The bound-limited decompositions all have ideal ceilings of 0.28–0.49, so a tighter
*sound* remainder bound would make them attractive. That is the main lever left (§8).

## 8. Recommendation for Phase 1C

**Implement q6+q4 with row-selective refinement and the tie-aware certificate as the
Phase 1C runtime for the LM head.**

**Why this decomposition:**

- It is the only decomposition whose realistic bound is both large in saving and
  close to its own ideal (98%).
- Its bytes are almost all a single sequential coarse read (0.375 + scales).
- Its refinement read is small (~0.01) and its exact-row read is negligible.
- It leaves the reference, the accumulation model, the grid rounding and the
  bitwise fallback untouched.

**Expected savings on this model and prompt set**, as a mean fraction of the BF16
LM-head bytes:

| Accounting | Mean |
| --- | --- |
| Conservative fallback | 0.50 |
| Median input | 0.40 |
| Certified inputs | 0.40 |
| With an adopted masked fallback | 0.40 |
| With 4 KiB block granularity (conservative) | 0.57 |

**Storage:** 93.2 MB, which is 1.65× the BF16 head:

- int6 level: 21.4 MB;
- int4 level: 14.3 MB;
- remainder norms: 0.79 MB;
- the original BF16 head: 56.6 MB.

**Remaining risks:**

1. **Fallback cost.** 9.6% of inputs fall back and dominate the mean. Two options
   need a decision:
   - adopt the masked fallback, after a robustness study across shapes, GPUs and
     library versions, and with a guard;
   - tighten the output-rounding model (for example, assume round-to-nearest in the
     cuBLAS epilogue), which would revise decision 0001.
2. **I/O granularity.** About 2,000 scattered int4 rows per input cost 0.07 of the
   head in 4 KiB blocks, against 0.01 logically. A row layout that clusters likely
   contenders, or a larger refinement unit, is needed before streaming.
3. **Runtime arithmetic.**
   - The oracle computes centres and masses in float64. A real coarse pass in
     FP32/BF16 needs its own error term and a new envelope validation.
   - The abs-mass pass doubles the coarse compute.
4. **Bound tightness.** The realistic bound is the limiting factor below 6 bits.
   With a bound as tight as the sign-agnostic tier, q5 would reach 0.47
   (conservative) or 0.38 (masked). Possible tighter sound
   bounds include block-wise norms and outlier-column splits, but each must pay for
   its metadata.
5. **Generality.** One 135M model, a 576-wide LM head, 1000 raw wikitext prompts, one
   GPU. Wider heads change the Cauchy–Schwarz and Hölder looseness, and larger
   vocabularies change the elimination statistics.

Phase 1C should keep the Phase 1A gate:

- zero mismatches;
- a bitwise fallback;
- reproducibility;
- an envelope check that includes the runtime's own coarse-pass arithmetic.

It should then measure real bytes and time against these oracle numbers.

## 9. Limitations

- The results are oracle results. Bytes are logical, apart from the 4 KiB secondary
  measure, and no time is measured.
- The final state charges the full BF16 row. A compact exact remainder could lower
  the refinement cost, but it barely matters here, because about one exact row is
  read per input.
- The accumulation model (u_acc = 2⁻²²) is the Phase 1A assumption, now used with the
  exact absolute mass. It is validated empirically, with 0 envelope violations, but
  is not derived from vendor documentation.
- `decision_state` in the raw records is a level name, which is ambiguous for
  repeated levels (`q4+q4`). `decision_state_index` is authoritative, and the report
  uses it.
- After both runs, `benchmarks/refinement_report.py` was fixed to label states by
  position. It only reads the raw data, so the current source-tree hash differs
  from the runs' `7aadc9ba…` without any change to the results.

## 10. Reproduction

```bash
uv sync
uv run pytest                                                                         # 112 tests
uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/<name>     # about 3.5 min on an RTX 4060 Ti
uv run python benchmarks/refinement_report.py experiments/phase1b/<name> --compare experiments/phase1b/refinement-run1
```

The run is reproduced if `digest.json` matches run1: records `9574d4a6…`,
validation `c660689a…`, reference `2f668b47…`, decompositions `77b63b53…`.
