# AWPMI Phase 2 report — Certification into the transformer (adaptive suffix in the last MLP)

Date: 2026-10-02 · Follows: `history/2026-10-02-awpmi-phase1c-report.md` (Phase 1C) ·
Decision: `decisions/0005-phase2-adaptive-suffix-rounding-and-fallback.md` ·
Status: **complete. The gate passes; the extension is not beneficial on this model.**

## Outcome in brief

Phase 2 extends certification beyond the LM head, into the last layer's MLP:

- **Depth 1, the final projection:** the last `down_proj` is read only in part.
- **Depth 2, the last MLP:** `gate_proj`, `up_proj` and `down_proj` are read only in part.

The reference's values are bounded through its own BF16 operations, the final RMSNorm and
the Phase 1C LM head. A scale-free pairwise certificate then decides through the norm.
Only the faithful rounding model certifies; round-to-nearest-even is recorded as a what-if.
The masked LM-head fallback is now the default.

| Over 1000 prompts | Result |
| --- | --- |
| Certified and fallback mismatches | **0** in 37,000 runs (both runs) |
| Bound violations at any intermediate, pairwise bound, logit | **0** |
| Exact prefix, exact suffix recomputation, fallbacks bitwise | 1000/1000, 11,073/11,073, 1,248/1,248 |
| Certified with part of the suffix unread | 232 decisions (depth 1, budgets 0.95 and 0.9) |
| Suffix certificate coverage, depth 1, faithful | 27.3% (all pages read) · 18.8% (95%) · 4.4% (90%) · 0% (≤ 80%) |
| Suffix certificate coverage, depth 2, faithful | 0% with row limits · 14.9% without them, at 1.49× the region's bytes |
| Same under RN-even (what-if) | depth 1: 63.7% · 36.0% · 10.3% · 0%; depth 2: 34.8% / 40.4% |
| Best mean saving against reading the suffix in full + Phase 1C | **−0.0002** of the region (depth 1 and 2): NOT BENEFICIAL |
| Reproducible | run1 = run2 on all four digests, same source tree `ed5cfd7a…` |

The obstacle is not weight bytes. Under the faithful model every BF16 rounding of a value
that is not known exactly can move it by up to one ulp. Behind the final norm, the LM head
turns those roundings into an uncertainty comparable to the median top-2 logit gap. With
every suffix weight read, the final projection certifies only tokens whose gap exceeds
about one logit. The certificate needs nearly all suffix pages, while reading them in full
allows the exact recomputation that Phase 1C then certifies 90.4% of the time.

## 1. Question

> Can the certificate reach into the transformer, deciding the reference token while part of
> the last layers' weights is still unread, and does that save bytes?

## 2. Setup

| Item | Value |
| --- | --- |
| Model, reference, numerics | As in Phase 1: SmolLM2-135M-Instruct @ `12fd25f7…`, BF16, unmodified HF forward, deterministic flags, u_acc = 2⁻²² |
| Hardware | RTX 4060 Ti 8 GB, Windows 11, torch 2.14.1+cu130 |
| Prompts | The same 1000 wikitext-2 prompts (`experiments/phase1/cs-column-baseline-run1/prompts.jsonl`) |
| LM head | Phase 1C q6+q4 runtime, `lowest_index` ties, masked fallback (self-test 16/16 passed) |
| Stages | `lm_head` (depth 0); `down`, the last `down_proj` (depth 1, 1.77 MB); `mlp`, the last MLP (depth 2, 5.31 MB) |
| Region | BF16 bytes of the stage's adaptive roles + LM head: 58.39 MB (depth 1), 61.93 MB (depth 2) |
| Budgets | 1.0, 0.95, 0.9, 0.8, 0.5 of the neuron pages |
| Rounding models | faithful (certified); RN-even in elementwise kernels; RN-even everywhere (what-ifs) |
| Policies | "limited" (row limits 32,768 int4 / 256 exact rows on the LM-head pass of an enclosure) at every budget; "unlimited" at budget 1.0 |
| Runs | per prompt: 1 depth-0 run + 2 stages × 6 budget-policy cells × 3 models = 37 runs; twice |

Feasibility probe, run before any design (200 prompts, exact suffix weights):

- With per-logit intervals, the faithful model certified 17% of tokens at depth 1 and 0%
  at depth 2.
- A pairwise certificate raised that to 30.5% and 11%.

This fixed the design: pairwise certification, whole neuron pages, no precision levels.

## 3. Runtime

For one token, stages down and mlp (`src/awpmi/suffix_runtime.py`):

1. **Exact prefix.** The model's own forward, stopped by a pre-hook where the suffix
   begins (`suffix_prefix`). The adaptive weights are never touched, and the captured
   tensors equal the reference's bitwise.
2. **Pages.** Read the budgeted share of the suffix's neuron pages (`MLPStore`: gate row, up
   row or down column of a neuron, 1,152 bytes each), largest bound contribution first.
   At depth 2 every gate row is read first.
3. **Enclosures.** Propagate enclosures through the reference operations
   (`awpmi.bounds.operators`): gate → SiLU, up → product → down projection → residual
   addition → final RMSNorm. Unread columns are bounded by resident row norms.
4. **LM head on the enclosure.** The Phase 1C states on the enclosure of h
   (`run_enclosure`), with the input's spread in every radius. CERTIFIED stops here.
5. **Pairwise certificate.** On the contenders left after the exact state
   (`awpmi.bounds.pairwise`).
6. **Otherwise, the exact path.** Read the rest of the suffix, recompute it with the
   reference's own operations on all positions (bitwise), and run Phase 1C on the exact h,
   reusing the first pass's reads.

## 4. Correctness

Source: `experiments/phase2/suffix-run1/summary.md` (and `suffix-run2`). Raw data:
`records.jsonl.gz`, `validation.jsonl.gz`, `reference.jsonl`, `store.json`.

| Criterion | Result |
| --- | --- |
| Certified and fallback mismatches = 0 | PASS: 0 in 37,000 runs |
| Exact prefix bitwise (every stage's boundary tensors) | PASS: 1000/1000 |
| Exact suffix recomputation bitwise (o, y and h, all positions) | PASS: 11,073/11,073 |
| LM-head fallbacks bitwise (masked on survivors, full on all rows) | PASS: 1,248/1,248, 0 guard trips |
| Enclosures, certified model: g, s, u, a (read and unread), o, y, n, q, h | PASS: 0 violations |
| Pairwise bounds (box and decomposed) against the reference's Σ Δ·y | PASS: 0 violations |
| Logit intervals of every LM-head state (enclosure and exact passes) | PASS: 0 violations |
| No page or row read twice; byte audits; budgets respected | PASS |
| Experimental models | 0 enclosure violations and 0 wrong would-certify tokens: RN-even held for every intermediate, on every prompt |
| Reproducible | PASS: identical store, reference, validation and records digests |

**Tests.** 323 pass on CPU and CUDA, 82 of them new:

- rounding models, exhaustive over every finite BF16 value;
- each operator against the reference kernels on every grid input of small enclosures,
  against adversarial faithful realizations, and against 60-digit `Decimal`/`Fraction`
  arithmetic;
- the LM head on enclosures: soundness over sampled inputs of the box;
- shared reads;
- the adaptive suffix end to end on a tiny random Llama, for every stage, budget and
  model, with intervals, pairwise certificates and fallbacks all occurring;
- the real model;
- KV-cache exactness over cached greedy generations (Mode A, roadmap §2.8).

**Phase 1C unchanged.** After the refactor of `refinement_head.py`, the Phase 1C benchmark on
50 prompts gives records, validation, reference and store identical to `runtime-run1`.

**Guards confirmed to fail when disabled:**

| Guard disabled | What failed |
| --- | --- |
| fp32 operation error in RMSNorm | An adversarial realization escapes the scale bound |
| Faithful rounding replaced by RN-even | A faithful realization escapes the SiLU enclosure |
| Input spread of the enclosure pass | Logits of other inputs in the box escape the states' intervals |
| Roundings of o and y in the pairwise bound | The decomposed bound exceeds the reference's value |
| Masked self-test (sabotaged GEMM) | The head runs FULL for every fallback |

**Independent validation (roadmap §2.5).** auto_LiRPA cannot be installed here: every release
pins torch < 1.13. The three independent checks above replace it. They are stronger for
soundness, because auto_LiRPA bounds real arithmetic only.

## 5. Results

### 5.1 Coverage of the suffix certificate (would-certify before the exact path)

| Stage, policy | Budget | faithful | RN-even elementwise | RN-even |
| --- | --- | --- | --- | --- |
| down, limited | 1.0 | **27.3%** | 49.6% | 63.7% |
| down, limited | 0.95 | **18.8%** | 30.0% | 36.0% |
| down, limited | 0.9 | **4.4%** | 8.3% | 10.3% |
| down, limited | 0.8, 0.5 | 0% | 0% | 0% |
| down, unlimited | 1.0 | 27.3% | 49.7% | 63.7% |
| mlp, limited | 1.0 | **0%** | 4.3% | 34.8% |
| mlp, limited | ≤ 0.95 | 0% | 0% | 0% |
| mlp, unlimited | 1.0 | **14.9%** | 29.8% | 40.4% |

Depth 0 (the Phase 1C LM head) certifies 90.4%, as in Phase 1C.

At depth 1 and budget 1.0, certifications split 162 by interval and 111 pairwise. At budgets
0.95 and 0.9, nearly all are pairwise: 137 of 188, and 44 of 44.

The coverage tracks the reference's top-2 gap. Depth 1, budget 1.0:

| Top-2 gap (logits) | Prompts | faithful | RN-even |
| --- | --- | --- | --- |
| < 0.5 | 317 | 0% | 3.8% |
| 0.5–1 | 221 | 0% | 74.2% |
| 1–2 | 245 | 26.9% | 99.6% |
| 2–4 | 133 | 92.5% | 100% |
| ≥ 4 | 84 | 100% | 100% |

### 5.2 Where the uncertainty comes from

Mean share of each term in the tightest pair's uncertainty, from the pairwise certificate:

| Stage, budget | Norm roundings (n, h) | y rounding | o rounding | down accumulation | unread neurons | LM accumulation |
| --- | --- | --- | --- | --- | --- | --- |
| down, 1.0 | 55% | 20% | 13% | 11% | 0% | 1% |
| down, 0.95 | 48% | 17% | 12% | 9% | 12% | 1% |
| down, 0.9 | 35% | 13% | 9% | 6% | 36% | 1% |
| mlp (unlimited), 1.0 | 55% | 20% | 14% | 10% | 0% | 1% |

The reference's own roundings make up 88% of the uncertainty with every suffix weight read.
Roundings of the norm's output alone make up 55%. Unread pages take over fast: the
Cauchy–Schwarz bound on an unread column costs about ‖Δ‖₂·‖W_d[:, i]‖₂·|a_i|, far more
than the column's actual contribution.

### 5.3 Bytes

Mean effective bytes per token, certified model:

| Stage, policy, budget | Mean MB | Baseline MB | Saving / region | Mean fraction of region |
| --- | --- | --- | --- | --- |
| lm_head (depth 0) | 22.85 | — | — | 0.404 of the head |
| down, limited, 1.0 | 26.30 | 24.62 | −0.029 | 0.450 |
| down, limited, 0.95 | 27.93 | 24.62 | −0.057 | 0.478 |
| down, limited, 0.9 | 25.61 | 24.62 | −0.017 | 0.439 |
| down, limited, 0.8 / 0.5 | 24.63 | 24.62 | −0.0002 | 0.422 |
| mlp, limited, any budget | 28.18 | 28.16 | −0.0003 | 0.455 |
| mlp, unlimited, 1.0 | 92.50 | 28.16 | −1.04 | 1.494 |

The baseline reads the whole suffix and runs Phase 1C on the exact h.

The adaptive suffix can only save the unread pages of the tokens it certifies:

- At depth 1, that is at most 0.017 MB per token on average (0.0003 of the region).
- The LM-head pass on the enclosure reads 1.0–3.3 MB more than the exact pass needs, for
  the 70–95% of tokens that end on the exact path anyway.
- At budgets ≤ 0.8 the enclosure is so wide that the pass stops before reading anything
  beyond the base level: almost no waste, and no certificate.
- At depth 2 the faithful certificates need the exact LM-head rows of a median of 45,845
  contenders: the interval pass eliminates nothing, and the pairwise certificate does all
  the work.

No budget or policy can make the extension pay here. Even a perfect policy is capped by
coverage × unread share × suffix bytes, about 0.3 × 0.1 × 1.77 MB = 0.05 MB at depth 1.
That is under 0.1% of the region, against the 1% threshold fixed before the run.

### 5.4 Time

Median wall time per run, synchronized:

| Run | Median |
| --- | --- |
| Depth 0 | 19 ms |
| Depth 1 | 42–65 ms |
| Depth 2 | 55 ms |
| Depth 2, unlimited | 454 ms |

Time is measured, not gated. As in Phase 1C, this eager PyTorch implementation is
launch-bound.

## 6. Gate

Thresholds fixed in `configs/phase2-suffix.yaml` before the full run:

| Criterion (roadmap Phase 2 gate) | Result |
| --- | --- |
| Certified mismatches = 0 | PASS |
| Fallback mismatches = 0 | PASS |
| Adaptive region extends beyond the LM head | PASS: 232 certified decisions with suffix pages unread |
| All participating bounds independently tested | PASS: test suite, §4 |
| Bound propagation remains conservative | PASS: 0 violations at every intermediate, pairwise bound and logit |
| Materialization curves recorded | PASS: 37 cells × 1000 prompts |
| Exact paths bitwise, audited bytes, reproducible | PASS |
| **Gate** | **PASS** (run1 and run2) |

Research decision, also fixed before the run: **NOT BENEFICIAL** at depth 1 (best budget 0.8,
saving −0.0002) and at depth 2 (best 1.0, −0.0003).

## 7. Conclusions

1. **Certification can reach into the transformer soundly.** Every intermediate is
   enclosed, and the pairwise certificate decides through the norm, with zero mismatches.
2. **On this model, under the faithful rounding model, it does not pay.** The reference's
   own BF16 roundings of the activations cap coverage below one token in three at depth 1,
   with every weight read. The depth-2 certificates need nearly the whole LM head. Deeper
   suffixes, such as the last block's attention, can only do worse, so they were not built
   (roadmap §2.3).
3. **The rounding model is the main lever.** RN-even in elementwise kernels alone would take
   depth-1 coverage from 27% to 50%, and RN-even everywhere to 64%. The model held on every
   intermediate of every prompt. Even so, savings would stay below 0.1% of the region here:
   the suffix is 3–9% of this model's adaptive bytes.
4. **The LM head stays the adaptive region that pays:** 0.404 of its bytes, 90.4% certified.

## 8. Recommendation

- **Phase 3 (real selective storage and streaming)** should take the LM head as its adaptive
  region, with the suffix exact. Phase 2's runtime remains as a tested option, not the
  default.
- **Decisions for the user:**
  - *RN-even for elementwise kernels.* Their BF16 conversion is PyTorch's own
    round-to-nearest-even. That is a narrower assumption than the cuBLAS epilogue, and it
    held on every prompt. Adopting it would revise decisions 0001 and 0005. On this model it
    would not change the byte verdict.
  - *Revisit Phase 2 on a model whose last layers weigh more relative to its LM head*
    (Llama-class models: one MLP is about 1/3 of the LM head), before investing further here.
- **Open bound levers, if Phase 2 is revisited:**
  - pairwise elimination at coarse precision, so that depth 2 need not read exact rows for
    most of the vocabulary;
  - sketch or low-rank metadata for unread columns, tighter than ‖Δ‖₂·‖W_d[:, i]‖₂.

## 9. Limitations

- **One setup.** One model, one GPU and one prompt set, as before.
- **Empirical models.** The fp32 operation error (2⁻¹⁸) and the accumulation model (2⁻²²)
  are documented, generous assumptions. They are validated against the reference on every
  prompt, not proven for every backend.
- **Experimental coverage is a lower bound.** The RN-even what-ifs keep faithful rounding
  for the logits themselves.
- **Logical reads.** The stores are resident, as in Phase 1C.
- **Single step.** Multi-token generation with KV reuse is tested, not benchmarked.

## 10. Reproduction

```bash
uv sync
uv run pytest                                                                   # 323 tests
uv run python benchmarks/suffix_runtime.py --output experiments/phase2/<name>   # about 41 min on an RTX 4060 Ti
uv run python benchmarks/suffix_report.py experiments/phase2/<name> --compare experiments/phase2/suffix-run1
```

The run is reproduced if `digest.json` matches run1:

| Digest | Value |
| --- | --- |
| store | `964d10ab…` |
| reference | `4b2f2523…` |
| validation | `1e3b3fde…` |
| records | `3f8186a4…` |
