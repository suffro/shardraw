# AWPMI Phase 1C report — Runtime of the q6+q4 LM head

Date: 2026-10-02 · Follows: `history/2026-10-01-awpmi-phase1b-report.md` (Phase 1B) ·
Decision: `decisions/0004-phase1c-runtime-store-arithmetic-fallbacks.md` ·
Status: **complete. The gate passes.**

## Outcome in brief

Phase 1B chose q6+q4 with row-selective refinement and the tie-aware certificate, from an
oracle that used float64 arithmetic over every row. Phase 1C implements it as a runtime
(`RefinementLMHead`):

- a bit-packed int6/int4 store whose every read is counted;
- a coarse pass over every row in binary32, with its own error term;
- float64 refinement of the surviving rows;
- the original BF16 rows for the last few;
- then a certificate or a fallback.

| | Phase 1B oracle | **Phase 1C runtime** |
| --- | --- | --- |
| Mean bytes / BF16 head, full fallback | 0.499 | **0.500** (+0.001) |
| Mean bytes / BF16 head, masked fallback | 0.403 | **0.404** |
| Median | 0.399 | 0.400 |
| Coverage (certified) | 90.4% | 90.4% |
| Certified and fallback mismatches | 0 | **0 in 2,000 runs** |
| Time per token, LM head | not measured | **24.5 ms** (reference GEMV: 0.21 ms) |

The runtime reproduces the oracle's bytes almost exactly. The decision state, coverage and
contenders after the int4 level match it on 1000 of 1000 prompts. Its own arithmetic costs
+0.001 of the head.

On time, the PyTorch implementation is about 115× slower than the resident BF16 GEMV
(24.5 ms against 0.21 ms):

- it is launch-bound, with 600–800 small kernels per token;
- decoding the int6 level dominates the GPU time.

Turning the byte saving into a time saving needs fused kernels (roadmap Phase 4) and a
storage tier where bytes cost time (Phase 3).

Two decisions were taken (decision 0004):

- The **masked fallback** is adopted as an opt-in mode, self-tested and guarded. Its
  evidence: 2,560 of 2,560 bitwise across 8 weight shapes, 2 dtypes and 2 devices, and
  96 of 96 in the benchmark.
- The **faithful output-rounding model is kept**. The reference GEMM's epilogue is
  round-to-nearest-even in 49,152 of 49,152 probe cases, and that model would certify 85
  of the 96 fallbacks. It is left for the user to adopt.

## 1. Question

> Does a real runtime of the Phase 1B decomposition keep zero mismatches and a bitwise
> fallback, and how many bytes and how much time does it really need, compared with the
> oracle?

## 2. Setup

| Item | Value |
| --- | --- |
| Model, reference, numerics | As in Phase 1A and 1B: SmolLM2-135M-Instruct @ `12fd25f7…`, BF16, unmodified HF forward, deterministic flags, u_acc = 2⁻²² |
| Hardware | RTX 4060 Ti 8 GB, Windows 11, torch 2.14.1+cu130 (CUDA 13.0), driver 610.62 |
| Prompts | The same 1000 wikitext-2 prompts (`experiments/phase1/cs-column-baseline-run1/prompts.jsonl`) |
| Decomposition | q6+q4 (`configs/phase1c-runtime.yaml`). Its level digest equals that of Phase 1B run1 |
| Store | int6 rows of 432 bytes, int4 rows of 288 bytes, a float32 scale per row, remainder norms of 16 bytes per row (resident), and the original BF16 rows of 1152 bytes. Total 93.2 MB, 1.65× the head |
| Runtime | `lowest_index` ties; coarse pass in chunks of 8192 rows, binary32, error model γ_{578}(2⁻²²) |
| Runs | 1000 prompts × 2 fallback modes (full, masked) = 2,000 runtime runs, executed twice |
| Comparison | The Phase 1B run1 record of each prompt (q6+q4, row-selective, realistic, `lowest_index`) |

## 3. Runtime

For one exact LM-head input h (`src/awpmi/refinement_head.py`):

1. **Coarse state.**
   - Read the whole int6 level, decode it 8192 rows at a time and compute
     S_j = Σ_k q_jk·h_k with binary32 `torch.mv`.
   - The centre c_j = s_j·fl(S_j) is exact in float64.
   - Radius: the oracle's realistic radius, with two changes. B_j = s_j·31·‖h‖₁
     (Hölder) replaces the level's exact mass, and the arithmetic error
     γ_{578}(2⁻²²)·B_j + s_j·τ is added (`src/awpmi/bounds/coarse.py`).
2. **Certificate, then elimination.** The tie-aware certificate runs on every row. The
   rows that can still win are the contenders.
3. **int4 state.**
   - Read only the contenders' int4 rows.
   - Recompute them in float64 (base plus refinement), as the oracle's realistic tier
     does. The radius formulas are shared.
   - Intersect with the coarse intervals, then certify and eliminate again.
4. **Exact state.** Read only the remaining contenders' BF16 rows: 1.4 rows per input on
   average, 15 at most. Compute fl₆₄(W_j·h) and the exact absolute mass, then certify.
5. **Fallback.**
   - **full**: read every BF16 row not read yet and apply `F.linear(h, W)`. This is the
     reference operation, bitwise.
   - **masked**: apply `F.linear(h, W')`, where W' keeps the contenders and has zeros
     elsewhere, and take the argmax over the contenders. It reads nothing more, and it
     is guarded by the contenders' certified intervals.

The store is the only path to weight values, and it logs every read. The effective bytes
of a run are the store's log, never a formula.

## 4. Correctness

Source: `experiments/phase1c/runtime-run1/summary.md` (machine-readable: `summary.json`;
raw: `records.jsonl`, `validation.jsonl`, `reference.jsonl`, `store.json`).

| Criterion | Result |
| --- | --- |
| Certified mismatches = 0 | PASS: 0 in 2,000 runs |
| Fallback mismatches = 0 | PASS: 0 (96 full and 96 masked fallbacks) |
| Reference prefix and full fallback `F.linear(h, W)` bitwise-equal | PASS: 1000 of 1000, plus 96 of 96 runtime full fallbacks |
| Masked fallback bitwise on the surviving rows | PASS: 96 of 96, 0 guard trips; self-test 16 of 16 |
| Envelope violations = 0, every state's own interval | PASS: 0 over 49.2 M coarse rows, 2.17 M int4 rows and 1,444 exact rows |
| Binary32 coarse error within its bound | PASS: 0 violations. The largest observed error is 2.1·10⁻⁴ of the bound |
| Byte audit | PASS: on every run, the store's log equals the decomposition's accounting |
| Store | Packing lossless; byte layout equal to decision 0003's accounting; level digest equal to Phase 1B run1 |
| Reproducible | PASS: run1 and run2 have identical store, reference, validation and records digests, with the same source tree `09c4723e…` |

**Tests.** 241 pass on CPU and CUDA, 129 of them new:

- packing round trips for 2–8 bits and any width, plus the exact bitstream layout;
- the binary32 error bound against exact rational arithmetic, on adversarial inputs:
  subnormals, 10^±20 dynamic range, cancellation, identical terms;
- runtime parity, enclosure and byte audit on random systems, with 5 decompositions, 2
  dtypes, 2 devices and both fallback modes;
- a replay check that each state advanced exactly the previous contenders;
- equality of the refined states with the oracle's arithmetic;
- BF16 ties won and lost;
- non-finite inputs;
- the real LM head.

**Guards confirmed to fail when disabled:**

| Guard disabled | What failed |
| --- | --- |
| Coarse error term set to zero | 15 tests |
| Remainder bound dropped | Wrong tokens, caught in 20 of 20 parametrizations |
| Interval intersection skipped | The replay check, in 4 parametrizations |
| Masked guard disabled | The sabotage test |
| Lossy packing | Refused |

**Phase 1B unchanged.** The radius refactor leaves the Phase 1B oracle bitwise
unchanged: its first 50 prompts give identical records (4,800), validation rows (600)
and decompositions.

## 5. Results

### 5.1 Bytes

Mean fraction of the BF16 LM-head bytes, all metadata included:

| Configuration | mean | median | p90 | p95 | certified inputs | 4 KiB I/O |
| --- | --- | --- | --- | --- | --- | --- |
| **Runtime, full fallback (primary)** | **0.500** | 0.400 | 0.472 | 1.397 | 0.403 | 0.578 |
| Runtime, masked fallback | **0.404** | 0.398 | 0.421 | 0.433 | 0.403 | 0.482 |
| Phase 1B oracle, conservative | 0.499 | 0.399 | 0.468 | 1.397 | 0.402 | 0.573 |
| Phase 1A best | 0.985 (pages only) | | | | | |

Byte breakdown for the full fallback:

| Part | Mean fraction |
| --- | --- |
| Metadata | 0.014 |
| int6 codes | 0.375 |
| int6 scales | 0.003 |
| int4 rows | 0.011 |
| Exact rows | 0.0000 |
| Fallback | 0.096 |

Certification by state:

- after the int6 pass: 0%;
- after the int4 refinement: 53.2%;
- after the exact state: 90.4% (the rest falls back).

A fallback costs 1.40 of the head with the full mode. With the masked mode it costs the
same as a certified input.

### 5.2 Against the oracle

| Same as the oracle | Prompts |
| --- | --- |
| Token, certified flag, decision state | 1000 of 1000 |
| Contenders after the int4 level | 1000 of 1000 |
| Contenders and rows after the exact state | 1000 of 1000 |
| Contenders after the base | 50 of 1000 |

After the base, the runtime keeps more contenders than the oracle:

| | Median | p90 | Mean |
| --- | --- | --- | --- |
| Runtime | 1,097 | 5,605 | 2,174 |
| Oracle | 980 | 5,092 | 1,983 |

The runtime-to-oracle ratio has a mean of 1.115 and a maximum of 1.52. The cause is the
Hölder mass bound and the binary32 error term. Their whole cost is the +0.001 difference
in mean fraction: int4 rows and scales cost 0.0112 of the head instead of 0.0102.

### 5.3 Time

Per token, run1, with the device synchronized at every stage boundary. Run2 measured
23.8 ms in total.

| Stage | Mean ms | Runs |
| --- | --- | --- |
| Input norms and checks | 1.15 | 1000 |
| Coarse pass (decode and binary32 mv, 49,152 rows) | 6.73 | 1000 |
| Coarse intervals (float64, 49,152 rows) | 2.69 | 1000 |
| Certificate and elimination after int6 | 1.93 | 1000 |
| int4 refinement (2,174 rows on average) | 7.50 | 1000 |
| Certificate after int4 | 1.96 | 1000 |
| Exact rows | 3.00 | 468 |
| Certificate after exact | 1.85 | 468 |
| Fallback, full / masked | 3.52 / 2.35 | 96 |
| **Total** | **24.5** (median 24.0, p95 33.4) | |
| Reference LM-head GEMV, same timer | 0.46 | |
| Reference LM-head GEMV, micro-benchmark | 0.21 | |

Profile (`profile.json`):

- Without stage synchronization, a run takes 14–26 ms of wall time.
- The GPU spends only 3.6–5.9 ms of that in kernels, launched as 610–812 separate kernels.
- In the coarse pass (3.36 ms), decoding the int6 codes takes 2.67 ms and the binary32
  GEMV only 0.43 ms.

**Reading.** The runtime does what it should with bytes, but the eager PyTorch
implementation is dominated by two costs:

1. **Host-side launch overhead.** There are hundreds of small kernels, at about
   15–20 µs each under Windows WDDM.
2. **Decoding.** The int6 level is unpacked into 113 MB of float32 codes before a GEMV
   that reads them again.

A fused decode-and-dot kernel would read only the 21.4 MB of packed codes. At this GPU's
bandwidth (288 GB/s) that is about 0.07 ms.

In a resident setting no byte saving can pay for 24 ms. In a streaming setting, reading
the 56.6 MB head at NVMe-class 3–7 GB/s costs 8–19 ms per token, and 0.40–0.50 of it
saves 4–11 ms. That pays off only once the runtime overhead is well below that. The
profile says where to look.

### 5.4 Fallbacks: the two levers

- **Masked fallback.**
  - Evidence from `experiments/phase1c/fallback-study`:
    - 2,560 of 2,560 bitwise checks;
    - 6 LM-head shapes (49152×{576, 960, 2048}, 151936×896, 128256×2048, 32000×4096)
      and 2 odd shapes;
    - BF16 and FP32, on CUDA, and on CPU for the shapes up to 64 M weights;
    - normal, heavy-tailed and sparse inputs, and 50 real SmolLM2 inputs;
    - masks from one row to half of the rows, including the top 1, 2, 5 and 20 rows of
      the logits.
  - In the benchmark, 96 of 96 masked fallbacks were bitwise on their survivors, with no
    guard trips.
  - Effect: the mean falls from 0.500 to 0.404, and the p95 from 1.397 to 0.433.
- **Output rounding.**
  - Probe method: rows with two non-zero products whose sum is exact in binary32, so the
    output shows the epilogue's rounding alone.
  - Result: 49,152 of 49,152 match round-to-nearest-even on CUDA and on CPU. That
    includes 16,384 exact midpoints, where rounding down or up each matches only half.
  - If certification used RN-even instead of faithful rounding, 85 of the 96 fallbacks
    would certify, none with a wrong token. Coverage would go from 90.4% to 98.9%, and
    the full-fallback mean from 0.500 to about 0.415.

## 6. Decisions (decision 0004)

1. Packed store with counted reads. The runtime's bytes are the store's log, and they are
   audited against the decomposition's accounting on every run.
2. A binary32 coarse pass with a γ_{K+2}(2⁻²²) error model and a free Hölder mass bound.
   It is validated against rational arithmetic in tests and against float64 on every
   prompt.
3. The refined and exact states reproduce the oracle's arithmetic, with shared radius
   formulas.
4. The full fallback stays the default. The masked fallback is opt-in, self-tested and
   guarded.
5. The faithful output-rounding model is kept. The RN-even evidence and its effect are
   recorded for a later decision.
6. Time is measured, not gated.

## 7. Gate

Thresholds fixed in `configs/phase1c-runtime.yaml` before the full run:

| Criterion | Result |
| --- | --- |
| Correctness, every row of §4 | PASS |
| Primary mean fraction ≤ 0.60 | PASS (0.500) |
| Primary mean ≤ oracle mean + 0.01 | PASS (+0.0010) |
| **Gate** | **PASS** (run1 and run2) |

## 8. Recommendation

Phase 1 (the adaptive LM head) is now validated end to end. The roadmap's next phase is
**Phase 2**: extend certification into the transformer, starting from the final norm and
last MLP. The LM-head runtime becomes its last stage.

Before Phase 2, two decisions belong to the user. In Phase 2 a fallback recomputes a
whole adaptive suffix, so they matter more than here:

1. **Make the masked fallback the default?** The evidence is strong and the mode is
   guarded, but the property is not documented. It changes nothing for certified
   inputs.
2. **Adopt round-to-nearest-even as the output model?** That would revise decision 0001.
   It would raise coverage from 90.4% to 98.9%. The cost is an undocumented library
   property inside the certified path. That property is probed and is checked again by
   every envelope check.

Open levers, by phase:

| Lever | Phase | What it would do |
| --- | --- | --- |
| A fused decode-and-dot coarse kernel, plus fewer, larger bound kernels | 4 (now justified by this profile) | Turn bytes into time |
| A row layout that limits 4 KiB amplification | 3 | Bring I/O from 0.578 toward the logical 0.500 |
| A tighter sound remainder bound | — | Allow a base of 5 bits or fewer; ideal-tier ceiling 0.19–0.28 |

## 9. Limitations

- **One setup.** One model, one GPU, one OS and one prompt set. Windows WDDM inflates the
  kernel-launch overhead that dominates the time.
- **Logical reads.** The store is resident, so "real bytes" are the bytes of the real
  packed buffers the store hands out, not storage reads. 4 KiB I/O remains a model.
- **Single platform for the empirical evidence.** The masked-fallback and RN-even
  evidence comes from one platform: driver 610.62, CUDA 13.0 and the cuBLAS shipped with
  torch 2.14.1. The masked mode self-tests at start-up for that reason.
- **The coarse error model** (u = 2⁻²²) is an assumption chosen to cover non-IEEE
  accumulation. It is validated against float64 on every prompt, with a largest observed
  share of its bound of 2.1·10⁻⁴. The reference's own accumulation model (decision 0001)
  is unchanged and still empirical.
- **Scope of the timing.** Batch size 1, one position. Times exclude the transformer
  body, which is identical in both paths.

## 10. Reproduction

```bash
uv sync
uv run pytest                                                                        # 241 tests
uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/<name>   # about 3 min on an RTX 4060 Ti
uv run python benchmarks/refinement_runtime_report.py experiments/phase1c/<name> --compare experiments/phase1c/runtime-run1
uv run python benchmarks/fallback_study.py --output experiments/phase1c/<study>      # about 1 min
```

The run is reproduced if `digest.json` matches run1:

| Digest | Value |
| --- | --- |
| store | `4fe2237a…` |
| reference | `2f668b47…` (identical to the Phase 1B reference digest) |
| validation | `aa4072fb…` |
| records | `34925984…` |

The fallback study ran on the same source tree (`09c4723e…`).
