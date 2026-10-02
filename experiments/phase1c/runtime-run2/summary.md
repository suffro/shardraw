# Phase 1C summary — `runtime-run2`

Inputs: 1000 prompts, 2000 runtime runs.

## Gate

| criterion | result |
|---|---|
| run_complete | PASS |
| certified_mismatches_zero | PASS |
| fallback_mismatches_zero | PASS |
| reference_prefix_and_full_fallback_bitwise | PASS |
| envelope_violations_zero | PASS |
| coarse_arithmetic_violations_zero | PASS |
| store_consistent | PASS |
| masked_fallback_bitwise | PASS |
| results_reproducible | PASS |
| mean_fraction_within_limit | PASS |
| excess_over_oracle_within_limit | PASS |
| passed | PASS |

Coarse binary32 error: largest observed share of its bound 2.12e-04.
Masked fallback: self-test {'trials': 16, 'mismatches': 0}, 96 fallbacks, 96 bitwise on the surviving rows, 0 guard trips.
Round-to-nearest-even what-if: 85 of 96 fallbacks would certify (0 with a wrong token).

## Runtime by fallback mode

| mode | coverage | mean | median | p90 | p95 | certified mean | 4 KiB I/O mean | contenders after base (median / p90) | time ms (mean / median / p95) | reference ms (mean) | cert. mism. | fb. mism. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 90.4 % | 0.500 | 0.400 | 0.472 | 1.397 | 0.403 | 0.578 | 1097 / 5605 | 23.80 / 23.42 / 31.35 | 0.460 | 0 | 0 |
| masked | 90.4 % | 0.404 | 0.398 | 0.421 | 0.433 | 0.403 | 0.482 | 1097 / 5605 | 23.76 / 23.50 / 31.09 | 0.460 | 0 | 0 |

## Byte breakdown (mean fraction of the BF16 LM head)

| mode | metadata | base_payload | base_scales | refinement_payload | refinement_scales | exact_rows | fallback_rows |
|---|---|---|---|---|---|---|---|
| full | 0.014 | 0.375 | 0.003 | 0.011 | 0.000 | 0.000 | 0.096 |
| masked | 0.014 | 0.375 | 0.003 | 0.011 | 0.000 | 0.000 | 0.000 |

## Certified after each state (cumulative)

- full: 0:q6: 0.0 %, 1:q4: 53.2 %, 2:exact: 90.4 %
- masked: 0:q6: 0.0 %, 1:q4: 53.2 %, 2:exact: 90.4 %

## Against the Phase 1B oracle (primary configuration)

- same token: 1000 of 1000
- same certified: 1000 of 1000
- same decision_state: 1000 of 1000
- same contenders_after_base: 50 of 1000
- same contenders_after_state_1: 1000 of 1000
- same rows_loaded_state_1: 50 of 1000
- same contenders_after_state_2: 1000 of 1000
- same rows_loaded_state_2: 1000 of 1000
- contenders after the base, runtime over oracle: mean 1.115, median 1.107, max 1.524
- mean fraction: runtime 0.500, oracle 0.499, excess +0.0010; oracle with masked fallback 0.403
- 4 KiB I/O mean bytes: runtime 32746508, oracle 32448897

## Time per stage (mean ms over the runs that reach it)

- full: 0:bounds 2.58 (n=1000), 0:certify 1.86 (n=1000), 0:matvec 6.61 (n=1000), 1:certify 1.83 (n=1000), 1:refine 7.30 (n=1000), input 1.12 (n=1000), total 23.80 (n=1000), 2:certify 1.85 (n=468), 2:exact 2.77 (n=468), fallback 3.50 (n=96); total / reference = 51.7×
- masked: 0:bounds 2.56 (n=1000), 0:certify 1.85 (n=1000), 0:matvec 6.60 (n=1000), 1:certify 1.87 (n=1000), 1:refine 7.31 (n=1000), input 1.17 (n=1000), total 23.76 (n=1000), 2:certify 1.86 (n=468), 2:exact 2.77 (n=468), fallback 2.35 (n=96); total / reference = 51.6×

## Profile

```json
{
  "repetitions": 200,
  "micro": {
    "reference_lm_head_ms": 0.21363899984862655,
    "coarse_decode_ms": 2.63814750011079,
    "coarse_mv_on_decoded_ms": 0.42300699977204204,
    "coarse_matvec_ms": 3.561139500234276
  },
  "runs": [
    {
      "prompt_position": 0,
      "decision_state_index": 1,
      "fallback": null,
      "rows_loaded": [
        49152,
        4,
        0
      ],
      "wall_ms_without_stage_sync": 17.353329999605194,
      "device_kernel_ms": 3.6647580000000373,
      "kernel_launches": 610
    },
    {
      "prompt_position": 1,
      "decision_state_index": 2,
      "fallback": null,
      "rows_loaded": [
        49152,
        3363,
        3
      ],
      "wall_ms_without_stage_sync": 21.78511999663897,
      "device_kernel_ms": 4.73190500000001,
      "kernel_launches": 782
    },
    {
      "prompt_position": 15,
      "decision_state_index": null,
      "fallback": "full",
      "rows_loaded": [
        49152,
        1177,
        5
      ],
      "wall_ms_without_stage_sync": 24.990140000591055,
      "device_kernel_ms": 6.205678999999949,
      "kernel_launches": 812
    }
  ]
}
```

## Primary mode by reference top-2 gap (coverage, mean fraction)

- [0, 0.5) n=317: 69.7 %, 0.709
- [0.5, 1) n=221: 100.0 %, 0.406
- [1, 2) n=245: 100.0 %, 0.403
- [2, 4) n=133: 100.0 %, 0.398
- [4, inf) n=84: 100.0 %, 0.395

## Reference

```json
{
  "top2_gap": {
    "mean": 1.39518359375,
    "median": 0.875,
    "p90": 3.5,
    "p95": 4.75,
    "min": 0.0,
    "max": 12.25
  },
  "bf16_top1_ties": 38
}
```
