# Phase 1 summary — `cs-column-baseline-run1`

Inputs: 1000 prompts, 12000 AWPMI runs.

## Gate

| criterion | result |
|---|---|
| run_complete | PASS |
| certified_mismatches_zero | PASS |
| fallback_mismatches_zero | PASS |
| full_materialization_reproduces_reference | PASS |
| certificate_triggers_before_final_page_on_some_inputs | PASS |
| results_reproducible | PASS |
| passed | PASS |

**Roadmap check: certification occurs almost exclusively near 100% materialization (best mean materialized fraction 0.985). Do not implement streaming; improve bounds, page decomposition and ordering first.**

## Configurations

| config | coverage | early | fallback | mean frac | median | p90 | p95 | cert. mism. | fb. mism. |
|---|---|---|---|---|---|---|---|---|---|
| width=8/bound_per_byte | 0.866 | 0.496 | 0.134 | 0.985 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=8/largest_residual | 0.866 | 0.470 | 0.134 | 0.986 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=8/sequential | 0.866 | 0.217 | 0.134 | 0.995 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=16/bound_per_byte | 0.865 | 0.170 | 0.135 | 0.994 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=16/largest_residual | 0.865 | 0.156 | 0.135 | 0.995 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=16/sequential | 0.865 | 0.090 | 0.135 | 0.997 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=32/bound_per_byte | 0.864 | 0.023 | 0.136 | 0.999 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=32/largest_residual | 0.864 | 0.021 | 0.136 | 0.999 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=32/sequential | 0.864 | 0.016 | 0.136 | 0.999 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=64/bound_per_byte | 0.863 | 0.000 | 0.137 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=64/largest_residual | 0.863 | 0.000 | 0.137 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| width=64/sequential | 0.863 | 0.000 | 0.137 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |

## Certification points (pages materialized when certified)

- width=8/bound_per_byte (of 72): 64: 3, 65: 7, 66: 12, 67: 22, 68: 37, 69: 77, 70: 126, 71: 212, 72: 370; fallbacks: 134
- width=8/largest_residual (of 72): 64: 1, 65: 5, 66: 12, 67: 18, 68: 32, 69: 60, 70: 123, 71: 219, 72: 396; fallbacks: 134
- width=8/sequential (of 72): 65: 1, 67: 4, 68: 13, 69: 24, 70: 53, 71: 122, 72: 649; fallbacks: 134
- width=16/bound_per_byte (of 36): 32: 1, 33: 5, 34: 34, 35: 130, 36: 695; fallbacks: 135
- width=16/largest_residual (of 36): 33: 3, 34: 25, 35: 128, 36: 709; fallbacks: 135
- width=16/sequential (of 36): 33: 1, 34: 15, 35: 74, 36: 775; fallbacks: 135
- width=32/bound_per_byte (of 18): 17: 23, 18: 841; fallbacks: 136
- width=32/largest_residual (of 18): 17: 21, 18: 843; fallbacks: 136
- width=32/sequential (of 18): 17: 16, 18: 848; fallbacks: 136
- width=64/bound_per_byte (of 9): 9: 863; fallbacks: 137
- width=64/largest_residual (of 9): 9: 863; fallbacks: 137
- width=64/sequential (of 9): 9: 863; fallbacks: 137

## Early certification by reference top-2 gap

- width=8/bound_per_byte: [0, 0.5) n=317 early=0.003, [0.5, 1) n=221 early=0.299, [1, 2) n=245 early=0.869, [2, 4) n=133 early=0.992, [4, 8) n=76 early=1.000, [8, inf) n=8 early=1.000
- width=8/largest_residual: [0, 0.5) n=317 early=0.003, [0.5, 1) n=221 early=0.235, [1, 2) n=245 early=0.820, [2, 4) n=133 early=0.992, [4, 8) n=76 early=1.000, [8, inf) n=8 early=1.000
- width=8/sequential: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.014, [1, 2) n=245 early=0.176, [2, 4) n=133 early=0.669, [4, 8) n=76 early=0.974, [8, inf) n=8 early=1.000
- width=16/bound_per_byte: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.069, [2, 4) n=133 early=0.556, [4, 8) n=76 early=0.934, [8, inf) n=8 early=1.000
- width=16/largest_residual: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.049, [2, 4) n=133 early=0.496, [4, 8) n=76 early=0.921, [8, inf) n=8 early=1.000
- width=16/sequential: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.008, [2, 4) n=133 early=0.165, [4, 8) n=76 early=0.763, [8, inf) n=8 early=1.000
- width=32/bound_per_byte: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.008, [4, 8) n=76 early=0.197, [8, inf) n=8 early=0.875
- width=32/largest_residual: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.008, [4, 8) n=76 early=0.171, [8, inf) n=8 early=0.875
- width=32/sequential: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.008, [4, 8) n=76 early=0.132, [8, inf) n=8 early=0.625
- width=64/bound_per_byte: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.000, [4, 8) n=76 early=0.000, [8, inf) n=8 early=0.000
- width=64/largest_residual: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.000, [4, 8) n=76 early=0.000, [8, inf) n=8 early=0.000
- width=64/sequential: [0, 0.5) n=317 early=0.000, [0.5, 1) n=221 early=0.000, [1, 2) n=245 early=0.000, [2, 4) n=133 early=0.000, [4, 8) n=76 early=0.000, [8, inf) n=8 early=0.000

## Full-materialization validation

```json
{
  "width=8": {
    "inputs": 1000,
    "prefix_mismatches": 0,
    "fallback_not_bitwise_equal": 0,
    "forced_fallback_token_mismatches": 0,
    "envelope_violations": 0,
    "max_abs_partial_minus_reference": 0.12486313749104738,
    "max_envelope_halfwidth": 0.25
  },
  "width=16": {
    "inputs": 1000,
    "prefix_mismatches": 0,
    "fallback_not_bitwise_equal": 0,
    "forced_fallback_token_mismatches": 0,
    "envelope_violations": 0,
    "max_abs_partial_minus_reference": 0.12486313749104738,
    "max_envelope_halfwidth": 0.25
  },
  "width=32": {
    "inputs": 1000,
    "prefix_mismatches": 0,
    "fallback_not_bitwise_equal": 0,
    "forced_fallback_token_mismatches": 0,
    "envelope_violations": 0,
    "max_abs_partial_minus_reference": 0.12486313749104738,
    "max_envelope_halfwidth": 0.25
  },
  "width=64": {
    "inputs": 1000,
    "prefix_mismatches": 0,
    "fallback_not_bitwise_equal": 0,
    "forced_fallback_token_mismatches": 0,
    "envelope_violations": 0,
    "max_abs_partial_minus_reference": 0.12486313749104738,
    "max_envelope_halfwidth": 0.25
  }
}
```

## Page layout

```json
{
  "8": {
    "pages_total": 72,
    "bytes_total": 56623104,
    "metadata_bytes": 14155776
  },
  "16": {
    "pages_total": 36,
    "bytes_total": 56623104,
    "metadata_bytes": 7077888
  },
  "32": {
    "pages_total": 18,
    "bytes_total": 56623104,
    "metadata_bytes": 3538944
  },
  "64": {
    "pages_total": 9,
    "bytes_total": 56623104,
    "metadata_bytes": 1769472
  }
}
```

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

## Diagnosis

```json
{
  "best_config_by_mean_materialized_fraction": "width=8/bound_per_byte",
  "best_mean_materialized_fraction": 0.9847499999999999,
  "best_certified_early_rate": 0.496,
  "certification_mostly_near_full_materialization": true
}
```
