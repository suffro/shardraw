# Phase 3 storage run: experiments\phase3\storage-run1

Gate A (physical selective materialization): **PASS** · Gate B (real byte reduction): **PASS** · Phase 1C assumption: **holds**

## Bytes per token (fraction of the BF16 LM head: mean / median / p95)

| Configuration | Logical | Drive | Host memory | Host→device | 4 KiB blocks | Amplification | Read calls (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-direct | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.00× | 55.0 |
| A-host | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | – | 1.0 |
| B/masked | 0.404 / 0.398 / 0.433 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | – | 2.5 |
| B/full | 0.500 / 0.400 / 1.397 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | – | 2.6 |
| C-direct/masked | 0.404 / 0.398 / 0.433 | 0.468 / 0.449 / 0.611 | 0.000 / 0.000 / 0.000 | 0.390 / 0.384 / 0.419 | 0.468 / 0.449 / 0.611 | 1.20× | 461.0 |
| C-direct/full | 0.500 / 0.400 / 1.397 | 0.564 / 0.465 / 1.443 | 0.000 / 0.000 / 0.000 | 0.486 / 0.386 / 1.383 | 0.564 / 0.465 / 1.443 | 1.16× | 466.3 |
| C-host/masked | 0.404 / 0.398 / 0.433 | 0.000 / 0.000 / 0.000 | 0.390 / 0.384 / 0.419 | 0.390 / 0.384 / 0.419 | 0.000 / 0.000 / 0.000 | – | 2.5 |
| C-cached-base/masked | 0.404 / 0.398 / 0.433 | 0.090 / 0.070 / 0.232 | 0.000 / 0.000 / 0.000 | 0.011 / 0.006 / 0.041 | 0.090 / 0.070 / 0.232 | 7.98× | 440.0 |

## Time per token (ms: mean / median / p95)

| Configuration | Total | Read (I/O + copy, waited) | Drive I/O | Copy (device) | Decode + matvec | Bounds | Certificate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-direct | 22.2 / 21.8 / 23.9 | 21.8 / 21.3 / 23.4 | 18.2 | 8.99 | 0.5 / 0.5 / 0.6 | – | – |
| A-host | 9.6 / 9.6 / 9.9 | 9.1 / 9.1 / 9.3 | 0.0 | 8.81 | 0.4 / 0.5 / 0.6 | – | – |
| B/masked | 20.1 / 20.0 / 25.8 | 2.1 / 1.9 / 2.8 | 0.0 | 0.00 | 4.3 / 4.5 / 4.9 | 8.9 / 9.1 / 12.2 | 3.7 / 3.3 / 4.7 |
| B/full | 20.1 / 19.4 / 27.3 | 2.3 / 1.7 / 5.4 | 0.0 | 0.00 | 4.3 / 4.5 / 4.9 | 8.9 / 8.8 / 12.4 | 3.7 / 3.2 / 4.7 |
| C-direct/masked | 43.3 / 43.9 / 56.3 | 24.6 / 26.4 / 34.1 | 15.6 | 3.61 | 4.5 / 4.6 / 6.6 | 9.3 / 9.3 / 14.2 | 3.7 / 4.1 / 4.8 |
| C-direct/full | 42.5 / 41.0 / 70.1 | 24.6 / 24.7 / 49.5 | 15.4 | 4.45 | 4.3 / 4.5 / 5.0 | 8.9 / 9.1 / 12.1 | 3.7 / 3.4 / 4.7 |
| C-host/masked | 23.9 / 24.1 / 29.1 | 6.1 / 5.9 / 7.0 | 0.0 | 3.50 | 4.3 / 4.5 / 4.9 | 8.8 / 9.1 / 11.7 | 3.7 / 3.3 / 4.7 |
| C-cached-base/masked | 31.3 / 31.5 / 42.2 | 13.4 / 14.6 / 21.1 | 6.8 | 0.18 | 4.2 / 4.5 / 5.0 | 8.9 / 9.1 / 12.1 | 3.7 / 3.5 / 4.7 |

## Memory

| Configuration | Device resident (MB) | Host resident (MB) | Peak device memory per token (MB, mean / max) |
| --- | --- | --- | --- |
| A-direct | 0.0 | 33.6 | 56.7 / 56.7 |
| A-host | 0.0 | 90.2 | 56.7 / 56.7 |
| B/masked | 93.2 | 0.0 | 97.8 / 355.5 |
| B/full | 93.2 | 0.0 | 102.3 / 355.6 |
| C-direct/masked | 0.8 | 33.6 | 97.8 / 355.4 |
| C-direct/full | 0.8 | 33.6 | 102.3 / 355.5 |
| C-host/masked | 0.8 | 126.7 | 97.8 / 355.4 |
| C-cached-base/masked | 22.2 | 33.6 | 97.8 / 355.4 |

## Correctness (gate A)

- Prompts completed: True; hard failures: 0
- Certified mismatches: 0; fallback mismatches: 0
- Reference prefix and GEMV bitwise: 1000; full-head GEMV from storage bitwise: 2000 of 2000
- Envelope violations: 0; coarse-arithmetic violations: 0 (largest ratio 2.12e-04)
- Parity failures (B against Phase 1C, C against B): {'B/full': 0, 'B/masked': 0, 'C-cached-base/masked': 0, 'C-direct/full': 0, 'C-direct/masked': 0, 'C-host/masked': 0}
- Storage audit failures: {'A-direct': 0, 'A-host': 0, 'C-cached-base/masked': 0, 'C-direct/full': 0, 'C-direct/masked': 0, 'C-host/masked': 0}
- Pack: {'levels_match_phase1c': True, 'store_matches_resident': True, 'store_matches_phase1c': True}

## Gates

- A: {'passed': True}
- B: {'primary': 'C-direct/masked', 'mean_drive_fraction': 0.46824270672268337, 'mean_h2d_fraction': 0.38971394567136414, 'baseline_mean_drive_fraction': 1.0000723379629632, 'baseline_mean_h2d_fraction': 1.0, 'max_mean_physical_fraction': 0.75, 'max_mean_h2d_fraction': 0.75, 'passed': True}

## Storage micro-benchmarks (profile.json)

```json
{
  "sequential_read_ms": 6.355399999301881,
  "sequential_read_gbps": 3.3719784753680404,
  "random_32_rows_ms": 0.8143550076056272,
  "random_32_rows_us_per_extent": 25.81156917925918,
  "random_1024_rows_ms": 9.409654996125028,
  "random_1024_rows_us_per_extent": 14.354927530320408,
  "h2d_pinned_ms": 3.3067300042603165,
  "h2d_pinned_gbps": 6.480804895588609,
  "repetitions": 20
}
```

## Reproducibility

{'other': 'experiments/phase3/storage-run2', 'same_source_tree': True, 'digests_equal': {'pack_sha256': True, 'reference_sha256': True, 'validation_sha256': True, 'records_sha256': True, 'ranges_sha256': True}}
