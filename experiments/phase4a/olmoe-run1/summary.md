# Phase 4A OLMoE run: experiments\phase4a\olmoe-run1

Correctness: **PASS** · Gate A (out of VRAM): **PASS** · Gate B (selective I/O): **PASS** · Gate C (compact residency): **PASS** · Gate D (model-general core): **PASS**

Model: allenai/OLMoE-1B-7B-0924@6d84c485 · 16 layers × 64 experts · 12.88 GB of experts (12 MiB each) · GPU 8.59 GB, budget 6.00 GB · non-expert weights on the device 0.95 GB · prompts 40

## Per step (fractions of all expert bytes: mean / median / p95)

| Configuration | Phase | Steps | All equal | Experts / token | Drive | Drive / token | H2D | Hit rate (lookups) | Reads | Compact MB (max) | Expert MB (max) | Peak device MB (max) | Step ms* | I/O ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stream | prefill | 40 | 40 | 26.7 | 0.8057 / 0.8521 / 0.9317 | 0.0261 | 0.8051 | – | 11539 | 830 | 830 | 1834 | 3699 | 3042 |
| stream | decode | 480 | 480 | 128.0 | 0.1251 / 0.1251 / 0.1251 | 0.1251 | 0.1250 | – | 1792 | 126 | 126 | 1137 | 762 | 496 |
| lru-12 | prefill | 40 | 40 | 26.7 | 0.8011 / 0.8472 / 0.9265 | 0.0260 | 0.8006 | 0.004 | 11477 | 805 | 2416 | 3430 | 3786 | 3029 |
| lru-12 | decode | 480 | 480 | 128.0 | 0.0819 / 0.0821 / 0.1244 | 0.0819 | 0.0818 | 0.345 | 1173 | 101 | 1711 | 2736 | 576 | 324 |
| lru-25 | prefill | 40 | 40 | 26.7 | 0.7761 / 0.8254 / 0.9049 | 0.0248 | 0.7756 | 0.035 | 11119 | 805 | 4027 | 5040 | 3696 | 2930 |
| lru-25 | decode | 480 | 480 | 128.0 | 0.0672 / 0.0664 / 0.1042 | 0.0672 | 0.0672 | 0.463 | 963 | 101 | 3322 | 4347 | 503 | 266 |
| hotness-25 | prefill | 40 | 40 | 26.7 | 0.7064 / 0.7674 / 0.8433 | 0.0219 | 0.7059 | 0.122 | 10121 | 805 | 4027 | 5040 | 3772 | 2677 |
| hotness-25 | decode | 480 | 480 | 128.0 | 0.0658 / 0.0674 / 0.0987 | 0.0658 | 0.0657 | 0.475 | 942 | 101 | 3322 | 4347 | 523 | 260 |
| all-experts | prefill | 3 | 3 | 13.8 | 1.0006 / 1.0006 / 1.0006 | 0.0135 | 1.0000 | – | 14332 | 805 | 805 | 1829 | 4633 | 3819 |
| all-experts | decode | 6 | 6 | 1024.0 | 1.0006 / 1.0006 / 1.0006 | 1.0006 | 1.0000 | – | 14332 | 805 | 805 | 1817 | 4787 | 3970 |

\* Step times include the benchmark's digests (hashing, the direct expert-output check); see the profile.

## Profile (no digests)

- hotness-25 prefill: 4827 ms (io 3332, assemble 660, other 368, h2d 245, admit 167, plan 33, route 21, cache 1); H2D device 1772 ms; experts device 117 ms; drive 3.22 GB/s
- hotness-25 decode: 549 ms (io 286, other 125, assemble 71, plan 26, h2d 20, admit 12, route 9, cache 0); H2D device 148 ms; experts device 42 ms; drive 3.16 GB/s
- lru-25 prefill: 4600 ms (io 3536, other 372, h2d 266, assemble 204, admit 165, plan 35, route 20, cache 1); H2D device 1874 ms; experts device 115 ms; drive 3.22 GB/s
- lru-25 decode: 525 ms (io 294, other 134, assemble 31, plan 25, h2d 20, admit 11, route 9, cache 0); H2D device 152 ms; experts device 42 ms; drive 3.16 GB/s
- stream prefill: 4508 ms (io 3639, other 402, h2d 291, assemble 121, plan 35, route 20); H2D device 1921 ms; experts device 124 ms; drive 3.20 GB/s
- stream decode: 759 ms (io 508, other 154, h2d 36, plan 26, assemble 25, route 10); H2D device 265 ms; experts device 46 ms; drive 3.18 GB/s
- lru-12 prefill: 4501 ms (io 3611, h2d 270, other 232, assemble 179, admit 165, plan 33, route 9, cache 1); H2D device 1920 ms; experts device 111 ms; drive 3.22 GB/s
- lru-12 decode: 579 ms (io 335, other 143, assemble 29, plan 27, h2d 23, admit 12, route 9, cache 0); H2D device 174 ms; experts device 44 ms; drive 3.17 GB/s

## Routing (reference)

- Experts per layer: decode 8.00, prefill 51.53 (of 64)
- Share of decode routings received by the most used (layer, expert) slots: {"top_10pct_slots": 0.25, "top_25pct_slots": 0.476, "top_50pct_slots": 0.753}
- Of a decode step's experts, routed at the previous step in the same layer: 0.376

## Gates

```json
{
  "correctness": {
    "hard_failures": 0,
    "reference_steps": 520,
    "reference_steps_expected": 520,
    "index_rows_audited": 2048,
    "index_rows_differing": 0,
    "source_files_verified_against_hub_sha256": true,
    "residency_checked_steps": 26,
    "residency_check_all_equal": true,
    "steps_all_equal": {
      "stream": 520,
      "lru-12": 520,
      "lru-25": 520,
      "hotness-25": 520,
      "all-experts": 9
    },
    "audit_failures": 0,
    "os_counters_match_all_steps": true,
    "poisoned_steps": 39,
    "passed": true
  },
  "A": {
    "expert_bytes": 12884901888,
    "gpu_budget_bytes": 6000000000,
    "gpu_total_bytes": 8585084928,
    "experts_over_budget": 2.147483648,
    "experts_over_gpu": 1.5008473411807814,
    "completed_steps": {
      "stream": 520,
      "lru-12": 520,
      "lru-25": 520,
      "hotness-25": 520,
      "all-experts": 9
    },
    "expected_steps": {
      "stream": 520,
      "lru-12": 520,
      "lru-25": 520,
      "hotness-25": 520,
      "all-experts": 9
    },
    "max_peak_device_bytes": 5040131072,
    "max_peak_reserved_bytes": 5999951872,
    "passed": true
  },
  "B": {
    "decode_drive_fraction_without_cache": 0.12508133451143896,
    "max_decode_drive_fraction_without_cache": 0.15,
    "all_experts_decode_drive_fraction": 1.0006497701009114,
    "decode_drive_ratio_to_all_experts": 0.12500011317528711,
    "max_decode_drive_ratio_to_all_experts": 0.2,
    "decode_drive_fraction_by_configuration": {
      "stream": 0.12508133451143896,
      "lru-12": 0.08189462290869817,
      "lru-25": 0.06724746955765619,
      "hotness-25": 0.06577625407112969,
      "all-experts": 1.0006497701009114
    },
    "passed": true
  },
  "C": {
    "compact_audit_failures": 0,
    "max_decode_compact_fraction_of_layer": 0.15625,
    "max_decode_compact_fraction_of_layer_allowed": 0.2,
    "layer_bytes": 805306368,
    "expert_working_set_within_capacity_plus_compact": true,
    "max_expert_peak_bytes": {
      "stream": 830472192,
      "lru-12": 2415919104,
      "lru-25": 4026531840,
      "hotness-25": 4026531840,
      "all-experts": 805306368
    },
    "cache_capacity_bytes": {
      "stream": 0,
      "lru-12": 1610612736,
      "lru-25": 3221225472,
      "hotness-25": 3221225472,
      "all-experts": 0
    },
    "passed": true
  },
  "D": {
    "test_layering": "2 passed in 0.04s",
    "passed": true
  }
}
```

## Reproducibility

```json
{
  "other": "experiments/phase4a/olmoe-run2",
  "same_source_tree": true,
  "python_hash_seeds": [
    "1",
    "2"
  ],
  "digests_equal": {
    "index_sha256": true,
    "prompts_sha256": true,
    "reference_sha256": true,
    "records_sha256": true,
    "ranges_sha256": true
  }
}
```
