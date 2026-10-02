# Phase 3 MoE run: experiments\phase3\moe-run2

Gate C (general backend, MoE): **PASS**

Model: ibm-granite/granite-3.1-1b-a400m-instruct@0da7a48b · experts: 24 layers × [32] · 2.42 GB of experts, 3 MiB each · copied into the pack: 0 segments · experts implementation: grouped_mm

Device memory: resident model 2.70 GB; streamed, before any cache: 0.39 GB (slot buffers 101 MB).

## Per step (fractions of all expert bytes: mean / median / p95)

| Configuration | Phase | Steps | Bitwise | Requested | Drive | Cache hit rate | H2D MB (mean) | Step ms (mean / p95) | Drive I/O ms | Copy ms | Peak device MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stream | prefill | 50 | 50 | 0.952 / 0.977 / 0.992 | 0.952 / 0.977 / 0.992 | 0.00 | 2300 | 1086 / 1153 | 745 | 379 | 491 |
| stream | decode | 600 | 600 | 0.250 / 0.250 / 0.250 | 0.251 / 0.251 / 0.251 | 0.00 | 604 | 451 / 501 | 201 | 102 | 417 |
| lru-25 | prefill | 50 | 50 | 0.952 / 0.977 / 0.992 | 0.946 / 0.970 / 0.985 | 0.01 | 2284 | 1143 / 1201 | 740 | 371 | 1124 |
| lru-25 | decode | 600 | 600 | 0.250 / 0.250 / 0.250 | 0.135 / 0.121 / 0.249 | 0.46 | 326 | 368 / 468 | 110 | 52 | 1030 |
| hotness-25 | prefill | 50 | 50 | 0.952 / 0.977 / 0.992 | 0.850 / 0.867 / 0.924 | 0.11 | 2054 | 1393 / 1496 | 660 | 334 | 1128 |
| hotness-25 | decode | 600 | 600 | 0.250 / 0.250 / 0.250 | 0.156 / 0.158 / 0.217 | 0.38 | 376 | 437 / 529 | 126 | 62 | 1030 |
| lru-50 | prefill | 50 | 50 | 0.952 / 0.977 / 0.992 | 0.825 / 0.847 / 0.878 | 0.13 | 1992 | 1098 / 1160 | 642 | 321 | 1729 |
| lru-50 | decode | 600 | 600 | 0.250 / 0.250 / 0.250 | 0.096 / 0.086 / 0.191 | 0.62 | 231 | 314 / 440 | 80 | 37 | 1634 |
| hotness-50 | prefill | 50 | 50 | 0.952 / 0.977 / 0.992 | 0.699 / 0.710 / 0.761 | 0.27 | 1686 | 1468 / 1583 | 546 | 269 | 1732 |
| hotness-50 | decode | 600 | 600 | 0.250 / 0.250 / 0.250 | 0.080 / 0.076 / 0.163 | 0.68 | 193 | 345 / 497 | 68 | 31 | 1634 |
| all-experts | prefill | 5 | 5 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.00 | 2416 | 1106 / 1128 | 790 | 397 | 494 |
| all-experts | decode | 20 | 20 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.00 | 2416 | 1079 / 1099 | 791 | 398 | 496 |

## Routing (resident reference)

- Experts per layer: decode 8.00, prefill 30.47 (of 32)
- Share of decode routings that the most used (layer, expert) slots receive: {"top_10pct_expert_slots": 0.229, "top_25pct_expert_slots": 0.451, "top_50pct_expert_slots": 0.731}
- Of a decode step's experts, routed at the previous step in the same layer: mean 0.502 (a previous-step prefetch would waste 0.498 of what it fetched)

## Gate C

{
  "hard_failures": 0,
  "all_steps_completed": true,
  "all_steps_bitwise": true,
  "all_routing_matches": true,
  "audit_failures": 0,
  "os_counters_match_all_steps": true,
  "poisoned_steps": 39,
  "decode_drive_fraction_without_cache": 0.25051865189163774,
  "max_decode_drive_fraction_without_cache": 0.3,
  "passed": true
}

## Reproducibility

{
  "other": "experiments/phase3/moe-run1",
  "same_source_tree": true,
  "digests_equal": {
    "pack_sha256": true,
    "prompts_sha256": true,
    "reference_sha256": true,
    "records_sha256": true
  }
}
