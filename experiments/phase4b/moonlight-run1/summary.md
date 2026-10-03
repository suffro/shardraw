# Phase 4B Moonlight run: experiments\phase4b\moonlight-run1

correctness: **PASS** · A: **PASS** · B: **PASS** · C: **PASS** · D: **PASS** · E: **PASS** · F: **PASS**

Model: moonshotai/Moonlight-16B-A3B@476b36a4 · 26 MoE layers × 64 experts · 28.79 GB of routed experts (16.5 MiB each) · GPU 8.59 GB, cap 6.00 GB · non-expert weights 3.21 GB on the device (shared experts 0.90 GB) · prompts 16

## Per step (fractions of all routed expert bytes: mean / median / p95)

| Configuration | Phase | Steps | All equal | Expert checks | Experts / layer | Drive | Drive / token | H2D | Hit rate | Reads | Chunked calls | Compact MB (max) | Expert MB (max) | Peak device MB | Step ms* | I/O ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stream | prefill | 16 | 16 | 416/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | – | 23727 | 416/416 | 173 | 173 | 3916 | 9043 | 6865 |
| stream | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/3328 | 138 | 138 | 3400 | 1303 | 831 |
| compact | prefill | 8 | 8 | 208/208 | 51.2 | 0.8001 / 0.8312 / 0.9435 | 0.0097 | 0.7997 | – | 23954 | 0/208 | 1107 | 1107 | 4579 | 8930 | 6831 |
| compact | decode | 64 | 64 | 1664/1664 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/1664 | 104 | 104 | 3401 | 1300 | 829 |
| chunk-1 | prefill | 4 | 4 | 104/104 | 43.3 | 0.6769 / 0.7224 / 0.7712 | 0.0175 | 0.6765 | – | 20264 | 104/104 | 12 | 12 | 3312 | 18673 | 6427 |
| chunk-1 | decode | 32 | 32 | 832/832 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 832/832 | 12 | 12 | 3276 | 2688 | 895 |
| lru-40 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | 0.000 | 23727 | 416/416 | 173 | 865 | 4676 | 9185 | 6853 |
| lru-40 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | 0.000 | 2808 | 0/3328 | 104 | 796 | 4159 | 1316 | 831 |
| lru-80 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7926 / 0.8228 / 0.9570 | 0.0096 | 0.7922 | 0.000 | 23727 | 416/416 | 173 | 1557 | 5426 | 9227 | 6850 |
| lru-80 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | 0.000 | 2808 | 0/3328 | 104 | 1488 | 4912 | 1324 | 831 |
| hotness-80 | prefill | 16 | 16 | 52/416 | 50.7 | 0.7814 / 0.8114 / 0.9459 | 0.0094 | 0.7811 | 0.014 | 23395 | 416/416 | 173 | 1557 | 5423 | 9107 | 6750 |
| hotness-80 | decode | 128 | 128 | 3328/3328 | 6.0 | 0.0796 / 0.0788 / 0.0938 | 0.0796 | 0.0796 | 0.151 | 2383 | 0/3328 | 104 | 1488 | 4908 | 1204 | 703 |
| shared-streamed | prefill | 4 | 4 | 52/104 | 43.3 | 0.6769 / 0.7224 / 0.7712 | 0.0175 | 0.6765 | – | 20264 | 104/104 | 173 | 173 | 2653 | 7911 | 5885 |
| shared-streamed | decode | 32 | 32 | 832/832 | 6.0 | 0.0938 / 0.0938 / 0.0938 | 0.0938 | 0.0938 | – | 2808 | 0/832 | 104 | 104 | 2373 | 1749 | 834 |
| all-experts | prefill | 2 | 2 | 52/52 | 64.0 | 1.0005 / 1.0005 / 1.0005 | 0.0469 | 1.0000 | – | 29952 | 52/52 | 173 | 173 | 3575 | 11198 | 8808 |
| all-experts | decode | 4 | 4 | 104/104 | 64.0 | 1.0005 / 1.0005 / 1.0005 | 1.0005 | 1.0000 | – | 29952 | 104/104 | 173 | 173 | 3740 | 11206 | 8894 |

\* Step times include the benchmark's digests and checks; see the profile for un-instrumented times.

## Profile (no digests): host milliseconds per step by region, and their shares

- chunk-1 prefill: 20108 ms over 3 steps: chunks 9661 (48%), io 7112 (35%), plan 1676 (8%), assemble 915 (5%), h2d 421 (2%), other 195 (1%), route 128 (1%); drive 2.95 GB/s
- chunk-1 decode: 2785 ms over 30 steps: chunks 1215 (44%), io 914 (33%), plan 216 (8%), other 202 (7%), assemble 118 (4%), route 66 (2%), h2d 54 (2%); drive 2.96 GB/s
  - trace prefill step 0: wall 29298 ms, GPU busy 11028 ms (idle 62%), 17999 launches (370 ms CPU); device ms: memcpy_h2d 8909, other_kernels 1063, gemm 974, attention_kernels 52, memcpy_d2d 23, memcpy_other 7
  - trace decode step 1: wall 3593 ms, GPU busy 1416 ms (idle 61%), 6131 launches (106 ms CPU); device ms: memcpy_h2d 1120, gemm 209, other_kernels 77, attention_kernels 8, memcpy_d2d 2, memcpy_other 1
  - trace decode step 2: wall 3631 ms, GPU busy 1461 ms (idle 60%), 6131 launches (110 ms CPU); device ms: memcpy_h2d 1153, gemm 215, other_kernels 82, attention_kernels 8, memcpy_d2d 2, memcpy_other 1
- compact prefill: 7708 ms over 3 steps: io 6555 (85%), other 480 (6%), h2d 400 (5%), assemble 203 (3%), plan 53 (1%), route 18 (0%); drive 3.20 GB/s
- compact decode: 1275 ms over 30 steps: io 848 (66%), other 284 (22%), h2d 47 (4%), plan 43 (3%), assemble 38 (3%), route 16 (1%); drive 3.19 GB/s
  - trace prefill step 0: wall 10116 ms, GPU busy 4721 ms (idle 53%), 8693 launches (150 ms CPU); device ms: memcpy_h2d 4361, gemm 197, other_kernels 154, attention_kernels 8, memcpy_other 1, memcpy_d2d 0
  - trace decode step 1: wall 1570 ms, GPU busy 629 ms (idle 60%), 5195 launches (111 ms CPU); device ms: memcpy_h2d 519, gemm 67, other_kernels 40, attention_kernels 3, memcpy_other 0, memcpy_d2d 0
  - trace decode step 2: wall 1583 ms, GPU busy 670 ms (idle 58%), 5195 launches (97 ms CPU); device ms: memcpy_h2d 535, gemm 86, other_kernels 46, attention_kernels 3, memcpy_other 0, memcpy_d2d 0
- hotness-80 prefill: 8506 ms over 3 steps: io 6425 (76%), admit 564 (7%), h2d 414 (5%), chunks 373 (4%), assemble 289 (3%), other 228 (3%), plan 191 (2%), route 20 (0%), cache 2 (0%); drive 3.21 GB/s
- hotness-80 decode: 1191 ms over 30 steps: io 725 (61%), other 271 (23%), admit 53 (4%), plan 44 (4%), assemble 43 (4%), h2d 39 (3%), route 15 (1%), cache 0 (0%); drive 3.19 GB/s
  - trace prefill step 0: wall 11667 ms, GPU busy 4876 ms (idle 58%), 12479 launches (264 ms CPU); device ms: memcpy_h2d 4322, gemm 202, other_kernels 188, memcpy_d2d 154, attention_kernels 8, memcpy_other 1
  - trace decode step 1: wall 1627 ms, GPU busy 683 ms (idle 58%), 5507 launches (109 ms CPU); device ms: memcpy_h2d 515, gemm 68, memcpy_d2d 50, other_kernels 46, attention_kernels 2, memcpy_other 0
  - trace decode step 2: wall 1639 ms, GPU busy 743 ms (idle 55%), 5507 launches (99 ms CPU); device ms: memcpy_h2d 537, gemm 90, memcpy_d2d 56, other_kernels 56, attention_kernels 3, memcpy_other 0
- lru-80 prefill: 8658 ms over 3 steps: io 6573 (76%), chunks 560 (6%), h2d 456 (5%), assemble 309 (4%), admit 287 (3%), other 245 (3%), plan 203 (2%), route 21 (0%), cache 2 (0%); drive 3.19 GB/s
- lru-80 decode: 1302 ms over 30 steps: io 848 (65%), other 271 (21%), h2d 47 (4%), plan 46 (4%), assemble 44 (3%), admit 30 (2%), route 16 (1%), cache 0 (0%); drive 3.18 GB/s
  - trace prefill step 0: wall 11510 ms, GPU busy 4877 ms (idle 58%), 12479 launches (255 ms CPU); device ms: memcpy_h2d 4319, gemm 202, other_kernels 189, memcpy_d2d 158, attention_kernels 8, memcpy_other 1
  - trace decode step 1: wall 1647 ms, GPU busy 743 ms (idle 55%), 5507 launches (105 ms CPU); device ms: memcpy_h2d 530, gemm 91, other_kernels 61, memcpy_d2d 58, attention_kernels 3, memcpy_other 0
  - trace decode step 2: wall 1612 ms, GPU busy 706 ms (idle 56%), 5507 launches (107 ms CPU); device ms: memcpy_h2d 533, gemm 72, memcpy_d2d 49, other_kernels 49, attention_kernels 2, memcpy_other 0
- stream prefill: 8364 ms over 3 steps: io 6570 (79%), chunks 694 (8%), h2d 414 (5%), assemble 246 (3%), other 239 (3%), plan 180 (2%), route 20 (0%); drive 3.19 GB/s
- stream decode: 1273 ms over 30 steps: io 847 (67%), other 283 (22%), h2d 46 (4%), plan 43 (3%), assemble 37 (3%), route 16 (1%); drive 3.19 GB/s
  - trace prefill step 0: wall 11450 ms, GPU busy 4660 ms (idle 59%), 9377 launches (212 ms CPU); device ms: memcpy_h2d 4294, gemm 197, other_kernels 153, attention_kernels 8, memcpy_d2d 6, memcpy_other 1
  - trace decode step 1: wall 1573 ms, GPU busy 724 ms (idle 54%), 5195 launches (92 ms CPU); device ms: memcpy_h2d 575, gemm 89, other_kernels 57, attention_kernels 3, memcpy_other 0, memcpy_d2d 0
  - trace decode step 2: wall 1558 ms, GPU busy 637 ms (idle 59%), 5195 launches (109 ms CPU); device ms: memcpy_h2d 529, gemm 67, other_kernels 37, attention_kernels 3, memcpy_other 0, memcpy_d2d 0

## Host and device memory

```json
{
 "stream": {
  "before_setup": {
   "resident_bytes": 632307712,
   "peak_resident_bytes": 632307712,
   "private_bytes": 1194106880
  },
  "after_non_expert_load": {
   "resident_bytes": 1496375296,
   "peak_resident_bytes": 2171584512,
   "private_bytes": 5489909760
  },
  "peak_prefill": {
   "resident_bytes": 3246411776,
   "private_bytes": 9715040256,
   "host_commit_bytes": 4362399744
  },
  "peak_decode": {
   "resident_bytes": 3253628928,
   "private_bytes": 9713987584,
   "host_commit_bytes": 4249915392
  },
  "peak_by_configuration": {
   "stream": {
    "prefill": {
     "resident_bytes": 3246411776,
     "private_bytes": 8178561024,
     "host_commit_bytes": 4362399744,
     "device_bytes": 3915995648
    },
    "decode": {
     "resident_bytes": 3253628928,
     "private_bytes": 8414859264,
     "host_commit_bytes": 4249915392,
     "device_bytes": 3400220160
    }
   },
   "compact": {
    "prefill": {
     "resident_bytes": 2720706560,
     "private_bytes": 9539371008,
     "host_commit_bytes": 3795058688,
     "device_bytes": 4578760704
    },
    "decode": {
     "resident_bytes": 2695725056,
     "private_bytes": 9540091904,
     "host_commit_bytes": 3654340608,
     "device_bytes": 3400710144
    }
   },
   "chunk-1": {
    "prefill": {
     "resident_bytes": 2711367680,
     "private_bytes": 7169126400,
     "host_commit_bytes": 3857723392,
     "device_bytes": 3311813120
    },
    "decode": {
     "resident_bytes": 2714816512,
     "private_bytes": 7070531584,
     "host_commit_bytes": 3668951040,
     "device_bytes": 3275921408
    }
   },
   "lru-40": {
    "prefill": {
     "resident_bytes": 2734366720,
     "private_bytes": 9153654784,
     "host_commit_bytes": 3713720320,
     "device_bytes": 4676213248
    },
    "decode": {
     "resident_bytes": 2734153728,
     "private_bytes": 9152212992,
     "host_commit_bytes": 3695431680,
     "device_bytes": 4159445504
    }
   },
   "lru-80": {
    "prefill": {
     "resident_bytes": 2741596160,
     "private_bytes": 9715040256,
     "host_commit_bytes": 3951042560,
     "device_bytes": 5425945088
    },
    "decode": {
     "resident_bytes": 2741596160,
     "private_bytes": 9713987584,
     "host_commit_bytes": 3724697600,
     "device_bytes": 4911793152
    }
   },
   "hotness-80": {
    "prefill": {
     "resident_bytes": 2748485632,
     "private_bytes": 9665626112,
     "host_commit_bytes": 3934384128,
     "device_bytes": 5422799360
    },
    "decode": {
     "resident_bytes": 2747547648,
     "private_bytes": 9646182400,
     "host_commit_bytes": 3733196800,
     "device_bytes": 4907866624
    }
   },
   "shared-streamed": {
    "prefill": {
     "resident_bytes": 3022479360,
     "private_bytes": 8923570176,
     "host_commit_bytes": 4244762624,
     "device_bytes": 2653307392
    },
    "decode": {
     "resident_bytes": 3022876672,
     "private_bytes": 8924487680,
     "host_commit_bytes": 4008763392,
     "device_bytes": 2372997632
    }
   },
   "all-experts": {
    "prefill": {
     "resident_bytes": 3032125440,
     "private_bytes": 8347922432,
     "host_commit_bytes": 4231057408,
     "device_bytes": 3574940160
    },
    "decode": {
     "resident_bytes": 3042426880,
     "private_bytes": 8539115520,
     "host_commit_bytes": 4002975744,
     "device_bytes": 3740409856
    }
   }
  },
  "lifetime_peak_resident_bytes": 3253768192,
  "pinned_staging_bytes_per_backend": 134217728,
  "pinned_host_memory": {
   "active_bytes.allocated": 2953301148,
   "active_bytes.current": 536870912,
   "active_bytes.freed": 2416430236,
   "active_bytes.peak": 1073741836,
   "allocated_bytes.allocated": 1073741836,
   "allocated_bytes.current": 1073741836,
   "allocated_bytes.freed": 0,
   "allocated_bytes.peak": 1073741836
  },
  "system": {
   "start": {
    "physical_available_bytes": 20720287744,
    "system_cache_bytes": 20113379328,
    "commit_total_bytes": 19805917184
   },
   "end": {
    "physical_available_bytes": 18077986816,
    "system_cache_bytes": 886046720,
    "commit_total_bytes": 27400167424
   }
  }
 },
 "reference": {
  "before_setup": {
   "resident_bytes": 631795712,
   "peak_resident_bytes": 631795712,
   "private_bytes": 1193586688
  },
  "after_non_expert_load": {
   "resident_bytes": 787116032,
   "peak_resident_bytes": 2795880448,
   "private_bytes": 4781301760
  },
  "peak_prefill": {
   "resident_bytes": 2107707392,
   "private_bytes": 10665660416,
   "host_commit_bytes": 3073970176
  },
  "peak_decode": {
   "resident_bytes": 2104745984,
   "private_bytes": 10659426304,
   "host_commit_bytes": 3069833216
  },
  "lifetime_peak_resident_bytes": 2835255296,
  "system": {
   "start": {
    "physical_available_bytes": 20662587392,
    "system_cache_bytes": 6924087296,
    "commit_total_bytes": 19800432640
   },
   "end": {
    "physical_available_bytes": 19592642560,
    "system_cache_bytes": 19897315328,
    "commit_total_bytes": 28897013760
   }
  }
 }
}
```

## Reference stage

- Load: 3.1 s (3.13 GB of non-expert weights).
- Streamed run: 3744 experts-layer loads, 4145.7 GB converted in 3901 s (1.06 GB/s); peak device 5.15 GB, experts at most 2.21 GB.

## Routing (reference)

- Experts per layer: decode 6.00, prefill 50.70 (of 64); by prompt length: {"16": 32.0, "32": 41.9, "64": 47.9, "128": 47.1, "256": 56.2, "512": 58.8, "768": 60.8, "1024": 60.9}
- Share of decode routings received by the most used (layer, expert) slots: {"top_10pct_slots": 0.303, "top_25pct_slots": 0.551, "top_50pct_slots": 0.834}
- Of a decode step's experts, routed at the previous step in the same layer: 0.412

## Cache replay

```json
{
 "replay_checked_against_measured": {
  "lru-40": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  },
  "lru-80": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  },
  "hotness-80": {
   "replay_equals_measured_hits_every_step": true,
   "steps": 144
  }
 },
 "capacities": {
  "lru-40": {
   "capacity_experts": 40,
   "capacity_gb": 0.69206016,
   "fraction_of_experts": 0.02403846153846154,
   "decode_hit_rate": 0.0,
   "prefill_hit_rate": 0.0
  },
  "lru-80": {
   "capacity_experts": 80,
   "capacity_gb": 1.38412032,
   "fraction_of_experts": 0.04807692307692308,
   "decode_hit_rate": 0.0,
   "prefill_hit_rate": 0.0
  },
  "lru-160": {
   "capacity_experts": 160,
   "capacity_gb": 2.76824064,
   "fraction_of_experts": 0.09615384615384616,
   "decode_hit_rate": 0.3616286057692308,
   "prefill_hit_rate": 0.0005689630648143758
  },
  "lru-320": {
   "capacity_experts": 320,
   "capacity_gb": 5.53648128,
   "fraction_of_experts": 0.19230769230769232,
   "decode_hit_rate": 0.4801432291666667,
   "prefill_hit_rate": 0.018159404485325496
  },
  "lru-640": {
   "capacity_experts": 640,
   "capacity_gb": 11.07296256,
   "fraction_of_experts": 0.38461538461538464,
   "decode_hit_rate": 0.6747546073717948,
   "prefill_hit_rate": 0.11087667725570148
  },
  "lru-960": {
   "capacity_experts": 960,
   "capacity_gb": 16.60944384,
   "fraction_of_experts": 0.5769230769230769,
   "decode_hit_rate": 0.8200620993589743,
   "prefill_hit_rate": 0.2286994452610118
  },
  "lru-1248": {
   "capacity_experts": 1248,
   "capacity_gb": 21.592276992,
   "fraction_of_experts": 0.75,
   "decode_hit_rate": 0.9263571714743589,
   "prefill_hit_rate": 0.3983215589587976
  },
  "hotness-40": {
   "capacity_experts": 40,
   "capacity_gb": 0.69206016,
   "fraction_of_experts": 0.02403846153846154,
   "decode_hit_rate": 0.06372696314102565,
   "prefill_hit_rate": 0.005215494760798444
  },
  "hotness-80": {
   "capacity_experts": 80,
   "capacity_gb": 1.38412032,
   "fraction_of_experts": 0.04807692307692308,
   "decode_hit_rate": 0.15129206730769232,
   "prefill_hit_rate": 0.013892181499217676
  },
  "hotness-160": {
   "capacity_experts": 160,
   "capacity_gb": 2.76824064,
   "fraction_of_experts": 0.09615384615384616,
   "decode_hit_rate": 0.30213341346153844,
   "prefill_hit_rate": 0.03238348110568489
  },
  "hotness-320": {
   "capacity_experts": 320,
   "capacity_gb": 5.53648128,
   "fraction_of_experts": 0.19230769230769232,
   "decode_hit_rate": 0.5001252003205128,
   "prefill_hit_rate": 0.05974112180550946
  },
  "hotness-640": {
   "capacity_experts": 640,
   "capacity_gb": 11.07296256,
   "fraction_of_experts": 0.38461538461538464,
   "decode_hit_rate": 0.6883513621794872,
   "prefill_hit_rate": 0.15046702384903513
  },
  "hotness-960": {
   "capacity_experts": 960,
   "capacity_gb": 16.60944384,
   "fraction_of_experts": 0.5769230769230769,
   "decode_hit_rate": 0.8304286858974359,
   "prefill_hit_rate": 0.26843203262054904
  },
  "hotness-1248": {
   "capacity_experts": 1248,
   "capacity_gb": 21.592276992,
   "fraction_of_experts": 0.75,
   "decode_hit_rate": 0.9335186298076923,
   "prefill_hit_rate": 0.44272438480868614
  }
 }
}
```

## Gates

```json
{
  "correctness": {
    "hard_failures": 0,
    "reference_steps": 144,
    "reference_steps_expected": 144,
    "source_files_verified_against_hub_sha256": true,
    "source_files_verified": 27,
    "index_rows_audited": 3328,
    "index_rows_expected": 3328,
    "index_rows_differing": 0,
    "residency_checked_steps": 18,
    "residency_check_all_equal": true,
    "steps_all_equal": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "steps_expected": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "quantities_compared": [
      "attention",
      "dense_mlp",
      "expert_outputs",
      "experts_output",
      "kv_cache",
      "logits",
      "moe_output",
      "routed",
      "router_indices",
      "router_logits",
      "router_scores",
      "router_weights",
      "shared_output",
      "token",
      "top_k_index",
      "top_k_weights"
    ],
    "expert_outputs_checked_layers": 17732,
    "expert_outputs_layers": 18876,
    "audit_failures": 0,
    "os_counters_match_all_steps": true,
    "poisoned_steps": 27,
    "passed": true
  },
  "A": {
    "expert_bytes": 28789702656,
    "gpu_budget_bytes": 6000000000,
    "gpu_total_bytes": 8585084928,
    "experts_over_budget": 4.798283776,
    "experts_over_gpu": 3.3534557779508085,
    "min_experts_over_budget": 4.0,
    "completed_steps": {
      "stream": 144,
      "compact": 72,
      "chunk-1": 36,
      "lru-40": 144,
      "lru-80": 144,
      "hotness-80": 144,
      "shared-streamed": 36,
      "all-experts": 6
    },
    "max_peak_device_bytes": 5425945088,
    "max_peak_reserved_bytes": 5999951872,
    "passed": true
  },
  "B": {
    "worst_process_bytes": {
      "stream": {
        "working_set": 3253768192,
        "host_commit": 4362399744,
        "private_bytes_with_device_memory": 9715040256
      },
      "reference": {
        "working_set": 2835255296,
        "host_commit": 3073970176,
        "private_bytes_with_device_memory": 10665660416
      }
    },
    "fraction_of_experts": {
      "stream": {
        "working_set": 0.11301847160001458,
        "host_commit": 0.15152639108937938
      },
      "reference": {
        "working_set": 0.09848157620374418,
        "host_commit": 0.10677325197588869
      }
    },
    "max_fraction_of_experts": 0.25,
    "passed": true
  },
  "C": {
    "steps_with_a_budget": 654,
    "steps_over_budget": 0,
    "max_compact_fraction_of_layer": 0.15625,
    "max_compact_fraction_of_layer_allowed": 0.25,
    "prefill_steps_completed": 74,
    "prefill_max_compact_bytes": 173015040,
    "prefill_max_routed_experts_per_layer": 64,
    "layer_bytes": 1107296256,
    "working_set_within_capacity_plus_budget": true,
    "unbounded_compact_max_bytes": 1107296256,
    "passed": true
  },
  "D": {
    "decode_drive_fraction_without_cache": 0.09379438920454543,
    "max_decode_drive_fraction_without_cache": 0.11,
    "all_experts_decode_drive_fraction": 1.0004734848484849,
    "decode_drive_ratio_to_all_experts": 0.09374999999999997,
    "max_decode_drive_ratio_to_all_experts": 0.12,
    "decode_drive_fraction_by_configuration": {
      "stream": 0.09379438920454543,
      "compact": 0.09379438920454544,
      "chunk-1": 0.09379438920454546,
      "lru-40": 0.09379438920454543,
      "lru-80": 0.09379438920454543,
      "hotness-80": 0.07961343218396594,
      "shared-streamed": 0.09379438920454546,
      "all-experts": 1.0004734848484849
    },
    "prefill_drive_fraction_by_configuration": {
      "stream": 0.7925550732023511,
      "compact": 0.8001082271406978,
      "chunk-1": 0.6768527797885708,
      "lru-40": 0.7925550732023511,
      "lru-80": 0.7925550732023511,
      "hotness-80": 0.7814445940208881,
      "shared-streamed": 0.6768527797885708,
      "all-experts": 1.0004734848484849
    },
    "passed": true
  },
  "E": {
    "configurations": {
      "compact": {
        "steps": 72,
        "equal": 72,
        "chunked_calls": 0
      },
      "chunk-1": {
        "steps": 36,
        "equal": 36,
        "chunked_calls": 936
      },
      "stream": {
        "steps": 144,
        "equal": 144,
        "chunked_calls": 416
      }
    },
    "passed": true
  },
  "F": {
    "test_layering": "3 passed in 0.21s",
    "passed": true
  }
}
```

## Reproducibility

```json
{
  "other": "experiments/phase4b/moonlight-run2",
  "same_source_tree": true,
  "python_hash_seeds": [
    "1",
    "2"
  ],
  "digests_equal": {
    "index_sha256": true,
    "reference_rows_sha256": true,
    "prompts_sha256": true,
    "reference_sha256": true,
    "records_sha256": true,
    "ranges_sha256": true
  }
}
```
