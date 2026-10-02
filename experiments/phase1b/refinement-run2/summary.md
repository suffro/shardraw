# Phase 1B summary — `refinement-run2`

Inputs: 1000 prompts, 96000 simulated runs.

## Correctness gate

| criterion | result |
|---|---|
| run_complete | PASS |
| certified_mismatches_zero | PASS |
| fallback_mismatches_zero | PASS |
| reference_prefix_and_fallback_bitwise | PASS |
| envelope_violations_zero | PASS |
| reconstruction_exact | PASS |
| results_reproducible | PASS |
| passed | PASS |

Masked-fallback diagnostic: 4980 checks, 0 mismatches.

## Phase 1A baseline

- Best Phase 1A configuration `width=8/bound_per_byte`: mean fraction 0.985 (pages only), 1.235 with its resident metadata.
- Lowest effective Phase 1A fraction: `width=64/bound_per_byte` at 1.031.

## Classification (primary configuration: row_selective, realistic, lowest_index)

| decomposition | realistic mean | ideal mean | share of ideal saving | verdict |
|---|---|---|---|---|
| none | 1.007 | 1.007 | 0.000 | NOT PROMISING (decomposition-limited) |
| q8 | 0.607 | 0.606 | 1.000 | NOT PROMISING (decomposition-limited) |
| q7 | 0.545 | 0.544 | 0.998 | NOT PROMISING (decomposition-limited) |
| q6 | 0.517 | 0.481 | 0.930 | PROMISING (large realistic saving, within reach of the ideal bound) |
| q5 | 1.124 | 0.419 | -0.213 | NOT PROMISING (bound-limited) |
| q4 | 1.260 | 0.357 | -0.405 | NOT PROMISING (bound-limited) |
| q3 | 1.198 | 0.299 | -0.282 | NOT PROMISING (bound-limited) |
| q2 | 1.135 | 0.491 | -0.266 | NOT PROMISING (bound-limited) |
| q4+q4 | 0.617 | 0.363 | 0.602 | NOT PROMISING (bound-limited) |
| q4+q2 | 1.257 | 0.363 | -0.404 | NOT PROMISING (bound-limited) |
| q6+q4 | 0.499 | 0.488 | 0.980 | PROMISING (large realistic saving, within reach of the ideal bound) |
| q2+q2+q4 | 0.689 | 0.284 | 0.435 | NOT PROMISING (bound-limited) |

Recommended: **q6+q4**

## All configurations

| config | coverage | mean | median | p90 | p95 | masked-fb mean | io 4K mean | contenders after base (median) | cert. mism. | fb. mism. |
|---|---|---|---|---|---|---|---|---|---|---|
| none/global/realistic/strict | 87.3 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/global/realistic/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/global/abs_mass/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/global/ideal/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/row_selective/realistic/strict | 87.3 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/row_selective/realistic/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/row_selective/abs_mass/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| none/row_selective/ideal/lowest_index | 90.4 % | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 1.007 | 49152 | 0 | 0 |
| q8/global/realistic/strict | 87.3 % | 1.348 | 1.510 | 1.510 | 1.510 | 1.348 | 1.348 | 5 | 0 | 0 |
| q8/global/realistic/lowest_index | 90.4 % | 1.337 | 1.510 | 1.510 | 1.510 | 1.337 | 1.337 | 5 | 0 | 0 |
| q8/global/abs_mass/lowest_index | 90.4 % | 1.152 | 1.510 | 1.510 | 1.510 | 1.152 | 1.152 | 2 | 0 | 0 |
| q8/global/ideal/lowest_index | 90.4 % | 0.660 | 0.510 | 1.510 | 1.510 | 0.660 | 0.660 | 1 | 0 | 0 |
| q8/row_selective/realistic/strict | 87.3 % | 0.638 | 0.511 | 1.510 | 1.510 | 0.511 | 0.638 | 5 | 0 | 0 |
| q8/row_selective/realistic/lowest_index | 90.4 % | 0.607 | 0.511 | 0.511 | 1.510 | 0.511 | 0.607 | 5 | 0 | 0 |
| q8/row_selective/abs_mass/lowest_index | 90.4 % | 0.606 | 0.510 | 0.511 | 1.510 | 0.510 | 0.607 | 2 | 0 | 0 |
| q8/row_selective/ideal/lowest_index | 90.4 % | 0.606 | 0.510 | 0.510 | 1.510 | 0.510 | 0.606 | 1 | 0 | 0 |
| q7/global/realistic/strict | 87.3 % | 1.413 | 1.448 | 1.448 | 1.448 | 1.413 | 1.413 | 22 | 0 | 0 |
| q7/global/realistic/lowest_index | 90.4 % | 1.411 | 1.448 | 1.448 | 1.448 | 1.411 | 1.411 | 21 | 0 | 0 |
| q7/global/abs_mass/lowest_index | 90.4 % | 1.289 | 1.448 | 1.448 | 1.448 | 1.289 | 1.289 | 5 | 0 | 0 |
| q7/global/ideal/lowest_index | 90.4 % | 0.679 | 0.448 | 1.448 | 1.448 | 0.679 | 0.679 | 1 | 0 | 0 |
| q7/row_selective/realistic/strict | 87.3 % | 0.576 | 0.448 | 1.448 | 1.448 | 0.449 | 0.580 | 22 | 0 | 0 |
| q7/row_selective/realistic/lowest_index | 90.4 % | 0.545 | 0.448 | 0.459 | 1.448 | 0.449 | 0.548 | 21 | 0 | 0 |
| q7/row_selective/abs_mass/lowest_index | 90.4 % | 0.544 | 0.448 | 0.449 | 1.448 | 0.448 | 0.545 | 5 | 0 | 0 |
| q7/row_selective/ideal/lowest_index | 90.4 % | 0.544 | 0.448 | 0.448 | 1.448 | 0.448 | 0.544 | 1 | 0 | 0 |
| q6/global/realistic/strict | 87.3 % | 1.385 | 1.385 | 1.385 | 1.385 | 1.385 | 1.385 | 1026 | 0 | 0 |
| q6/global/realistic/lowest_index | 90.4 % | 1.385 | 1.385 | 1.385 | 1.385 | 1.385 | 1.385 | 980 | 0 | 0 |
| q6/global/abs_mass/lowest_index | 90.4 % | 1.353 | 1.385 | 1.385 | 1.385 | 1.353 | 1.353 | 24 | 0 | 0 |
| q6/global/ideal/lowest_index | 90.4 % | 0.713 | 0.385 | 1.385 | 1.385 | 0.713 | 0.713 | 1 | 0 | 0 |
| q6/row_selective/realistic/strict | 87.3 % | 0.548 | 0.414 | 1.385 | 1.385 | 0.427 | 0.664 | 1026 | 0 | 0 |
| q6/row_selective/realistic/lowest_index | 90.4 % | 0.517 | 0.411 | 0.682 | 1.385 | 0.426 | 0.628 | 980 | 0 | 0 |
| q6/row_selective/abs_mass/lowest_index | 90.4 % | 0.482 | 0.386 | 0.396 | 1.385 | 0.387 | 0.486 | 24 | 0 | 0 |
| q6/row_selective/ideal/lowest_index | 90.4 % | 0.481 | 0.385 | 0.386 | 1.385 | 0.385 | 0.481 | 1 | 0 | 0 |
| q5/global/realistic/strict | 87.3 % | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 41254 | 0 | 0 |
| q5/global/realistic/lowest_index | 90.4 % | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 41068 | 0 | 0 |
| q5/global/abs_mass/lowest_index | 90.4 % | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 1.323 | 1617 | 0 | 0 |
| q5/global/ideal/lowest_index | 90.4 % | 0.842 | 1.323 | 1.323 | 1.323 | 0.842 | 0.842 | 2 | 0 | 0 |
| q5/row_selective/realistic/strict | 87.3 % | 1.135 | 1.189 | 1.323 | 1.323 | 1.102 | 1.413 | 41254 | 0 | 0 |
| q5/row_selective/realistic/lowest_index | 90.4 % | 1.124 | 1.180 | 1.322 | 1.323 | 1.098 | 1.380 | 41068 | 0 | 0 |
| q5/row_selective/abs_mass/lowest_index | 90.4 % | 0.470 | 0.363 | 0.658 | 1.323 | 0.380 | 0.619 | 1617 | 0 | 0 |
| q5/row_selective/ideal/lowest_index | 90.4 % | 0.419 | 0.323 | 0.323 | 1.323 | 0.323 | 0.419 | 2 | 0 | 0 |
| q4/global/realistic/strict | 87.3 % | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 49152 | 0 | 0 |
| q4/global/realistic/lowest_index | 90.4 % | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 49152 | 0 | 0 |
| q4/global/abs_mass/lowest_index | 90.4 % | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 44124 | 0 | 0 |
| q4/global/ideal/lowest_index | 90.4 % | 1.092 | 1.260 | 1.260 | 1.260 | 1.092 | 1.092 | 4 | 0 | 0 |
| q4/row_selective/realistic/strict | 87.3 % | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 1.268 | 49152 | 0 | 0 |
| q4/row_selective/realistic/lowest_index | 90.4 % | 1.260 | 1.260 | 1.260 | 1.260 | 1.260 | 1.267 | 49152 | 0 | 0 |
| q4/row_selective/abs_mass/lowest_index | 90.4 % | 1.120 | 1.175 | 1.259 | 1.260 | 1.098 | 1.333 | 44124 | 0 | 0 |
| q4/row_selective/ideal/lowest_index | 90.4 % | 0.357 | 0.261 | 0.262 | 1.260 | 0.261 | 0.357 | 4 | 0 | 0 |
| q3/global/realistic/strict | 87.3 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 49152 | 0 | 0 |
| q3/global/realistic/lowest_index | 90.4 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 49152 | 0 | 0 |
| q3/global/abs_mass/lowest_index | 90.4 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 49152 | 0 | 0 |
| q3/global/ideal/lowest_index | 90.4 % | 1.197 | 1.198 | 1.198 | 1.198 | 1.197 | 1.197 | 207 | 0 | 0 |
| q3/row_selective/realistic/strict | 87.3 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 49152 | 0 | 0 |
| q3/row_selective/realistic/lowest_index | 90.4 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 49152 | 0 | 0 |
| q3/row_selective/abs_mass/lowest_index | 90.4 % | 1.198 | 1.198 | 1.198 | 1.198 | 1.198 | 1.204 | 49152 | 0 | 0 |
| q3/row_selective/ideal/lowest_index | 90.4 % | 0.299 | 0.203 | 0.240 | 1.198 | 0.204 | 0.320 | 207 | 0 | 0 |
| q2/global/realistic/strict | 87.3 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/global/realistic/lowest_index | 90.4 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/global/abs_mass/lowest_index | 90.4 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/global/ideal/lowest_index | 90.4 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 13793 | 0 | 0 |
| q2/row_selective/realistic/strict | 87.3 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/row_selective/realistic/lowest_index | 90.4 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/row_selective/abs_mass/lowest_index | 90.4 % | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 1.135 | 49152 | 0 | 0 |
| q2/row_selective/ideal/lowest_index | 90.4 % | 0.491 | 0.433 | 0.714 | 1.135 | 0.424 | 0.938 | 13793 | 0 | 0 |
| q4+q4/global/realistic/strict | 87.3 % | 1.415 | 1.521 | 1.521 | 1.521 | 1.415 | 1.415 | 49152 | 0 | 0 |
| q4+q4/global/realistic/lowest_index | 90.4 % | 1.410 | 1.521 | 1.521 | 1.521 | 1.410 | 1.410 | 49152 | 0 | 0 |
| q4+q4/global/abs_mass/lowest_index | 90.4 % | 1.241 | 1.521 | 1.521 | 1.521 | 1.241 | 1.241 | 44124 | 0 | 0 |
| q4+q4/global/ideal/lowest_index | 90.4 % | 0.606 | 0.521 | 1.521 | 1.521 | 0.606 | 0.606 | 4 | 0 | 0 |
| q4+q4/row_selective/realistic/strict | 87.3 % | 0.648 | 0.521 | 1.521 | 1.521 | 0.521 | 0.649 | 49152 | 0 | 0 |
| q4+q4/row_selective/realistic/lowest_index | 90.4 % | 0.617 | 0.521 | 0.523 | 1.521 | 0.521 | 0.618 | 49152 | 0 | 0 |
| q4+q4/row_selective/abs_mass/lowest_index | 90.4 % | 0.576 | 0.499 | 0.521 | 1.484 | 0.480 | 0.616 | 44124 | 0 | 0 |
| q4+q4/row_selective/ideal/lowest_index | 90.4 % | 0.363 | 0.267 | 0.268 | 1.267 | 0.267 | 0.364 | 4 | 0 | 0 |
| q4+q2/global/realistic/strict | 87.3 % | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 49152 | 0 | 0 |
| q4+q2/global/realistic/lowest_index | 90.4 % | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 49152 | 0 | 0 |
| q4+q2/global/abs_mass/lowest_index | 90.4 % | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 1.396 | 44124 | 0 | 0 |
| q4+q2/global/ideal/lowest_index | 90.4 % | 0.616 | 0.396 | 1.396 | 1.396 | 0.616 | 0.616 | 4 | 0 | 0 |
| q4+q2/row_selective/realistic/strict | 87.3 % | 1.266 | 1.318 | 1.396 | 1.396 | 1.241 | 1.502 | 49152 | 0 | 0 |
| q4+q2/row_selective/realistic/lowest_index | 90.4 % | 1.257 | 1.312 | 1.396 | 1.396 | 1.237 | 1.469 | 49152 | 0 | 0 |
| q4+q2/row_selective/abs_mass/lowest_index | 90.4 % | 0.544 | 0.446 | 0.822 | 1.377 | 0.455 | 0.757 | 44124 | 0 | 0 |
| q4+q2/row_selective/ideal/lowest_index | 90.4 % | 0.363 | 0.267 | 0.268 | 1.267 | 0.267 | 0.364 | 4 | 0 | 0 |
| q6+q4/global/realistic/strict | 87.3 % | 1.140 | 0.646 | 1.646 | 1.646 | 1.140 | 1.140 | 1026 | 0 | 0 |
| q6+q4/global/realistic/lowest_index | 90.4 % | 1.114 | 0.646 | 1.646 | 1.646 | 1.114 | 1.114 | 980 | 0 | 0 |
| q6+q4/global/abs_mass/lowest_index | 90.4 % | 0.962 | 0.646 | 1.646 | 1.646 | 0.962 | 0.962 | 24 | 0 | 0 |
| q6+q4/global/ideal/lowest_index | 90.4 % | 0.580 | 0.392 | 1.646 | 1.646 | 0.580 | 0.580 | 1 | 0 | 0 |
| q6+q4/row_selective/realistic/strict | 87.3 % | 0.530 | 0.400 | 1.393 | 1.403 | 0.403 | 0.606 | 1026 | 0 | 0 |
| q6+q4/row_selective/realistic/lowest_index | 90.4 % | 0.499 | 0.399 | 0.468 | 1.397 | 0.403 | 0.573 | 980 | 0 | 0 |
| q6+q4/row_selective/abs_mass/lowest_index | 90.4 % | 0.489 | 0.393 | 0.395 | 1.392 | 0.393 | 0.492 | 24 | 0 | 0 |
| q6+q4/row_selective/ideal/lowest_index | 90.4 % | 0.488 | 0.392 | 0.392 | 1.392 | 0.392 | 0.488 | 1 | 0 | 0 |
| q2+q2+q4/global/realistic/strict | 87.3 % | 1.531 | 1.531 | 1.531 | 1.531 | 1.531 | 1.531 | 49152 | 0 | 0 |
| q2+q2+q4/global/realistic/lowest_index | 90.4 % | 1.531 | 1.531 | 1.531 | 1.531 | 1.531 | 1.531 | 49152 | 0 | 0 |
| q2+q2+q4/global/abs_mass/lowest_index | 90.4 % | 1.504 | 1.531 | 1.531 | 1.531 | 1.504 | 1.504 | 49152 | 0 | 0 |
| q2+q2+q4/global/ideal/lowest_index | 90.4 % | 0.703 | 0.531 | 1.531 | 1.531 | 0.703 | 0.703 | 13793 | 0 | 0 |
| q2+q2+q4/row_selective/realistic/strict | 87.3 % | 0.720 | 0.583 | 1.531 | 1.531 | 0.602 | 0.894 | 49152 | 0 | 0 |
| q2+q2+q4/row_selective/realistic/lowest_index | 90.4 % | 0.689 | 0.579 | 0.953 | 1.531 | 0.599 | 0.856 | 49152 | 0 | 0 |
| q2+q2+q4/row_selective/abs_mass/lowest_index | 90.4 % | 0.629 | 0.532 | 0.548 | 1.531 | 0.533 | 0.634 | 49152 | 0 | 0 |
| q2+q2+q4/row_selective/ideal/lowest_index | 90.4 % | 0.284 | 0.189 | 0.229 | 1.186 | 0.188 | 0.387 | 13793 | 0 | 0 |

## Byte breakdown (mean fraction of the BF16 LM head)

| config | metadata | base_payload | base_scales | refinement_payload | refinement_scales | exact_rows | fallback_rows |
|---|---|---|---|---|---|---|---|
| none/row_selective/realistic/strict | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| none/row_selective/realistic/lowest_index | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| none/row_selective/abs_mass/lowest_index | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| none/row_selective/ideal/lowest_index | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| q8/row_selective/realistic/strict | 0.007 | 0.500 | 0.003 | 0.000 | 0.000 | 0.000 | 0.127 |
| q8/row_selective/realistic/lowest_index | 0.007 | 0.500 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q8/row_selective/abs_mass/lowest_index | 0.007 | 0.500 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q8/row_selective/ideal/lowest_index | 0.007 | 0.500 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q7/row_selective/realistic/strict | 0.007 | 0.438 | 0.003 | 0.000 | 0.000 | 0.001 | 0.127 |
| q7/row_selective/realistic/lowest_index | 0.007 | 0.438 | 0.003 | 0.000 | 0.000 | 0.001 | 0.096 |
| q7/row_selective/abs_mass/lowest_index | 0.007 | 0.438 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q7/row_selective/ideal/lowest_index | 0.007 | 0.438 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q6/row_selective/realistic/strict | 0.007 | 0.375 | 0.003 | 0.000 | 0.000 | 0.042 | 0.121 |
| q6/row_selective/realistic/lowest_index | 0.007 | 0.375 | 0.003 | 0.000 | 0.000 | 0.040 | 0.092 |
| q6/row_selective/abs_mass/lowest_index | 0.007 | 0.375 | 0.003 | 0.000 | 0.000 | 0.001 | 0.096 |
| q6/row_selective/ideal/lowest_index | 0.007 | 0.375 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q5/row_selective/realistic/strict | 0.007 | 0.312 | 0.003 | 0.000 | 0.000 | 0.779 | 0.033 |
| q5/row_selective/realistic/lowest_index | 0.007 | 0.312 | 0.003 | 0.000 | 0.000 | 0.775 | 0.026 |
| q5/row_selective/abs_mass/lowest_index | 0.007 | 0.312 | 0.003 | 0.000 | 0.000 | 0.057 | 0.090 |
| q5/row_selective/ideal/lowest_index | 0.007 | 0.312 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q4/row_selective/realistic/strict | 0.007 | 0.250 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q4/row_selective/realistic/lowest_index | 0.007 | 0.250 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q4/row_selective/abs_mass/lowest_index | 0.007 | 0.250 | 0.003 | 0.000 | 0.000 | 0.837 | 0.022 |
| q4/row_selective/ideal/lowest_index | 0.007 | 0.250 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q3/row_selective/realistic/strict | 0.007 | 0.188 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q3/row_selective/realistic/lowest_index | 0.007 | 0.188 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q3/row_selective/abs_mass/lowest_index | 0.007 | 0.188 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q3/row_selective/ideal/lowest_index | 0.007 | 0.188 | 0.003 | 0.000 | 0.000 | 0.006 | 0.095 |
| q2/row_selective/realistic/strict | 0.007 | 0.125 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q2/row_selective/realistic/lowest_index | 0.007 | 0.125 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q2/row_selective/abs_mass/lowest_index | 0.007 | 0.125 | 0.003 | 0.000 | 0.000 | 1.000 | 0.000 |
| q2/row_selective/ideal/lowest_index | 0.007 | 0.125 | 0.003 | 0.000 | 0.000 | 0.288 | 0.067 |
| q4+q4/row_selective/realistic/strict | 0.014 | 0.250 | 0.003 | 0.250 | 0.003 | 0.000 | 0.127 |
| q4+q4/row_selective/realistic/lowest_index | 0.014 | 0.250 | 0.003 | 0.250 | 0.003 | 0.000 | 0.096 |
| q4+q4/row_selective/abs_mass/lowest_index | 0.014 | 0.250 | 0.003 | 0.209 | 0.003 | 0.000 | 0.096 |
| q4+q4/row_selective/ideal/lowest_index | 0.014 | 0.250 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q4+q2/row_selective/realistic/strict | 0.014 | 0.250 | 0.003 | 0.125 | 0.003 | 0.845 | 0.025 |
| q4+q2/row_selective/realistic/lowest_index | 0.014 | 0.250 | 0.003 | 0.125 | 0.003 | 0.842 | 0.020 |
| q4+q2/row_selective/abs_mass/lowest_index | 0.014 | 0.250 | 0.003 | 0.105 | 0.003 | 0.080 | 0.088 |
| q4+q2/row_selective/ideal/lowest_index | 0.014 | 0.250 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q6+q4/row_selective/realistic/strict | 0.014 | 0.375 | 0.003 | 0.010 | 0.000 | 0.000 | 0.127 |
| q6+q4/row_selective/realistic/lowest_index | 0.014 | 0.375 | 0.003 | 0.010 | 0.000 | 0.000 | 0.096 |
| q6+q4/row_selective/abs_mass/lowest_index | 0.014 | 0.375 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q6+q4/row_selective/ideal/lowest_index | 0.014 | 0.375 | 0.003 | 0.000 | 0.000 | 0.000 | 0.096 |
| q2+q2+q4/row_selective/realistic/strict | 0.021 | 0.125 | 0.003 | 0.375 | 0.007 | 0.071 | 0.118 |
| q2+q2+q4/row_selective/realistic/lowest_index | 0.021 | 0.125 | 0.003 | 0.375 | 0.007 | 0.068 | 0.089 |
| q2+q2+q4/row_selective/abs_mass/lowest_index | 0.021 | 0.125 | 0.003 | 0.375 | 0.007 | 0.002 | 0.096 |
| q2+q2+q4/row_selective/ideal/lowest_index | 0.021 | 0.125 | 0.003 | 0.037 | 0.001 | 0.000 | 0.096 |

## Certified after each state (cumulative)

- none/row_selective/realistic/strict: 0:zero: 0.0 %, 1:exact: 87.3 %
- none/row_selective/realistic/lowest_index: 0:zero: 0.0 %, 1:exact: 90.4 %
- none/row_selective/abs_mass/lowest_index: 0:zero: 0.0 %, 1:exact: 90.4 %
- none/row_selective/ideal/lowest_index: 0:zero: 0.0 %, 1:exact: 90.4 %
- q8/row_selective/realistic/strict: 0:q8: 16.2 %, 1:exact: 87.3 %
- q8/row_selective/realistic/lowest_index: 0:q8: 17.3 %, 1:exact: 90.4 %
- q8/row_selective/abs_mass/lowest_index: 0:q8: 35.8 %, 1:exact: 90.4 %
- q8/row_selective/ideal/lowest_index: 0:q8: 85.0 %, 1:exact: 90.4 %
- q7/row_selective/realistic/strict: 0:q7: 3.5 %, 1:exact: 87.3 %
- q7/row_selective/realistic/lowest_index: 0:q7: 3.7 %, 1:exact: 90.4 %
- q7/row_selective/abs_mass/lowest_index: 0:q7: 15.9 %, 1:exact: 90.4 %
- q7/row_selective/ideal/lowest_index: 0:q7: 76.9 %, 1:exact: 90.4 %
- q6/row_selective/realistic/strict: 0:q6: 0.0 %, 1:exact: 87.3 %
- q6/row_selective/realistic/lowest_index: 0:q6: 0.0 %, 1:exact: 90.4 %
- q6/row_selective/abs_mass/lowest_index: 0:q6: 3.2 %, 1:exact: 90.4 %
- q6/row_selective/ideal/lowest_index: 0:q6: 67.2 %, 1:exact: 90.4 %
- q5/row_selective/realistic/strict: 0:q5: 0.0 %, 1:exact: 87.3 %
- q5/row_selective/realistic/lowest_index: 0:q5: 0.0 %, 1:exact: 90.4 %
- q5/row_selective/abs_mass/lowest_index: 0:q5: 0.0 %, 1:exact: 90.4 %
- q5/row_selective/ideal/lowest_index: 0:q5: 48.1 %, 1:exact: 90.4 %
- q4/row_selective/realistic/strict: 0:q4: 0.0 %, 1:exact: 87.3 %
- q4/row_selective/realistic/lowest_index: 0:q4: 0.0 %, 1:exact: 90.4 %
- q4/row_selective/abs_mass/lowest_index: 0:q4: 0.0 %, 1:exact: 90.4 %
- q4/row_selective/ideal/lowest_index: 0:q4: 16.8 %, 1:exact: 90.4 %
- q3/row_selective/realistic/strict: 0:q3: 0.0 %, 1:exact: 87.3 %
- q3/row_selective/realistic/lowest_index: 0:q3: 0.0 %, 1:exact: 90.4 %
- q3/row_selective/abs_mass/lowest_index: 0:q3: 0.0 %, 1:exact: 90.4 %
- q3/row_selective/ideal/lowest_index: 0:q3: 0.1 %, 1:exact: 90.4 %
- q2/row_selective/realistic/strict: 0:q2: 0.0 %, 1:exact: 87.3 %
- q2/row_selective/realistic/lowest_index: 0:q2: 0.0 %, 1:exact: 90.4 %
- q2/row_selective/abs_mass/lowest_index: 0:q2: 0.0 %, 1:exact: 90.4 %
- q2/row_selective/ideal/lowest_index: 0:q2: 0.0 %, 1:exact: 90.4 %
- q4+q4/row_selective/realistic/strict: 0:q4: 0.0 %, 1:q4: 10.6 %, 2:exact: 87.3 %
- q4+q4/row_selective/realistic/lowest_index: 0:q4: 0.0 %, 1:q4: 11.1 %, 2:exact: 90.4 %
- q4+q4/row_selective/abs_mass/lowest_index: 0:q4: 0.0 %, 1:q4: 28.0 %, 2:exact: 90.4 %
- q4+q4/row_selective/ideal/lowest_index: 0:q4: 16.8 %, 1:q4: 87.2 %, 2:exact: 90.4 %
- q4+q2/row_selective/realistic/strict: 0:q4: 0.0 %, 1:q2: 0.0 %, 2:exact: 87.3 %
- q4+q2/row_selective/realistic/lowest_index: 0:q4: 0.0 %, 1:q2: 0.0 %, 2:exact: 90.4 %
- q4+q2/row_selective/abs_mass/lowest_index: 0:q4: 0.0 %, 1:q2: 0.0 %, 2:exact: 90.4 %
- q4+q2/row_selective/ideal/lowest_index: 0:q4: 16.8 %, 1:q2: 75.8 %, 2:exact: 90.4 %
- q6+q4/row_selective/realistic/strict: 0:q6: 0.0 %, 1:q4: 50.6 %, 2:exact: 87.3 %
- q6+q4/row_selective/realistic/lowest_index: 0:q6: 0.0 %, 1:q4: 53.2 %, 2:exact: 90.4 %
- q6+q4/row_selective/abs_mass/lowest_index: 0:q6: 3.2 %, 1:q4: 67.6 %, 2:exact: 90.4 %
- q6+q4/row_selective/ideal/lowest_index: 0:q6: 67.2 %, 1:q4: 89.5 %, 2:exact: 90.4 %
- q2+q2+q4/row_selective/realistic/strict: 0:q2: 0.0 %, 1:q2: 0.0 %, 2:q4: 0.0 %, 3:exact: 87.3 %
- q2+q2+q4/row_selective/realistic/lowest_index: 0:q2: 0.0 %, 1:q2: 0.0 %, 2:q4: 0.0 %, 3:exact: 90.4 %
- q2+q2+q4/row_selective/abs_mass/lowest_index: 0:q2: 0.0 %, 1:q2: 0.0 %, 2:q4: 2.7 %, 3:exact: 90.4 %
- q2+q2+q4/row_selective/ideal/lowest_index: 0:q2: 0.0 %, 1:q2: 3.5 %, 2:q4: 81.9 %, 3:exact: 90.4 %

## Primary configuration by reference top-2 gap (coverage, mean fraction)

- none: [0, 0.5) n=317: 69.7 %, 1.007, [0.5, 1) n=221: 100.0 %, 1.007, [1, 2) n=245: 100.0 %, 1.007, [2, 4) n=133: 100.0 %, 1.007, [4, inf) n=84: 100.0 %, 1.007
- q8: [0, 0.5) n=317: 69.7 %, 0.813, [0.5, 1) n=221: 100.0 %, 0.511, [1, 2) n=245: 100.0 %, 0.511, [2, 4) n=133: 100.0 %, 0.510, [4, inf) n=84: 100.0 %, 0.510
- q7: [0, 0.5) n=317: 69.7 %, 0.752, [0.5, 1) n=221: 100.0 %, 0.449, [1, 2) n=245: 100.0 %, 0.449, [2, 4) n=133: 100.0 %, 0.448, [4, inf) n=84: 100.0 %, 0.448
- q6: [0, 0.5) n=317: 69.7 %, 0.726, [0.5, 1) n=221: 100.0 %, 0.436, [1, 2) n=245: 100.0 %, 0.424, [2, 4) n=133: 100.0 %, 0.406, [4, inf) n=84: 100.0 %, 0.394
- q5: [0, 0.5) n=317: 69.7 %, 1.194, [0.5, 1) n=221: 100.0 %, 1.146, [1, 2) n=245: 100.0 %, 1.116, [2, 4) n=133: 100.0 %, 1.054, [4, inf) n=84: 100.0 %, 0.934
- q4: [0, 0.5) n=317: 69.7 %, 1.260, [0.5, 1) n=221: 100.0 %, 1.260, [1, 2) n=245: 100.0 %, 1.260, [2, 4) n=133: 100.0 %, 1.260, [4, inf) n=84: 100.0 %, 1.260
- q3: [0, 0.5) n=317: 69.7 %, 1.198, [0.5, 1) n=221: 100.0 %, 1.198, [1, 2) n=245: 100.0 %, 1.198, [2, 4) n=133: 100.0 %, 1.198, [4, inf) n=84: 100.0 %, 1.198
- q2: [0, 0.5) n=317: 69.7 %, 1.135, [0.5, 1) n=221: 100.0 %, 1.135, [1, 2) n=245: 100.0 %, 1.135, [2, 4) n=133: 100.0 %, 1.135, [4, inf) n=84: 100.0 %, 1.135
- q4+q4: [0, 0.5) n=317: 69.7 %, 0.824, [0.5, 1) n=221: 100.0 %, 0.521, [1, 2) n=245: 100.0 %, 0.521, [2, 4) n=133: 100.0 %, 0.521, [4, inf) n=84: 100.0 %, 0.521
- q4+q2: [0, 0.5) n=317: 69.7 %, 1.309, [0.5, 1) n=221: 100.0 %, 1.279, [1, 2) n=245: 100.0 %, 1.252, [2, 4) n=133: 100.0 %, 1.204, [4, inf) n=84: 100.0 %, 1.107
- q6+q4: [0, 0.5) n=317: 69.7 %, 0.708, [0.5, 1) n=221: 100.0 %, 0.405, [1, 2) n=245: 100.0 %, 0.402, [2, 4) n=133: 100.0 %, 0.398, [4, inf) n=84: 100.0 %, 0.395
- q2+q2+q4: [0, 0.5) n=317: 69.7 %, 0.896, [0.5, 1) n=221: 100.0 %, 0.615, [1, 2) n=245: 100.0 %, 0.598, [2, 4) n=133: 100.0 %, 0.572, [4, inf) n=84: 100.0 %, 0.548

## Row-selective savings over global refinement

| decomposition / bound / tie | global mean | row-selective mean | relative saving |
|---|---|---|---|
| none/realistic/strict | 1.007 | 1.007 | 0.0 % |
| none/realistic/lowest_index | 1.007 | 1.007 | 0.0 % |
| none/abs_mass/lowest_index | 1.007 | 1.007 | 0.0 % |
| none/ideal/lowest_index | 1.007 | 1.007 | 0.0 % |
| q8/realistic/strict | 1.348 | 0.638 | 52.7 % |
| q8/realistic/lowest_index | 1.337 | 0.607 | 54.6 % |
| q8/abs_mass/lowest_index | 1.152 | 0.606 | 47.4 % |
| q8/ideal/lowest_index | 0.660 | 0.606 | 8.2 % |
| q7/realistic/strict | 1.413 | 0.576 | 59.2 % |
| q7/realistic/lowest_index | 1.411 | 0.545 | 61.4 % |
| q7/abs_mass/lowest_index | 1.289 | 0.544 | 57.8 % |
| q7/ideal/lowest_index | 0.679 | 0.544 | 19.9 % |
| q6/realistic/strict | 1.385 | 0.548 | 60.4 % |
| q6/realistic/lowest_index | 1.385 | 0.517 | 62.6 % |
| q6/abs_mass/lowest_index | 1.353 | 0.482 | 64.4 % |
| q6/ideal/lowest_index | 0.713 | 0.481 | 32.5 % |
| q5/realistic/strict | 1.323 | 1.135 | 14.2 % |
| q5/realistic/lowest_index | 1.323 | 1.124 | 15.1 % |
| q5/abs_mass/lowest_index | 1.323 | 0.470 | 64.5 % |
| q5/ideal/lowest_index | 0.842 | 0.419 | 50.2 % |
| q4/realistic/strict | 1.260 | 1.260 | 0.0 % |
| q4/realistic/lowest_index | 1.260 | 1.260 | 0.0 % |
| q4/abs_mass/lowest_index | 1.260 | 1.120 | 11.1 % |
| q4/ideal/lowest_index | 1.092 | 0.357 | 67.4 % |
| q3/realistic/strict | 1.198 | 1.198 | 0.0 % |
| q3/realistic/lowest_index | 1.198 | 1.198 | 0.0 % |
| q3/abs_mass/lowest_index | 1.198 | 1.198 | 0.0 % |
| q3/ideal/lowest_index | 1.197 | 0.299 | 75.0 % |
| q2/realistic/strict | 1.135 | 1.135 | 0.0 % |
| q2/realistic/lowest_index | 1.135 | 1.135 | 0.0 % |
| q2/abs_mass/lowest_index | 1.135 | 1.135 | 0.0 % |
| q2/ideal/lowest_index | 1.135 | 0.491 | 56.7 % |
| q4+q4/realistic/strict | 1.415 | 0.648 | 54.2 % |
| q4+q4/realistic/lowest_index | 1.410 | 0.617 | 56.2 % |
| q4+q4/abs_mass/lowest_index | 1.241 | 0.576 | 53.6 % |
| q4+q4/ideal/lowest_index | 0.606 | 0.363 | 40.1 % |
| q4+q2/realistic/strict | 1.396 | 1.266 | 9.3 % |
| q4+q2/realistic/lowest_index | 1.396 | 1.257 | 9.9 % |
| q4+q2/abs_mass/lowest_index | 1.396 | 0.544 | 61.0 % |
| q4+q2/ideal/lowest_index | 0.616 | 0.363 | 41.0 % |
| q6+q4/realistic/strict | 1.140 | 0.530 | 53.5 % |
| q6+q4/realistic/lowest_index | 1.114 | 0.499 | 55.2 % |
| q6+q4/abs_mass/lowest_index | 0.962 | 0.489 | 49.2 % |
| q6+q4/ideal/lowest_index | 0.580 | 0.488 | 15.9 % |
| q2+q2+q4/realistic/strict | 1.531 | 0.720 | 53.0 % |
| q2+q2+q4/realistic/lowest_index | 1.531 | 0.689 | 55.0 % |
| q2+q2+q4/abs_mass/lowest_index | 1.504 | 0.629 | 58.2 % |
| q2+q2+q4/ideal/lowest_index | 0.703 | 0.284 | 59.7 % |

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
