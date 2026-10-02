# Phase 2 summary — experiments/phase2/suffix-run2

## Gate

| Criterion | Result |
| --- | --- |
| certified_mismatches_zero | PASS |
| fallback_mismatches_zero | PASS |
| adaptive_region_extends_beyond_lm_head | PASS |
| bound_propagation_conservative | PASS |
| exact_paths_bitwise | PASS |
| reads_and_bytes_audited | PASS |
| materialization_curves_recorded | PASS |
| no_hard_failure | PASS |
| reproducible | PASS |
| passed | PASS |

## Correctness (certified model)

- prompts: 1000
- prefix_bitwise: 1000
- reference_fallback_bitwise: 1000
- certified_mismatches: 0
- fallback_mismatches: 0
- enclosure_violations: {'a_read': 0, 'a_unread': 0, 'h': 0, 'n': 0, 'o': 0, 'q': 0, 'y': 0, 'g': 0, 's': 0, 'u': 0}
- pairwise_violations: {'box': 0, 'decomposed': 0}
- logit_violations: 0
- exact_suffix_runs: 11073
- exact_suffix_bitwise: 11073
- fallbacks_checked: 1248
- fallbacks_bitwise: 1248
- reads_twice: 0
- byte_audits_failed: 0
- masked_self_test: {'trials': 16, 'mismatches': 0}
- masked_enabled: True

## Experimental models (what-ifs, never certified)

- enclosure_violations: {'a_read': 0, 'a_unread': 0, 'h': 0, 'n': 0, 'o': 0, 'q': 0, 'y': 0, 'g': 0, 's': 0, 'u': 0}
- pairwise_violations: {'box': 0, 'decomposed': 0}
- logit_violations: 0
- wrong_would_certify: 0

## Curves

| stage | policy | budget | model | suffix coverage | coverage | mean fraction | mean MB | baseline MB | saving / region | h width (median) | paths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down | limited | 1 | faithful | 27.3% | 90.4% | 0.450 | 26.30 | 24.62 | -0.0288 | 0.0437 | {'interval': 162, 'exact_suffix': 727, 'pairwise': 111} |
| down | limited | 1 | nearest_even | 63.7% | – | – | – | – | – | 0.0117 | {'interval': 623, 'none': 363, 'pairwise': 14} |
| down | limited | 1 | nearest_even_elementwise | 49.6% | – | – | – | – | – | 0.0174 | {'interval': 452, 'none': 504, 'pairwise': 44} |
| down | limited | 0.95 | faithful | 18.8% | 90.4% | 0.478 | 27.93 | 24.62 | -0.0566 | 0.0852 | {'pairwise': 137, 'exact_suffix': 812, 'interval': 51} |
| down | limited | 0.95 | nearest_even | 36.0% | – | – | – | – | – | 0.0532 | {'interval': 151, 'none': 640, 'pairwise': 209} |
| down | limited | 0.95 | nearest_even_elementwise | 30.0% | – | – | – | – | – | 0.0592 | {'interval': 123, 'none': 700, 'pairwise': 177} |
| down | limited | 0.9 | faithful | 4.4% | 90.4% | 0.439 | 25.61 | 24.62 | -0.0169 | 0.2286 | {'pairwise': 44, 'exact_suffix': 956} |
| down | limited | 0.9 | nearest_even | 10.3% | – | – | – | – | – | 0.1960 | {'pairwise': 97, 'none': 897, 'interval': 6} |
| down | limited | 0.9 | nearest_even_elementwise | 8.3% | – | – | – | – | – | 0.2024 | {'pairwise': 80, 'none': 917, 'interval': 3} |
| down | limited | 0.8 | faithful | 0.0% | 90.4% | 0.422 | 24.63 | 24.62 | -0.0002 | 0.7061 | {'exact_suffix': 1000} |
| down | limited | 0.8 | nearest_even | 0.0% | – | – | – | – | – | 0.6733 | {'none': 1000} |
| down | limited | 0.8 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 0.6786 | {'none': 1000} |
| down | limited | 0.5 | faithful | 0.0% | 90.4% | 0.422 | 24.63 | 24.62 | -0.0002 | 4.2462 | {'exact_suffix': 1000} |
| down | limited | 0.5 | nearest_even | 0.0% | – | – | – | – | – | 4.1736 | {'none': 1000} |
| down | limited | 0.5 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 4.1902 | {'none': 1000} |
| down | unlimited | 1 | faithful | 27.3% | 90.4% | 0.452 | 26.39 | 24.62 | -0.0303 | 0.0437 | {'interval': 162, 'exact_suffix': 727, 'pairwise': 111} |
| down | unlimited | 1 | nearest_even | 63.7% | – | – | – | – | – | 0.0117 | {'interval': 623, 'none': 363, 'pairwise': 14} |
| down | unlimited | 1 | nearest_even_elementwise | 49.7% | – | – | – | – | – | 0.0174 | {'interval': 452, 'none': 503, 'pairwise': 45} |
| lm_head | limited | 1 | faithful | 90.4% | 90.4% | 0.404 | 22.85 | 22.85 | +0.0000 | nan | {'lm_head': 1000} |
| mlp | limited | 1 | faithful | 0.0% | 90.4% | 0.455 | 28.18 | 28.16 | -0.0003 | 0.4559 | {'exact_suffix': 1000} |
| mlp | limited | 1 | nearest_even | 34.8% | – | – | – | – | – | 0.1007 | {'pairwise': 297, 'none': 652, 'interval': 51} |
| mlp | limited | 1 | nearest_even_elementwise | 4.3% | – | – | – | – | – | 0.2648 | {'pairwise': 43, 'none': 957} |
| mlp | limited | 0.95 | faithful | 0.0% | 90.4% | 0.455 | 28.18 | 28.16 | -0.0003 | 11.6883 | {'exact_suffix': 1000} |
| mlp | limited | 0.95 | nearest_even | 0.0% | – | – | – | – | – | 10.5641 | {'none': 1000} |
| mlp | limited | 0.95 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 11.0424 | {'none': 1000} |
| mlp | limited | 0.9 | faithful | 0.0% | 90.4% | 0.455 | 28.18 | 28.16 | -0.0003 | 87.0193 | {'exact_suffix': 1000} |
| mlp | limited | 0.9 | nearest_even | 0.0% | – | – | – | – | – | 79.0593 | {'none': 1000} |
| mlp | limited | 0.9 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 82.7899 | {'none': 1000} |
| mlp | limited | 0.8 | faithful | 0.0% | 90.4% | 0.455 | 28.18 | 28.16 | -0.0003 | 805156.5714 | {'exact_suffix': 1000} |
| mlp | limited | 0.8 | nearest_even | 0.0% | – | – | – | – | – | 781321.8511 | {'none': 1000} |
| mlp | limited | 0.8 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 790207.9182 | {'none': 1000} |
| mlp | limited | 0.5 | faithful | 0.0% | 90.4% | 0.455 | 28.18 | 28.16 | -0.0003 | 2747899.5636 | {'exact_suffix': 1000} |
| mlp | limited | 0.5 | nearest_even | 0.0% | – | – | – | – | – | 2682982.0201 | {'none': 1000} |
| mlp | limited | 0.5 | nearest_even_elementwise | 0.0% | – | – | – | – | – | 2705833.4016 | {'none': 1000} |
| mlp | unlimited | 1 | faithful | 14.9% | 90.4% | 1.494 | 92.50 | 28.16 | -1.0389 | 0.4559 | {'pairwise': 149, 'exact_suffix': 851} |
| mlp | unlimited | 1 | nearest_even | 40.4% | – | – | – | – | – | 0.1007 | {'pairwise': 353, 'none': 596, 'interval': 51} |
| mlp | unlimited | 1 | nearest_even_elementwise | 29.8% | – | – | – | – | – | 0.2648 | {'pairwise': 298, 'none': 702} |

## Pairwise certificate: share of each uncertainty term (tightest pair, mean)

| stage | policy | budget | model | runs | certified | unread_neurons | down_accumulation | o_rounding | y_rounding | norm_rounding | lm_accumulation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down | limited | 1 | faithful | 824 | 111 | 0.0% | 10.6% | 13.4% | 19.8% | 55.1% | 1.0% |
| down | limited | 1 | nearest_even | 377 | 14 | 0.0% | 19.4% | 12.1% | 17.7% | 48.9% | 1.9% |
| down | limited | 1 | nearest_even_elementwise | 547 | 44 | 0.0% | 17.2% | 21.6% | 15.8% | 43.7% | 1.7% |
| down | limited | 0.95 | faithful | 801 | 137 | 12.4% | 9.1% | 11.8% | 17.4% | 48.4% | 0.9% |
| down | limited | 0.95 | nearest_even | 830 | 209 | 20.8% | 15.0% | 9.6% | 14.1% | 39.0% | 1.5% |
| down | limited | 0.95 | nearest_even_elementwise | 845 | 177 | 18.9% | 13.7% | 17.6% | 12.9% | 35.6% | 1.3% |
| down | limited | 0.9 | faithful | 155 | 44 | 36.0% | 6.3% | 9.1% | 12.6% | 35.2% | 0.9% |
| down | limited | 0.9 | nearest_even | 289 | 97 | 50.3% | 8.8% | 6.3% | 8.9% | 24.7% | 1.1% |
| down | limited | 0.9 | nearest_even_elementwise | 264 | 80 | 47.1% | 8.2% | 12.0% | 8.4% | 23.3% | 1.0% |
| down | unlimited | 1 | faithful | 838 | 111 | 0.0% | 10.6% | 13.5% | 19.8% | 55.1% | 1.0% |
| down | unlimited | 1 | nearest_even | 377 | 14 | 0.0% | 19.4% | 12.1% | 17.7% | 48.9% | 1.9% |
| down | unlimited | 1 | nearest_even_elementwise | 548 | 45 | 0.0% | 17.2% | 21.6% | 15.8% | 43.7% | 1.7% |
| mlp | limited | 1 | nearest_even | 726 | 297 | 0.0% | 18.4% | 12.2% | 17.9% | 49.5% | 1.9% |
| mlp | limited | 1 | nearest_even_elementwise | 77 | 43 | 0.0% | 15.6% | 22.6% | 15.7% | 43.6% | 2.5% |
| mlp | unlimited | 1 | faithful | 1000 | 149 | 0.0% | 9.6% | 14.2% | 19.9% | 55.3% | 1.0% |
| mlp | unlimited | 1 | nearest_even | 949 | 353 | 0.0% | 18.8% | 12.2% | 17.8% | 49.3% | 1.8% |
| mlp | unlimited | 1 | nearest_even_elementwise | 1000 | 298 | 0.0% | 16.1% | 22.3% | 15.9% | 44.1% | 1.6% |

## Decision (per stage, best budget)

- **down**: NOT BENEFICIAL — best budget 0.8, mean 24.635 MB against 24.624 MB, saving -0.0002 of the region
- **mlp**: NOT BENEFICIAL — best budget 1.0, mean 28.180 MB against 28.163 MB, saving -0.0003 of the region

Wall time per certified run: {'mean': 78.52241692363393, 'median': 53.70614997809753, 'p90': 77.55976000335068, 'max': 534.1032999567688}

## Reproducibility

Compared with experiments/phase2/suffix-run1: {'store_sha256': True, 'reference_sha256': True, 'validation_sha256': True, 'records_sha256': True}
