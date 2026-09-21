# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYB; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.205 | 7.293 | 33.75% | 69.40% | 98.11% |
| w/o Kalman | 3.736 | 6.690 | 41.64% | 74.13% | 99.37% |
| w/o MeanShift | 6.402 | 11.308 | 18.93% | 35.96% | 84.23% |
| Full | 3.746 | 6.711 | 41.96% | 73.50% | 99.37% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.809 | 6.687 | 41.64% | 72.56% |
| 2 frames | 3.805 | 6.650 | 41.32% | 72.87% |
| Full | 3.746 | 6.711 | 41.96% | 73.50% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.028 | 7.246 | 70.66% | 2.903 |
| 5x5 | 25 | 3.755 | 6.710 | 73.19% | 4.872 |
| 6x6 | 36 | 3.746 | 6.711 | 73.50% | 5.107 |
| 7x7 | 49 | 3.738 | 6.674 | 73.50% | 6.355 |
| 8x8 | 64 | 3.738 | 6.674 | 73.50% | 8.093 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
