# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYB; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 3.856 | 6.772 | 37.86% | 73.99% | 99.13% |
| w/o Kalman | 3.273 | 5.880 | 52.02% | 82.66% | 99.71% |
| w/o MeanShift | 5.906 | 11.031 | 23.12% | 44.80% | 86.71% |
| Full | 3.290 | 5.875 | 50.58% | 82.66% | 99.71% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.180 | 5.685 | 52.60% | 83.53% |
| 2 frames | 3.235 | 5.871 | 50.87% | 82.66% |
| Full | 3.290 | 5.875 | 50.58% | 82.66% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 3.744 | 6.972 | 74.28% | 2.909 |
| 5x5 | 25 | 3.291 | 5.877 | 82.66% | 3.786 |
| 6x6 | 36 | 3.290 | 5.875 | 82.66% | 5.098 |
| 7x7 | 49 | 3.276 | 5.843 | 82.95% | 7.832 |
| 8x8 | 64 | 3.277 | 5.845 | 82.95% | 7.646 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
