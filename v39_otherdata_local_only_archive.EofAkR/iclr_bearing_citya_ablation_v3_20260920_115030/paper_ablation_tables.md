# Bearing-UAV four-city ICLR ablation (measured)

Dataset domains: CITYA; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.406 | 7.257 | 28.88% | 62.13% | 98.37% |
| w/o Kalman | 3.674 | 6.668 | 41.14% | 76.57% | 99.73% |
| w/o MeanShift | 6.331 | 11.461 | 21.80% | 38.69% | 85.01% |
| Full | 3.785 | 6.750 | 39.24% | 75.48% | 99.46% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.763 | 6.784 | 39.51% | 74.11% |
| 2 frames | 3.723 | 6.653 | 43.60% | 76.02% |
| Full | 3.785 | 6.750 | 39.24% | 75.48% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.234 | 7.712 | 69.21% | 12.725 |
| 5x5 | 25 | 3.784 | 6.749 | 75.48% | 10.953 |
| 6x6 | 36 | 3.785 | 6.750 | 75.48% | 9.445 |
| 7x7 | 49 | 3.761 | 6.728 | 76.02% | 17.301 |
| 8x8 | 64 | 3.761 | 6.729 | 76.02% | 18.164 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
