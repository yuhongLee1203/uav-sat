# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYD; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.254 | 7.077 | 30.41% | 67.64% | 97.81% |
| w/o Kalman | 4.111 | 6.899 | 34.06% | 72.02% | 97.81% |
| w/o MeanShift | 7.780 | 12.778 | 11.68% | 27.25% | 68.37% |
| Full | 4.092 | 6.929 | 34.55% | 72.26% | 97.81% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.987 | 6.861 | 35.77% | 73.24% |
| 2 frames | 3.963 | 6.869 | 36.50% | 73.24% |
| Full | 4.092 | 6.929 | 34.55% | 72.26% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.702 | 8.474 | 58.64% | 2.859 |
| 5x5 | 25 | 4.097 | 6.926 | 72.02% | 3.905 |
| 6x6 | 36 | 4.092 | 6.929 | 72.26% | 6.877 |
| 7x7 | 49 | 4.062 | 6.899 | 72.75% | 6.332 |
| 8x8 | 64 | 4.063 | 6.899 | 72.75% | 7.697 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
