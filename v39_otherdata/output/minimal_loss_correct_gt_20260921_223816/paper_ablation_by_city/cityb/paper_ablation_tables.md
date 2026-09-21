# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYB; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.172 | 6.989 | 33.44% | 69.40% | 98.11% |
| w/o Kalman | 3.671 | 6.673 | 43.85% | 75.08% | 99.05% |
| w/o MeanShift | 6.147 | 10.647 | 17.98% | 37.85% | 87.70% |
| Full | 3.648 | 6.545 | 44.16% | 75.08% | 99.05% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.735 | 6.633 | 41.96% | 73.50% |
| 2 frames | 3.752 | 6.634 | 41.32% | 73.19% |
| Full | 3.648 | 6.545 | 44.16% | 75.08% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 3.896 | 7.193 | 72.87% | 2.852 |
| 5x5 | 25 | 3.649 | 6.543 | 75.08% | 3.737 |
| 6x6 | 36 | 3.648 | 6.545 | 75.08% | 5.152 |
| 7x7 | 49 | 3.644 | 6.542 | 75.08% | 6.844 |
| 8x8 | 64 | 3.645 | 6.544 | 75.08% | 7.957 |

## Integrity audit

`FULL_TREND_CHECK=PASS`

Full is numerically best on the predeclared primary metrics; inspect paired confidence intervals before claiming significance.
