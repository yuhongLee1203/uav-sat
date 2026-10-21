# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYA; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.273 | 7.299 | 32.15% | 65.67% | 98.64% |
| w/o Kalman | 3.945 | 6.843 | 38.69% | 71.39% | 98.91% |
| w/o MeanShift | 6.837 | 12.116 | 16.62% | 37.60% | 79.29% |
| Full | 3.937 | 6.804 | 38.15% | 71.93% | 99.18% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.953 | 6.805 | 38.69% | 71.93% |
| 2 frames | 3.977 | 6.810 | 38.42% | 70.03% |
| Full | 3.937 | 6.804 | 38.15% | 71.93% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.430 | 8.121 | 64.03% | 10.334 |
| 5x5 | 25 | 3.936 | 6.804 | 71.93% | 11.046 |
| 6x6 | 36 | 3.937 | 6.804 | 71.93% | 9.721 |
| 7x7 | 49 | 3.899 | 6.766 | 72.75% | 16.063 |
| 8x8 | 64 | 3.899 | 6.767 | 72.75% | 22.216 |

## Integrity audit

`FULL_TREND_CHECK=PASS`

Full is numerically best on the predeclared primary metrics; inspect paired confidence intervals before claiming significance.
