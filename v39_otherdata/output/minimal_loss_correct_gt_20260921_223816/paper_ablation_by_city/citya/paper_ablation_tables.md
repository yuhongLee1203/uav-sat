# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYA; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.541 | 7.812 | 29.21% | 59.74% | 98.95% |
| w/o Kalman | 4.143 | 7.263 | 35.00% | 66.05% | 99.47% |
| w/o MeanShift | 7.601 | 13.017 | 15.79% | 27.37% | 72.63% |
| Full | 4.133 | 7.245 | 35.53% | 66.05% | 99.47% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.132 | 7.221 | 35.26% | 65.79% |
| 2 frames | 4.142 | 7.230 | 35.53% | 66.05% |
| Full | 4.133 | 7.245 | 35.53% | 66.05% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.807 | 8.525 | 56.58% | 3.024 |
| 5x5 | 25 | 4.139 | 7.245 | 66.05% | 3.815 |
| 6x6 | 36 | 4.133 | 7.245 | 66.05% | 6.872 |
| 7x7 | 49 | 4.102 | 7.103 | 66.58% | 7.187 |
| 8x8 | 64 | 4.103 | 7.104 | 66.58% | 8.476 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
