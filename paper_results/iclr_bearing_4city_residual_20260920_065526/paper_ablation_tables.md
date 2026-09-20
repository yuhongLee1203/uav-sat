# Bearing-UAV four-city ICLR ablation (measured)

Dataset domains: CITYA, CITYB, CITYC, CITYD; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.419 | 7.637 | 29.72% | 65.74% | 98.17% |
| w/o Kalman | 3.533 | 6.273 | 46.00% | 78.56% | 99.05% |
| w/o MeanShift | 7.346 | 12.592 | 12.82% | 30.60% | 78.49% |
| Full | 4.127 | 7.487 | 36.91% | 70.01% | 97.69% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.991 | 7.192 | 40.16% | 72.73% |
| 2 frames | 4.191 | 7.532 | 36.16% | 68.18% |
| Full | 4.127 | 7.487 | 36.91% | 70.01% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.628 | 8.513 | 62.96% | 11.345 |
| 5x5 | 25 | 4.129 | 7.492 | 70.01% | 11.353 |
| 6x6 | 36 | 4.127 | 7.487 | 70.01% | 9.349 |
| 7x7 | 49 | 4.095 | 7.406 | 70.08% | 16.925 |
| 8x8 | 64 | 4.096 | 7.407 | 70.08% | 20.146 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
