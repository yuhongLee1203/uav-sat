# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYD; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.172 | 6.618 | 29.07% | 70.13% | 98.40% |
| w/o Kalman | 3.433 | 5.857 | 44.27% | 82.93% | 100.00% |
| w/o MeanShift | 6.589 | 10.627 | 16.27% | 32.00% | 86.93% |
| Full | 3.565 | 5.929 | 43.20% | 81.07% | 99.47% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.580 | 6.061 | 41.33% | 80.80% |
| 2 frames | 3.530 | 5.941 | 42.67% | 81.07% |
| Full | 3.565 | 5.929 | 43.20% | 81.07% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 3.975 | 6.824 | 73.33% | 3.011 |
| 5x5 | 25 | 3.565 | 5.927 | 81.07% | 3.856 |
| 6x6 | 36 | 3.565 | 5.929 | 81.07% | 5.056 |
| 7x7 | 49 | 3.532 | 5.927 | 81.60% | 6.460 |
| 8x8 | 64 | 3.532 | 5.928 | 81.60% | 7.902 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
