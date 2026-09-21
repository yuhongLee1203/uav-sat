# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYD; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.263 | 7.080 | 30.66% | 65.94% | 98.05% |
| w/o Kalman | 4.109 | 7.110 | 33.58% | 69.83% | 98.30% |
| w/o MeanShift | 7.806 | 12.731 | 11.44% | 27.49% | 70.07% |
| Full | 4.089 | 6.976 | 34.06% | 70.07% | 97.81% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.013 | 7.054 | 36.01% | 73.97% |
| 2 frames | 3.956 | 7.024 | 35.77% | 74.45% |
| Full | 4.089 | 6.976 | 34.06% | 70.07% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.712 | 8.400 | 55.96% | 2.930 |
| 5x5 | 25 | 4.093 | 6.975 | 69.59% | 4.861 |
| 6x6 | 36 | 4.089 | 6.976 | 70.07% | 5.139 |
| 7x7 | 49 | 4.065 | 6.955 | 71.29% | 6.333 |
| 8x8 | 64 | 4.066 | 6.953 | 71.05% | 7.695 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
