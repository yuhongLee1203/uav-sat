# Bearing-UAV ICLR ablation (measured)

Evaluation preserves the existing Bearing-UAV controlled-GT reference protocol.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.958 | 9.031 | 31.00% | 62.50% | 92.50% |
| w/o Kalman | 4.104 | 7.825 | 43.50% | 69.50% | 96.50% |
| w/o MeanShift | 5.126 | 10.028 | 32.50% | 57.50% | 89.50% |
| Full | 4.832 | 8.946 | 32.50% | 62.50% | 93.00% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.832 | 8.952 | 32.50% | 62.50% |
| 2 frames | 4.832 | 8.925 | 32.50% | 62.50% |
| Full | 4.832 | 8.946 | 32.50% | 62.50% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.835 | 8.957 | 62.50% | 3.997 |
| 5x5 | 25 | 4.833 | 8.954 | 62.50% | 4.970 |
| 6x6 | 36 | 4.832 | 8.946 | 62.50% | 4.975 |
| 7x7 | 49 | 4.830 | 8.949 | 62.50% | 7.407 |
| 8x8 | 64 | 4.828 | 8.931 | 62.50% | 9.980 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; report the measured result and revise the method only through a new train-only validation study.
