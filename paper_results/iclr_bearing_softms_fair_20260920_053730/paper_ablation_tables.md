# Bearing-UAV ICLR ablation (measured)

Evaluation preserves the existing Bearing-UAV controlled-GT reference protocol.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 5.087 | 9.005 | 28.00% | 60.00% | 91.50% |
| w/o Kalman | 4.389 | 8.039 | 39.50% | 65.00% | 95.00% |
| w/o MeanShift | 5.862 | 11.782 | 33.50% | 49.50% | 86.50% |
| Full | 4.979 | 8.991 | 31.00% | 59.00% | 92.50% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.966 | 8.992 | 31.50% | 59.50% |
| 2 frames | 4.961 | 8.992 | 31.50% | 59.50% |
| Full | 4.979 | 8.991 | 31.00% | 59.00% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.981 | 8.991 | 59.50% | 5.159 |
| 5x5 | 25 | 4.980 | 8.991 | 59.00% | 6.522 |
| 6x6 | 36 | 4.979 | 8.991 | 59.00% | 7.604 |
| 7x7 | 49 | 4.977 | 8.992 | 59.00% | 11.387 |
| 8x8 | 64 | 4.976 | 8.992 | 59.50% | 13.804 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; report the measured result and revise the method only through a new train-only validation study.
