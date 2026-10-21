# Bearing-UAV ICLR ablation (measured)

Evaluation preserves the existing Bearing-UAV controlled-GT reference protocol.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.858 | 9.008 | 33.00% | 61.50% | 93.00% |
| w/o Kalman | 4.388 | 8.038 | 39.50% | 65.00% | 95.00% |
| w/o MeanShift | 6.998 | 13.628 | 25.00% | 41.50% | 76.00% |
| Full | 5.062 | 9.228 | 30.50% | 57.50% | 91.50% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 5.050 | 9.153 | 30.00% | 58.50% |
| 2 frames | 5.054 | 9.221 | 30.50% | 57.50% |
| Full | 5.062 | 9.228 | 30.50% | 57.50% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 5.063 | 9.228 | 57.50% | 5.471 |
| 5x5 | 25 | 5.063 | 9.228 | 57.50% | 6.452 |
| 6x6 | 36 | 5.062 | 9.228 | 57.50% | 7.413 |
| 7x7 | 49 | 5.061 | 9.228 | 57.50% | 12.024 |
| 8x8 | 64 | 5.061 | 9.228 | 57.50% | 13.791 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; report the measured result and revise the method only through a new train-only validation study.
