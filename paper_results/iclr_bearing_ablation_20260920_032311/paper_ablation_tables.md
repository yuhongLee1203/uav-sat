# Bearing-UAV ICLR ablation (measured)

Evaluation preserves the existing Bearing-UAV controlled-GT reference protocol.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 5.212 | 9.730 | 32.50% | 59.00% | 90.00% |
| w/o Kalman | 4.397 | 8.015 | 38.50% | 65.00% | 94.50% |
| w/o MeanShift | 6.912 | 13.079 | 26.50% | 40.00% | 75.50% |
| Full | 5.146 | 9.245 | 30.50% | 58.00% | 92.00% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 5.147 | 9.269 | 31.50% | 58.00% |
| 2 frames | 5.147 | 9.262 | 31.50% | 58.00% |
| Full | 5.146 | 9.245 | 30.50% | 58.00% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 5.149 | 9.249 | 58.00% | 4.037 |
| 5x5 | 25 | 5.147 | 9.246 | 58.00% | 4.776 |
| 6x6 | 36 | 5.146 | 9.245 | 58.00% | 5.050 |
| 7x7 | 49 | 5.144 | 9.243 | 58.00% | 8.593 |
| 8x8 | 64 | 5.143 | 9.241 | 58.00% | 10.299 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; report the measured result and revise the method only through a new train-only validation study.
