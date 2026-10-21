# Bearing-UAV ICLR ablation (measured)

Evaluation preserves the existing Bearing-UAV controlled-GT reference protocol.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.946 | 9.195 | 32.50% | 60.00% | 92.50% |
| w/o Kalman | 4.067 | 7.480 | 42.50% | 69.50% | 96.00% |
| w/o MeanShift | 5.853 | 11.992 | 30.50% | 48.00% | 82.00% |
| Full | 4.927 | 8.940 | 32.00% | 60.00% | 92.00% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.926 | 8.940 | 32.00% | 60.00% |
| 2 frames | 4.926 | 8.940 | 32.00% | 60.00% |
| Full | 4.927 | 8.940 | 32.00% | 60.00% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.927 | 8.942 | 60.00% | 3.698 |
| 5x5 | 25 | 4.928 | 8.940 | 60.00% | 4.376 |
| 6x6 | 36 | 4.927 | 8.940 | 60.00% | 4.875 |
| 7x7 | 49 | 4.926 | 8.940 | 60.00% | 8.970 |
| 8x8 | 64 | 4.924 | 8.940 | 60.00% | 10.400 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; report the measured result and revise the method only through a new train-only validation study.
