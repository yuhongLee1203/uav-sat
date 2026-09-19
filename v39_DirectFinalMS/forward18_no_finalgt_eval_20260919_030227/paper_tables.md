# Forward18 eval-only, Final-MS direct GT/reference prior disabled

> Reused existing trained checkpoints. No temporal retraining was performed.

> Important: the front local search still uses the controlled GT+jitter prior; only the direct GT/reference term inside Final MS is disabled.

## Component ablation

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| Full | 5.205 | 4.060 | 4.798 | 8.934 | 56.42% |
| w/o GRU | 4.870 | 3.730 | 4.464 | 8.618 | 61.49% |
| w/o Kalman | 6.303 | 5.103 | 5.876 | 10.713 | 45.78% |
| w/o Final MS | 4.629 | 3.975 | 4.396 | 8.138 | 61.49% |

## Temporal frames

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.172 | 4.033 | 4.767 | 8.930 | 57.05% |
| 2 | 5.209 | 4.061 | 4.800 | 8.920 | 56.48% |
| 3 | 5.206 | 4.060 | 4.798 | 8.940 | 56.45% |

## Final MeanShift window

| Window | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency |
|---|---:|---:|---:|---:|
| 4x4 | 4.811 | 8.921 | 56.20% | 3.591 ms |
| 5x5 | 4.798 | 8.940 | 56.45% | 5.432 ms |
| 6x6 | 4.803 | 8.948 | 56.37% | 7.094 ms |
| 7x7 | 4.798 | 8.940 | 56.48% | 10.489 ms |
| 8x8 | 4.803 | 8.945 | 56.42% | 11.402 ms |

## Full-model E2E runtime

- Mean: **34.561 ms/frame**
- FPS: **28.93**
