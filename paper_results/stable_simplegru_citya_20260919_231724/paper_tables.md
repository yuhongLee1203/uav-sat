## Table 1. Component removal ablation

| Setting | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|
| Full | 1.828 | 3.483 | 97.96% |
| w/o GRU | 1.806 | 3.503 | 97.65% |
| w/o Kalman | 2.221 | 4.177 | 95.39% |
| w/o Final MS | 3.945 | 7.672 | 69.16% |

## Table 2. Temporal-context input ablation

All rows reuse the same 3-frame-trained checkpoint; older temporal context is masked at inference.

| Frames | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|
| 1 | 1.824 | 3.483 | 98.02% |
| 2 | 1.827 | 3.488 | 97.91% |
| 3 | 1.828 | 3.483 | 97.96% |

## Table 3. Final MeanShift window sensitivity

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 1.942 | 3.896 | 95.39% |
| 5x5 | 25 | 1.827 | 3.482 | 97.96% |
| 6x6 | 36 | 1.828 | 3.483 | 97.96% |
| 7x7 | 49 | 1.827 | 3.481 | 97.99% |
| 8x8 | 64 | 1.827 | 3.482 | 97.99% |

## Table 4. End-to-end runtime

- Route B: 29.940 ms/frame
- Route C: 30.026 ms/frame
