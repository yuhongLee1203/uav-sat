## Table 1. Component removal ablation

| Setting | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|
| Full | 1.955 | 3.688 | 97.51% |
| w/o GRU | 1.806 | 3.503 | 97.65% |
| w/o Kalman | 2.347 | 4.253 | 94.45% |
| w/o Final MS | 4.300 | 8.033 | 64.18% |

## Table 2. Temporal-context input ablation

All rows reuse the same 3-frame-trained checkpoint; older temporal context is masked at inference.

| Frames | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|
| 1 | 1.951 | 3.706 | 97.48% |
| 2 | 1.952 | 3.705 | 97.48% |
| 3 | 1.955 | 3.688 | 97.51% |

## Table 3. Final MeanShift window sensitivity

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.080 | 4.155 | 94.79% |
| 5x5 | 25 | 1.955 | 3.685 | 97.54% |
| 6x6 | 36 | 1.955 | 3.688 | 97.51% |
| 7x7 | 49 | 1.954 | 3.684 | 97.57% |
| 8x8 | 64 | 1.954 | 3.686 | 97.57% |

## Table 4. End-to-end runtime

- Route B: 30.831 ms/frame
- Route C: 30.635 ms/frame
