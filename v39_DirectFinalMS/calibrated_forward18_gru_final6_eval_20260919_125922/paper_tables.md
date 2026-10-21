# Calibrated Forward18 / GRU / Kalman / Final-6x6 evaluation

Hyperparameters were selected on Route-A validation only, then locked before Route-B/C evaluation.

## Component ablation

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| Full | 1.855 | 1.537 | 1.742 | 3.359 | 98.19% |
| w/o GRU | 1.809 | 1.497 | 1.698 | 3.346 | 98.05% |
| w/o Kalman | 2.301 | 1.891 | 2.155 | 4.050 | 96.43% |
| w/o Final MS | 4.381 | 3.699 | 4.138 | 7.803 | 65.05% |

## Temporal context ablation

All rows reuse the same 3-frame-trained checkpoint; 1/2-frame rows mask older context at inference.

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.853 | 1.536 | 1.740 | 3.369 | 98.16% |
| 2 | 1.853 | 1.536 | 1.740 | 3.370 | 98.19% |
| 3 | 1.855 | 1.537 | 1.742 | 3.359 | 98.19% |

## Final MeanShift window sensitivity

The 6x6 main-window hyperparameters were selected on Route-A validation and are held fixed for every window.

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 1.901 | 3.812 | 95.02% |
| 5x5 | 25 | 1.741 | 3.359 | 98.19% |
| 6x6 | 36 | 1.742 | 3.359 | 98.19% |
| 7x7 | 49 | 1.740 | 3.356 | 98.25% |
| 8x8 | 64 | 1.740 | 3.357 | 98.25% |
