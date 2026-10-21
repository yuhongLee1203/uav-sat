# Temporal-context V3 split-gate evaluation

GRU input: temporal mean + delta + delta2 + SAT context + direct Forward18 SoftMS visual position + previous state.
No position-innovation feature is used. Split GRU gains are selected on Route-A validation only and locked before B/C evaluation.

## Component ablation

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| Full | 1.801 | 1.490 | 1.690 | 3.315 | 98.27% |
| w/o GRU | 1.809 | 1.497 | 1.698 | 3.346 | 98.05% |
| w/o Kalman | 2.269 | 1.833 | 2.114 | 3.993 | 97.09% |
| w/o Final MS | 4.224 | 3.577 | 3.994 | 7.791 | 67.94% |

## Temporal context ablation

All rows reuse the same 3-frame-trained checkpoint; 1/2-frame rows mask older context at inference.

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.804 | 1.486 | 1.691 | 3.327 | 98.25% |
| 2 | 1.803 | 1.486 | 1.690 | 3.325 | 98.27% |
| 3 | 1.801 | 1.490 | 1.690 | 3.318 | 98.30% |

## Final MeanShift window sensitivity

The main method uses 6x6; all Route-A-selected GRU/MS parameters are held fixed across windows.

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 1.848 | 3.771 | 95.13% |
| 5x5 | 25 | 1.689 | 3.316 | 98.30% |
| 6x6 | 36 | 1.690 | 3.318 | 98.30% |
| 7x7 | 49 | 1.687 | 3.313 | 98.30% |
| 8x8 | 64 | 1.688 | 3.314 | 98.30% |
