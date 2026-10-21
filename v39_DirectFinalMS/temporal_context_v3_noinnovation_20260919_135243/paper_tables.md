# Temporal-context v3 (no position innovation)

GRU input: temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state.
No visual-anchor minus predicted-position feature is used as GRU input.
Hyperparameters are selected on Route-A validation only, then locked for Route-B/C.

## Component ablation

| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|---:|
| Full | 1.815 | 1.499 | 1.702 | 3.327 | 98.22% |
| w/o GRU | 1.809 | 1.497 | 1.698 | 3.346 | 98.05% |
| w/o Kalman | 2.276 | 1.844 | 2.122 | 4.014 | 97.06% |
| w/o Final MS | 4.264 | 3.612 | 4.031 | 7.818 | 66.75% |

## Temporal context ablation

All rows reuse the same 3-frame-trained checkpoint; 1/2-frame rows mask older temporal context at inference.

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.817 | 1.496 | 1.703 | 3.324 | 98.19% |
| 2 | 1.816 | 1.496 | 1.702 | 3.324 | 98.22% |
| 3 | 1.814 | 1.499 | 1.702 | 3.327 | 98.25% |

## Final MeanShift window sensitivity

The main method uses 6x6. Route-A-selected parameters are held fixed for every window.

| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 1.860 | 3.776 | 95.05% |
| 5x5 | 25 | 1.702 | 3.330 | 98.25% |
| 6x6 | 36 | 1.702 | 3.327 | 98.25% |
| 7x7 | 49 | 1.700 | 3.328 | 98.27% |
| 8x8 | 64 | 1.701 | 3.327 | 98.27% |
