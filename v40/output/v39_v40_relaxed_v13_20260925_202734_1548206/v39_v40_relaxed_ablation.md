# Four-city v13 4 m GT-assisted ablation, fresh v39 training with v40 limit fixes

This is a new experiment. The archived 3.759 m is a historical comparator, not an expected result.

## Component removal

| Variant | Frames | MLE (m) ↓ | LSR@5 ↑ |
|---|---:|---:|---:|
| w/o GRU | 1482 | 4.334 | 65.59% |
| w/o Kalman | 1482 | 4.068 | 71.12% |
| w/o final Mean Shift | 1482 | 7.920 | 23.55% |
| Full | 1482 | 4.076 | 70.99% |

## Temporal input

| Input | MLE (m) ↓ | LSR@5 ↑ |
|---|---:|---:|
| 1 frame | 3.883 | 73.75% |
| 2 frames | 3.931 | 73.48% |
| Full | 4.076 | 70.99% |

## Final Mean Shift window

| Window | MLE (m) ↓ | LSR@5 ↑ |
|---|---:|---:|
| 4x4 | 4.760 | 60.59% |
| 5x5 | 4.081 | 70.72% |
| 6x6 | 4.076 | 70.99% |
| 7x7 | 4.001 | 72.00% |
| 8x8 | 4.001 | 72.00% |

Other pooled R@1*, LSR@15, HSR@15, MLE and MHE metrics are in `paper_ablation_tables.md`.
R@1* is a same-quadrant proxy. These synthetic routes are not recorded flights.
GT is a current-frame search input; these are GT-assisted results.
