# Paper Core Tables

## Table 1. Core component ablation

| Label | MLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct |
| --- | --- | --- | --- | --- | --- |
| w/o temporal GRU | 4.158 | 7.053 | 68.792 | 99.932 | 0.882 |
| w/o final MeanShift | 6.658 | 11.497 | 34.600 | 98.168 | 7.598 |
| Full (3-frame) | 3.658 | 6.502 | 76.866 | 100.000 | 0.746 |

## Table 2. Search-region efficiency

| Search | Candidates | MLE_m | P90_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| Full 6x6 | 36 | 3.448 | 6.273 | 100.000 | 60.099 | 16.639 |
| Forward 3x6 | 18 | 3.658 | 6.502 | 99.050 | 47.553 | 21.029 |

## Table 3. Temporal context (single seed; multi-seed table should be used for the final paper)

| FramesInput | MLE_m | P90_m | LSR@5_pct | LSR@15_pct | JumpRate_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.641 | 6.392 | 76.934 | 100.000 | 0.475 |
| 2 | 3.634 | 6.379 | 77.544 | 100.000 | 0.543 |
| 3 | 3.658 | 6.502 | 76.866 | 100.000 | 0.746 |

## Supplementary. Final MeanShift grid size

| Grid | MLE_m | P90_m | LSR@5_pct | InferenceMean_ms |
| --- | --- | --- | --- | --- |
| 4x4 | 4.131 | 7.457 | 68.860 | 52.939 |
| 5x5 | 3.659 | 6.502 | 76.662 | 52.674 |
| 6x6 | 3.658 | 6.502 | 76.866 | 47.553 |
| 7x7 | 3.636 | 6.434 | 77.544 | 53.811 |
| 8x8 | 3.637 | 6.435 | 77.544 | 58.062 |

## Notes

- Top-1 and prior-jitter sensitivity are not included in paper-facing tables.
- The 1/2/3-frame rows use separately trained temporal checkpoints.
- Do not change or select settings using nav50/nav51 to force a preferred ordering.
- Current Full results still use the controlled local-prior/jitter protocol; removing a sensitivity table does not make the inference GT-free.
