# Frozen V5 paper tables

## Main localization by city

| City | Recall@1* | MLE m | MedLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|---:|---:|
| citya | 92.632 | 4.133 | 3.734 | 7.245 | 66.053 | 100.000 |
| cityb | 95.268 | 3.648 | 3.328 | 6.545 | 75.079 | 100.000 |
| cityc | 94.652 | 3.611 | 3.374 | 6.295 | 74.866 | 100.000 |
| cityd | 94.161 | 4.092 | 3.846 | 6.929 | 72.263 | 100.000 |
| ALL_4_CITIES | 94.130 | 3.886 | 3.582 | 6.804 | 71.930 | 100.000 |

*Recall@1 is derived with the Bearing-UAV four-RST same-quadrant decision criterion from continuous localization output.

## Bearing-UAV-aligned comparison

| Method | Recall@1 | LSR@15 | HSR@15 | MLE m | MedLE m | MHE deg | MedHE deg | SR@20 | SPL | NE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| University-1652 | 60.200 | 15.110 | — | 33.150 | — | — | — | 0.000 | — | 602.960 |
| SUES-200 | 66.600 | 15.760 | — | 30.830 | — | — | — | 0.000 | — | 618.850 |
| DenseUAV | 73.430 | 16.540 | — | 28.790 | — | — | — | 0.000 | — | 651.930 |
| GTA-UAV | 70.710 | 27.960 | — | 28.430 | — | — | — | 0.000 | — | 661.910 |
| Bearing-UAV (VGG-16) | 83.170 | 89.360 | 77.210 | 8.610 | 7.300 | 12.900 | 7.200 | 50.000 | — | 275.610 |
| Yours (Frozen V5 Forward18 + 3f GRU + Kalman + FinalMS) | 94.130 | 100.000 | — | 3.886 | 3.582 | — | — | — | — | — |

**Protocol:** Ours uses controlled local-prior temporal refinement. Bearing-UAV uses four-adjacent-RST pose regression; navigation values are closed-loop Bearing-Naver. Therefore Ours camera-heading and closed-loop navigation cells are intentionally not fabricated.

## Core ablation pooled across four cities

| Variant | MLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|
| full | 3.886 | 6.804 | 71.930 | 100.000 |
| no_gru | 4.267 | 7.329 | 66.127 | 100.000 |
| no_kalman | 3.897 | 6.849 | 71.997 | 100.000 |
| no_ms | 7.040 | 12.170 | 31.039 | 98.178 |

## Temporal ablation pooled across four cities

| Variant | MLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|
| full | 3.886 | 6.804 | 71.930 | 100.000 |
| frames1 | 3.873 | 6.751 | 71.997 | 100.000 |
| frames2 | 3.878 | 6.772 | 71.862 | 100.000 |

## Output files

- `table_bearinguav_comparison.csv`: paper comparison table
- `table_main_per_city.csv`: City A/B/C/D main results
- `table_ablation_core_pooled.csv`: GRU/Kalman/MeanShift/Full
- `table_temporal_context_pooled.csv`: 1f/2f/3f
- `table_grid_pooled.csv`: final MeanShift window
- `table_motion_heading_diagnostic.csv`: motion-heading only
- `table_route_replay_navigation.csv`: offline replay diagnostic only
