# Frozen V5 paper tables

## Main localization by city

| City | Recall@1* | MLE m | MedLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|---:|---:|
| citya | 88.828 | 3.943 | 3.716 | 6.805 | 71.390 | 100.000 |
| cityb | 28.324 | 3.290 | 2.964 | 5.875 | 82.659 | 100.000 |
| cityc | 25.130 | 3.667 | 3.327 | 6.631 | 76.684 | 100.000 |
| cityd | 21.867 | 3.565 | 3.341 | 5.929 | 81.067 | 99.733 |
| ALL_4_CITIES | 40.909 | 3.621 | 3.346 | 6.484 | 77.883 | 99.932 |

*Recall@1 is derived with the Bearing-UAV four-RST same-quadrant decision criterion from continuous localization output.

## Bearing-UAV-aligned comparison

| Method | Recall@1 | LSR@15 | HSR@15 | MLE m | MedLE m | MHE deg | MedHE deg | SR@20 | SPL | NE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| University-1652 | 60.200 | 15.110 | — | 33.150 | — | — | — | 0.000 | — | 602.960 |
| SUES-200 | 66.600 | 15.760 | — | 30.830 | — | — | — | 0.000 | — | 618.850 |
| DenseUAV | 73.430 | 16.540 | — | 28.790 | — | — | — | 0.000 | — | 651.930 |
| GTA-UAV | 70.710 | 27.960 | — | 28.430 | — | — | — | 0.000 | — | 661.910 |
| Bearing-UAV (VGG-16) | 83.170 | 89.360 | 77.210 | 8.610 | 7.300 | 12.900 | 7.200 | 50.000 | — | 275.610 |
| Yours (Frozen V5 Forward18 + 3f GRU + Kalman + FinalMS) | 40.909 | 99.932 | — | 3.621 | 3.346 | — | — | — | — | — |

**Protocol:** Ours uses controlled local-prior temporal refinement. Bearing-UAV uses four-adjacent-RST pose regression; navigation values are closed-loop Bearing-Naver. Therefore Ours camera-heading and closed-loop navigation cells are intentionally not fabricated.

## Core ablation pooled across four cities

| Variant | MLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|
| full | 3.621 | 6.484 | 77.883 | 99.932 |
| no_gru | 4.102 | 7.011 | 69.878 | 99.932 |
| no_kalman | 3.587 | 6.433 | 78.290 | 100.000 |
| no_ms | 6.559 | 11.475 | 37.449 | 98.100 |

## Temporal ablation pooled across four cities

| Variant | MLE m | P90 m | LSR@5 | LSR@15 |
|---|---:|---:|---:|---:|
| full | 3.621 | 6.484 | 77.883 | 99.932 |
| frames1 | 3.597 | 6.431 | 78.155 | 99.932 |
| frames2 | 3.604 | 6.423 | 77.680 | 100.000 |

## Output files

- `table_bearinguav_comparison.csv`: paper comparison table
- `table_main_per_city.csv`: City A/B/C/D main results
- `table_ablation_core_pooled.csv`: GRU/Kalman/MeanShift/Full
- `table_temporal_context_pooled.csv`: 1f/2f/3f
- `table_grid_pooled.csv`: final MeanShift window
- `table_motion_heading_diagnostic.csv`: motion-heading only
- `table_route_replay_navigation.csv`: offline replay diagnostic only
