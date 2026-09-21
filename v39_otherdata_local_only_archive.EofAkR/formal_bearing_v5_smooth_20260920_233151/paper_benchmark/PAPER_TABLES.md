# Bearing-UAV aligned paper tables

## A. Localization comparison (paper-facing)

| Method | Recall@1_UAV_pct | LSR@15_UAV_pct | MLE_UAV_m | MedLE_UAV_m |
| --- | --- | --- | --- | --- |
| University-1652 | 60.200 | 15.110 | 33.150 | — |
| SUES-200 | 66.600 | 15.760 | 30.830 | — |
| DenseUAV | 73.430 | 16.540 | 28.790 | — |
| GTA-UAV | 70.710 | 27.960 | 28.430 | — |
| Bearing-UAV (VGG-16) | 83.170 | 89.360 | 8.610 | 7.300 |
| Yours (Forward-18 + GRU + Kalman + SoftMS) | 97.286 | 100.000 | 3.658 | 3.366 |

## B. Camera-heading comparison (Bearing definition)

| Method | HSR@15_camera_pct | MHE_camera_deg | MedHE_camera_deg |
| --- | --- | --- | --- |
| University-1652 | — | — | — |
| SUES-200 | — | — | — |
| DenseUAV | — | — | — |
| GTA-UAV | — | — | — |
| Bearing-UAV (VGG-16) | 77.210 | 12.900 | 7.200 |
| Yours (current model) | — | — | — |

## C. Current model motion-heading diagnostic (NOT camera yaw)

| Method | MotionMHE_deg | MotionMedHE_deg | MotionHSR@15_pct |
| --- | --- | --- | --- |
| Yours (motion/ground-track heading diagnostic) | 37.404 | 23.698 | 37.381 |

## D. Per-city localization + motion-heading diagnostic

| City | Recall@1_4RST_derived_pct | MLE_m | MedLE_m | LSR@15_pct | MotionMHE_deg | MotionHSR@15_pct |
| --- | --- | --- | --- | --- | --- | --- |
| citya | 97.820 | 4.065 | 3.904 | 100.000 | 36.739 | 40.599 |
| cityb | 97.110 | 3.400 | 3.071 | 100.000 | 39.069 | 36.127 |
| cityc | 97.409 | 3.674 | 3.344 | 100.000 | 37.419 | 35.492 |
| cityd | 96.800 | 3.483 | 3.256 | 100.000 | 36.502 | 37.333 |

## E. Bearing-Naver closed-loop reference

| Method | SR@20_UAV_pct | SPL_UAV_pct | NE_UAV_m |
| --- | --- | --- | --- |
| University-1652 | 0.000 | 0.000 | 602.960 |
| SUES-200 | 0.000 | 0.000 | 618.850 |
| DenseUAV | 0.000 | 0.000 | 651.930 |
| GTA-UAV | 0.000 | 0.000 | 661.910 |
| Bearing-UAV (VGG-16) | 50.000 | 29.820 | 275.610 |
| Yours (current offline replay) | — | — | — |

## F. Offline route-replay diagnostic

| City | Route | NE_m_route_replay | SR@20_pct_route_replay | SPL_pct_route_replay | JumpRate_pct | MaxFinalStep_m |
| --- | --- | --- | --- | --- | --- | --- |
| citya | nav50 | 2.706 | 100.000 | 100.000 | 0.000 | 21.150 |
| citya | nav51 | 6.484 | 100.000 | 100.000 | 1.951 | 19.903 |
| cityb | nav50 | 9.128 | 100.000 | 100.000 | 0.625 | 22.758 |
| cityb | nav51 | 10.666 | 100.000 | 100.000 | 2.151 | 22.662 |
| cityc | nav50 | 0.731 | 100.000 | 100.000 | 0.000 | 19.882 |
| cityc | nav51 | 6.818 | 100.000 | 100.000 | 0.441 | 24.657 |
| cityd | nav50 | 1.036 | 100.000 | 100.000 | 0.000 | 22.449 |
| cityd | nav51 | 8.283 | 100.000 | 100.000 | 0.426 | 22.654 |

## G. Efficiency

| Method | OnDiskCheckpointSize_MB_per_city | EndToEndInferenceMean_ms | FPS | GFLOPs |
| --- | --- | --- | --- | --- |
| Yours | 64.194 | 47.553 | 21.029 | — |

## Required protocol notes

- Current localization is a controlled local-prior/jitter experiment; do not claim fully GT-free deployment.
- Our Recall@1 is explicitly 4-RST-derived from continuous XY.
- Current MHE/HSR are ground-track motion-heading metrics, not Bearing-UAV camera-heading metrics.
- Route-replay SR/SPL/NE are not closed-loop Bearing-Naver metrics.
