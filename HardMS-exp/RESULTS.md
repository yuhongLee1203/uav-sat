# HardMS and Temporal Result Summary

All values are archived experiment outputs or recomputed directly from their per-frame predictions.

## 1. Archived single-frame HardMS controls (B+C, 3,534 frames)
| Method | N | MLE | MedLE | P90 | CVaR90 | LSR@10 | LSR@15 | LSR@20 | MaxLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top-1 patch center | 3534 | 14.75 | 14.34 | 24.86 | 27.44 | 29.82 | 53.11 | 72.58 | 36.26 |
| Fixed HardMS (snapped anchor) | 3534 | **11.46** | **11.20** | **18.23** | **20.89** | **41.14** | **73.46** | **93.80** | **29.91** |

## 2. New temporal architecture final output (B+C, 3,526 frames)
| Method | Frames | MLE_m | MedLE_m | P90_m | CVaR90_m | LSR@10_pct | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RawTop1 | 3526 | 14.70 | 14.31 | 23.73 | 26.25 | 26.23 | 53.80 | 77.03 | 13.88 | 46.23 |
| FixedHardMS | 3526 | 10.13 | 9.72 | 16.81 | 19.40 | 51.73 | 83.01 | 96.74 | 11.54 | 42.96 |
| RTL_CRF | 3526 | **5.04** | **4.56** | **8.88** | **10.79** | **94.10** | **99.83** | **100.00** | **4.42** | **1.76** |

## 3. Common-frame jump comparison (3,526 frames)
| Method | Decoder | MLE_m | P90_m | CVaR90_m | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Our visual branch | RawTop1 | 14.70 | 23.73 | 26.25 | 53.80 | 77.03 | 13.88 | 46.23 |
| Our visual branch | FixedHardMS | **10.13** | **16.81** | **19.40** | **83.01** | **96.74** | 11.54 | 42.96 |
| Sample4Geo-style (adapted) | Raw Top-1 | 14.88 | 24.57 | 27.37 | 51.02 | 73.79 | 18.08 | 62.37 |
| DenseUAV-style (adapted) | Raw Top-1 | 14.93 | 24.40 | 27.06 | 50.74 | 73.74 | 18.79 | 65.30 |
| Game4Loc-style (adapted) | Raw Top-1 | 14.22 | 24.08 | 26.69 | 54.68 | 77.06 | 17.78 | 64.02 |
| Bearing-UAV (archived; trimmed) | Coordinate regression | 17.31 | 28.06 | 31.64 | 41.55 | 64.52 | **9.30** | **16.20** |

`JumpRate`: a predicted adjacent-frame displacement greater than the route-specific 99th-percentile GT displacement plus 3 m. External rows are local-36 adaptations; Bearing-UAV's value is recomputed here from archived predictions, not copied from its paper.

## 4. Final RTL-CRF temporal-window ablation
| Run | Method | Frames | MLE_m | P90_m | LSR@15_pct | RPE_m | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict_train_A_test_BC_t2only_w3 | RTL_CRF | 3530 | 5.41 | 9.40 | 99.72 | 5.07 | 2.47 |
| strict_train_A_test_BC_t2only_w4 | RTL_CRF | 3528 | 5.12 | 8.85 | 99.80 | 4.68 | 2.58 |
| strict_train_A_test_BC_t2only_w5 | RTL_CRF | 3526 | 5.02 | 8.85 | 99.86 | 4.46 | 1.70 |

## 5. Archived decoder overhead
| batch_size | num_candidates | top1_decoder_ms_per_batch | hardms_decoder_ms_per_batch | hardms_extra_ms_per_frame | hardms_iterations |
| --- | --- | --- | --- | --- | --- |
| 64 | 36 | 0.01 | 1.07 | 0.02 | 3 |

## 6. Archived Fixed HardMS seed stability
| Seed | Split | MLE | P90 | CVaR90 | LSR@15 | LSR@20 |
| --- | --- | --- | --- | --- | --- | --- |
| 2027 | model_dataset_new_2_flight+model_dataset_flight | 11.46 | 18.23 | 20.89 | 73.46 | 93.80 |
| 2027 | model_dataset_new_2_flight | 11.63 | 18.62 | 21.31 | 72.14 | 93.06 |
| 2027 | model_dataset_flight | 11.16 | 17.57 | 20.02 | 75.83 | 95.15 |
| 2028 | model_dataset_new_2_flight+model_dataset_flight | 11.47 | 18.29 | 20.93 | 73.94 | 93.60 |
| 2028 | model_dataset_new_2_flight | 11.59 | 18.65 | 21.19 | 72.80 | 93.06 |
| 2028 | model_dataset_flight | 11.26 | 17.71 | 20.36 | 75.99 | 94.59 |
| 2029 | model_dataset_new_2_flight+model_dataset_flight | 11.50 | 18.41 | 21.01 | 73.32 | 93.66 |
| 2029 | model_dataset_new_2_flight | 11.65 | 18.80 | 21.36 | 71.79 | 93.01 |
| 2029 | model_dataset_flight | 11.23 | 17.53 | 20.22 | 76.07 | 94.83 |

## 7. Archived grid-size and training-scale ablations
| GridSize | Split | MLE | P90 | CVaR90 | LSR@15 | LSR@20 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | model_dataset_new_2_flight+model_dataset_flight | 7.63 | 11.91 | 13.12 | 99.12 | 100.00 |
| 4 | model_dataset_new_2_flight | 7.65 | 11.93 | 13.13 | 99.08 | 100.00 |
| 4 | model_dataset_flight | 7.60 | 11.85 | 13.10 | 99.21 | 100.00 |
| 8 | model_dataset_new_2_flight+model_dataset_flight | 16.16 | 27.28 | 31.14 | 48.36 | 69.33 |
| 8 | model_dataset_new_2_flight | 16.31 | 27.81 | 31.64 | 47.19 | 68.85 |
| 8 | model_dataset_flight | 15.88 | 26.24 | 30.05 | 50.48 | 70.19 |
| 10 | model_dataset_new_2_flight+model_dataset_flight | 20.82 | 36.28 | 41.21 | 35.54 | 51.61 |
| 10 | model_dataset_new_2_flight | 21.30 | 37.08 | 41.91 | 34.31 | 49.91 |
| 10 | model_dataset_flight | 19.95 | 35.48 | 39.80 | 37.76 | 54.69 |

| TrainDataPercent | Split | MLE | P90 | CVaR90 | LSR@15 | LSR@20 |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | model_dataset_new_2_flight+model_dataset_flight | 11.24 | 17.22 | 19.43 | 76.88 | 96.69 |
| 25 | model_dataset_new_2_flight | 11.40 | 17.65 | 19.93 | 75.66 | 95.74 |
| 25 | model_dataset_flight | 10.95 | 16.84 | 18.46 | 79.09 | 98.41 |
| 50 | model_dataset_new_2_flight+model_dataset_flight | 11.29 | 17.51 | 19.93 | 75.92 | 95.81 |
| 50 | model_dataset_new_2_flight | 11.48 | 17.80 | 20.41 | 74.12 | 94.82 |
| 50 | model_dataset_flight | 10.95 | 17.07 | 18.95 | 79.17 | 97.62 |
| 75 | model_dataset_new_2_flight+model_dataset_flight | 11.40 | 17.94 | 20.60 | 74.48 | 94.51 |
| 75 | model_dataset_new_2_flight | 11.58 | 18.39 | 21.03 | 72.98 | 93.80 |
| 75 | model_dataset_flight | 11.07 | 17.42 | 19.71 | 77.19 | 95.79 |

The temporal model uses a controlled GT-jitter local prior and contiguous frame windows. It is therefore reported separately from independent-frame retrieval methods rather than claimed as a directly interchangeable baseline.
