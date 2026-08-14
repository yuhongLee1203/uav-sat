# New Temporal Architecture: Main Comparison

| Run | Method | Frames | MLE_m | MedLE_m | P90_m | P95_m | P99_m | CVaR90_m | MaxLE_m | LSR@5_pct | LSR@10_pct | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct | JumpThreshold_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_train_A_test_BC_no_position_scale | RawTop1 | 3526 | 14.70 | 14.31 | 23.73 | 25.80 | 28.77 | 26.25 | 33.46 | 6.52 | 26.23 | 53.80 | 77.03 | 13.88 | 46.23 | 13.12 |
| strict_train_A_test_BC_no_position_scale | FixedHardMS | 3526 | 10.13 | 9.72 | 16.81 | 18.83 | 22.34 | 19.40 | 30.12 | 15.46 | 51.73 | 83.01 | 96.74 | 11.54 | 42.96 | 13.12 |
| strict_train_A_test_BC_no_position_scale | RTL_CRF | 3526 | **5.04** | **4.56** | **8.88** | 10.36 | 13.18 | **10.79** | 18.09 | **56.86** | **94.10** | **99.83** | **100.00** | **4.42** | **1.76** | 13.12 |
