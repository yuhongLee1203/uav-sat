# New Temporal Architecture: Final RTL-CRF Window Ablation

| Run | Method | Frames | MLE_m | MedLE_m | P90_m | P95_m | P99_m | CVaR90_m | MaxLE_m | LSR@5_pct | LSR@10_pct | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct | JumpThreshold_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_train_A_test_BC_t2only_w3 | RTL_CRF | 3530 | 5.41 | 5.03 | 9.40 | 10.92 | 13.30 | 11.26 | 18.15 | 49.66 | 92.52 | 99.72 | 100.00 | 5.07 | 2.47 | 13.12 |
| strict_train_A_test_BC_t2only_w4 | RTL_CRF | 3528 | 5.12 | 4.72 | 8.85 | 10.31 | 12.89 | 10.74 | 16.16 | 54.54 | 94.30 | 99.80 | 100.00 | 4.68 | 2.58 | 13.12 |
| strict_train_A_test_BC_t2only_w5 | RTL_CRF | 3526 | **5.02** | 4.53 | **8.85** | 10.34 | 13.24 | 10.73 | 16.02 | 56.61 | 94.36 | **99.86** | 100.00 | **4.46** | **1.70** | 13.12 |
