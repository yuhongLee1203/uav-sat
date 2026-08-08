# Full External Localization Comparison

| Method | Frames | MLE_m | MedLE_m | P90_m | CVaR90_m | LSR@5_pct | LSR@10_pct | LSR@15_pct | LSR@20_pct | R@1_pct | MRR | Online_ms | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileCLIP basic (adapted) | 3534 | 15.17 | 14.99 | 24.84 | 27.77 | 8.77 | 26.37 | 50.08 | 71.90 | 3.31 | 0.13 | 48.19 | nan |
| Sample4Geo-style (adapted) | 3534 | 14.86 | 14.88 | 24.56 | 27.36 | 9.54 | 28.10 | 51.10 | 73.85 | 3.17 | 0.12 | **1.43** | nan |
| DenseUAV-style (adapted) | 3534 | 14.91 | 14.85 | 24.39 | 27.05 | 7.98 | 27.76 | 50.79 | 73.77 | 3.17 | 0.12 | 4.03 | nan |
| Game4Loc-style (adapted) | 3534 | 14.21 | 13.97 | 24.07 | 26.68 | 9.65 | 31.86 | 54.70 | 77.11 | **3.42** | **0.13** | 3.97 | nan |
| Archived Fixed HardMS | 3534 | **11.46** | **11.20** | **18.23** | **20.89** | **11.04** | **41.14** | **73.46** | **93.80** | nan | nan | 47.85 | nan |
| Bearing-UAV (archived) | 3534 | 17.29 | 16.71 | 28.05 | 31.63 | 4.75 | 18.51 | 41.65 | 64.60 | nan | nan | nan | nan |
