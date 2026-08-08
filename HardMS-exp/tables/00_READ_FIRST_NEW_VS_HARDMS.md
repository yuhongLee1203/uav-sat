# 先看這裡：HardMS 與新架構總覽

**這份正式總覽只保留 Route A 訓練、Route B+C 測試的結果。資料來源已包含 `uav-sat/outputs/` 內所有符合此協議且已完成的 runs。**

## 舊 HardMS：單幀視覺定位（完整 B+C，3,534 frames）
| Method | MLE | P90 | CVaR90 | LSR@10 | LSR@15 | LSR@20 | MaxLE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top-1 patch center | 14.75 | 24.86 | 27.44 | 29.82 | 53.11 | 72.58 | 36.26 |
| Fixed HardMS (continuous mode; diagnostic) | **11.29** | **17.86** | **20.49** | **42.08** | **76.49** | **94.96** | 29.92 |
| Fixed HardMS (snapped anchor) | 11.46 | 18.23 | 20.89 | 41.14 | 73.46 | 93.80 | **29.91** |

## 新架構：最終 RTL-CRF 輸出（B+C，3,526 frames）
| Method | MLE_m | MedLE_m | P90_m | CVaR90_m | LSR@10_pct | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RawTop1 | 14.70 | 14.31 | 23.73 | 26.25 | 26.23 | 53.80 | 77.03 | 13.88 | 46.23 |
| FixedHardMS | 10.13 | 9.72 | 16.81 | 19.40 | 51.73 | 83.01 | 96.74 | 11.54 | 42.96 |
| RTL_CRF | **5.04** | **4.56** | **8.88** | **10.79** | **94.10** | **99.83** | **100.00** | **4.42** | **1.76** |

## 所有已完成時序 runs 中，MLE 最佳設定
| Run | Method | Frames | MLE_m | P90_m | CVaR90_m | LSR@15_pct | LSR@20_pct | RPE_m | JumpRate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_train_A_test_BC_no_position_scale_w4 | RTL_CRF | 3528 | **4.94** | **8.80** | **10.65** | **99.89** | **100.00** | **4.30** | **1.93** |

## 注意
- 新時序架構使用連續 frame 與 controlled GT-jitter local prior；它必須和單幀 HardMS 分開解讀。
- 正式表格只保留最終 `RTL_CRF`；中間 path expectation 不列為獨立方法。
- 更完整的 route-wise 與 final-window 消融在 `10–13` 表格。
