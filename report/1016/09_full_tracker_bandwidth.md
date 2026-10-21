# 完整 tracker 重播：Final MS bandwidth

| Bandwidth (m) | N | MLE (m) | LSR@3 (%) | LSR@5 (%) | LSR@15 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 1482 | 3.7591 | 41.90 | 73.75 | 100.00 |
| 4 | 1482 | 3.8045 | 40.89 | 73.01 | 100.00 |
| 14 | 1482 | 3.7506 | 42.44 | 74.16 | 100.00 |

這三組重新執行完整 tracker：18 patch、三幀 GRU、Kalman、最終6×6MS。每條route從新狀態開始。使用相同歷史 checkpoint／prepared資料／calibration，只有 bandwidth 改變，GPU0。7m baseline逐條 route 與原3.759m run之MLE差異<1e-5m。這是原 GT-assisted test 上的探索性 sensitivity，沒有重新訓練，也不能用 test最佳列替換未獨立驗證的新模型主結果。

raw outputs / logs: `full_tracker_bandwidth/bw*/city*/variants/full/`。
