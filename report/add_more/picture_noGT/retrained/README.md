# 無 GT 模型：四城市全部 test route 結果圖

本次使用四城市 validation 選定的固定 6×6 模型；八條 test route 僅在設定固定後推論，沒有用 test GT 選模型或搜尋位置。GT 只用於事後計算誤差及畫綠色顯示路徑。
模型仍使用外部提供的 route/path reference；「無 GT」指當前幀 GT 不進模型。
City A 使用先前已在 train split 以 GT-loss 訓練的視覺權重；City B/C/D 在本輪各自重新訓練視覺模型。四城市的共用時序模型在本輪從隨機初始化重新訓練，checkpoint 依四城市 validation MLE 選定，最後才評估 test。
綠色 GT 路徑僅供顯示，以 30 m Ramer–Douglas–Peucker 簡化成較直的線段；所有數值用未簡化的逐幀 GT 座標。紅色預測線僅供顯示，以 40 m 容差保留主要轉向並移除逐幀鋸齒；逐幀誤差與全部指標使用未簡化的原始預測。四城市 test_01 的官方 waypoint 像素路線相同，但衛星底圖、測試 UAV 影像與預測不同。
這次使用 v40 official96 固定 test split，共 613 幀；它與 v39 的 1,482 幀 GT-assisted replay 是不同資料／推論設定，不能直接以 3.76 m 對照。

## 測試數值

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | test_01 | 64 | 4.93 | 8.55 | 95.31 |
| A | test_02 | 89 | 3.97 | 6.42 | 97.75 |
| B | test_01 | 64 | 10.51 | 20.64 | 76.56 |
| B | test_02 | 72 | 10.85 | 20.31 | 80.56 |
| C | test_01 | 64 | 8.09 | 15.87 | 87.50 |
| C | test_02 | 98 | 7.59 | 12.49 | 92.86 |
| D | test_01 | 64 | 11.14 | 23.32 | 75.00 |
| D | test_02 | 98 | 7.48 | 13.53 | 91.84 |
| Overall | — | 613 | 7.88 | 16.30 | 88.09 |

## 圖片

- [八條路線合圖](all_eight_routes_noGT.png)（每格是對應城市的完整衛星圖）
- 各 route 圖有完整衛星圖、路線局部衛星圖和逐幀誤差：

- City A: [test_01](citya_test_01_noGT.png) · [test_02](citya_test_02_noGT.png)
- City B: [test_01](cityb_test_01_noGT.png) · [test_02](cityb_test_02_noGT.png)
- City C: [test_01](cityc_test_01_noGT.png) · [test_02](cityc_test_02_noGT.png)
- City D: [test_01](cityd_test_01_noGT.png) · [test_02](cityd_test_02_noGT.png)

## 可稽核來源

- 每個城市的 `inference/city*/report.json` 記錄 checkpoint、gallery 來源與測試摘要；`inference/city*/city*/route_*_frames.csv`（實際檔名見 report.json 的 `CSV`）保留每幀 GT 與預測。
- 每個城市的 `run.log` 保留執行輸出；train/validation/test 各階段權重來源見正式報告。
- 所有八條路線的誤差序列雜湊互不相同；四座城市的衛星圖雜湊互不相同。顯示線段數與容差見 [display_path_audit.json](display_path_audit.json)；完整訓練／驗證／測試來源見 [training_validation_test_provenance.json](training_validation_test_provenance.json)。
