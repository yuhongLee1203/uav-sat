# v40：僅保留最佳、固定 6×6 最佳及最新四城市版本

## 目前最好的結果用哪個檔案訓練？

「最好」先固定在**不輸入當前幀 GT、同一條 CityA val_01 驗證路線（63 幀）**比較。GT 輔助 oracle 的較小數字不列入；不同城市、不同 test route、診斷 ensemble 也不混合排名。

| 保留版本 | 結果 | 實際訓練檔案 | 證據 |
|---|---|---|---|
| CityA 驗證最低 MLE，最終 MS 18×18 | **4.402212m** | 視覺入口 `train_visual_seed2059.py` 呼叫 `train_visual_backbone_gtloss.py`；GRU `train_multicity_routes.py` | [原始報告](checkpoint1/best_no_gt_20260926/report.json)／[視覺訓練參數](checkpoint1/best_no_gt_20260926/visual_training_report.json)／[GRU 訓練參數](checkpoint1/best_no_gt_20260926/temporal_training_settings.json) |
| 原固定 6×6 MS 架構的最低驗證 MLE | **4.666516m** | 視覺 `train_visual_backbone_gtloss.py`，seed 2061；GRU `train_multicity_routes.py` | [報告](output/no_gt_6x6_20261008_ms_sweep/bandwidth18/report.json)／[視覺訓練](output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/report.json)／[GRU 設定](output/no_gt_6x6_20260927_a/temporal_visual2061/raw10_sigma4/train/run_settings.json) |
| 最新，2026-10-09 四城市重訓，固定 6×6 | **八條 test route、613 幀，MLE 7.881161m、LSR@15 88.091354%** | 入口 `run_no_gt_multicity_retrain.py`；實際視覺 `train_visual_global_gtloss.py` + `train_visual_backbone_gtloss.py`；GRU `train_multicity_routes.py` | [逐路線與整體數據](retained/latest_abcd/test_metrics.csv)／[訓練來源](retained/latest_abcd/training_validation_test_provenance.json) |

4.4022m／4.6665m 都是重複開發過的單條驗證結果，**不是獨立測試或官方 Table 6**。最新四城市 test 結果不能與它們直接判定高低。最新 CityA test_02 的 3.9715m 是另一條路線，不冒充同一驗證基準的改進。最新四城市 GRU 在沒有 TTA 的訓練驗證中選 epoch 8（8.135549m）；正式測試另使用四轉角 TTA，請勿混合分母與推論設定。

## 保留目錄

- `checkpoint1/best_no_gt_20260926/`：4.4022m 的成對權重、原始報告、父 checkpoint、GT 擾動檢查和訓練設定。
- `checkpoint1/code_before_fusion/`：得到該歷史最佳結果時的凍結程式快照；只作歷史版本來源，保留當時的原始路徑記錄。
- `output/no_gt_6x6_20260927_a/`：只留下固定 6×6 最佳視覺／GRU 訓練結果和跨橫向 prior=0 的設定，其他候選分支已移出。
- `output/no_gt_6x6_20261008_ms_sweep/bandwidth18/`：同權重、bandwidth 18 的最低 6×6 驗證結果。
- `output/no_gt_multicity_retrain_20261009/`：最新四城市的視覺、GRU、validation 選 checkpoint 紀錄。
- `retained/latest_abcd/`：從原始逐幀 `error_final_m` 重新計算確認的八條 test 數據及原始預測 CSV。原圖仍在 `../report/add_more/picture_noGT/retrained/`。
- `runtime/`、14 個根目錄 Python 檔及 `data/official96_fixed_routes`、`data/official96_fixed_train_grid`：上述版本的共同必要依賴，沒有為了刪舊版而刪掉仍需使用的程式／資料。

四城市 latest 的 CityA 視覺權重正是上述最佳 6×6 視覺權重，因此不能把它當成「舊檔」刪除。CityB/C/D 的視覺模型及共享 GRU 才是在 10-09 本輪重新訓練。保留的 10 個實際權重／引用均已核對 SHA256，清理前後一致；只保留於本機，不上傳 GitHub。

## 最新版本的訓練入口

```bash
cd /yh/study/uav-sat
V40_MULTICITY_OUTPUT=/yh/study/uav-sat/v40/output/new_abcd_run python3 v40/run_no_gt_multicity_retrain.py
```

該程式按原配置使用 GPU 0／5／6；以上只是入口說明，本次清理沒有啟動訓練。自訂新目錄可避免混入現有結果。訓練後用相同 `V40_MULTICITY_OUTPUT`，並設定新的 `V40_MULTICITY_EVAL_OUTPUT` 結果目錄，呼叫 `run_no_gt_multicity_evaluate.py`；環境須有原 README 所用的 PyTorch／torchvision／open_clip、Bearing-UAV-90K 原始資料，以及本機保留的 parent CityA 視覺權重。

## GT、清理與上傳範圍

GT 用於訓練 loss 和事後誤差；當前幀 GT 不作推論輸入，固定路線 waypoint 仍是已知參考。這些是 Bearing-UAV-90K 官方 metadata 切分內組裝的 synthetic routes，不是原生連續航拍，也不是 TravelUAV 的新訓練。

335 個舊實驗／程式／快取入口已移出 v40，放在 `/yh/study/uav-sat_archives/v40_retired_20261021/`。因此 v40 已精簡，誤刪時仍可找回；封存目錄在 repository 外，不會上傳。完整清單與保留權重 SHA256 見 [RETENTION_MANIFEST.json](RETENTION_MANIFEST.json)。GitHub main 的舊 v40 檔案會移除；既有歷史 commit 不改寫。上傳仍排除權重及大型檔案，大型數值 CSV 使用無損 gzip。
