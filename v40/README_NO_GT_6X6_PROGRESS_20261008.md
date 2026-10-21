# CityA 無當前幀 GT、最終 6×6 Mean Shift：研究進度（2026-10-08）

## 固定條件與可比較的基準

- 架構維持「每個搜尋視窗前方 18 個衛星 patch → 三幀融合 GRU → Kalman → 最終 6×6 Mean Shift」。UAV 三幀的 mean、diff1、diff2 經單一融合 MLP，沒有刪掉任何一幀。
- GT 僅用於訓練 loss、驗證標籤、以及離線建立 train-only route；驗證推論時所有 GT oracle 關閉。外部 supplied route/path reference 仍存在，故這不是沒有任何路線資訊的全球檢索。
- 固定開發集是 CityA `val_01` 的 63 幀。現有最佳固定 6×6 結果是 **4.6677 m MLE**（原設定 bandwidth 25 m）；在同權重、bandwidth 18 m 的診斷重播為 **4.6665 m**。這兩個數值都只是驗證集，不能作為四城市或獨立 test 結果。
- 權重：`output/no_gt_6x6_20260927_a/temporal_visual2061/raw10_sigma4/train/best_temporal_interim.pt` 及 `output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt`。固定 6×6 基準報告：`output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00/report.json`。
- `checkpoint1` 所存的 4.402 m 使用最終 **18×18**，因此不屬於現在的 6×6 目標。

## 已驗證的瓶頸

同一組權重在 `output/no_gt_6x6_20261008_acq_diagnostic/` 重播，僅額外輸出評估欄位，預測未變。63 幀中，候選庫涵蓋 GT 的比率為 98.4%，但選中視窗涵蓋 GT 的比率僅 88.9%。最佳可選視覺錨點的平均 GT 誤差是 3.835 m，實際選中視覺錨點是 6.195 m。這個 oracle 式「最佳可選」數字**只用來定位瓶頸**，不是可部署推論成績。

主要錯誤來自前段的參考點／搜尋視窗選擇：第 29 幀最終誤差 32.45 m，但候選庫內有距 GT 1.58 m 的錨點；相似度分數反而強烈支持錯位的錨點。第 10 幀最終誤差約 27 m，庫中最佳錨點仍距 GT 13 m；第 62 幀庫中最佳錨點距 GT 16.64 m。這表示只有改 Mean Shift bandwidth 很難達 4 m，必須改善視覺辨識與候選選取，並同時處理候選庫漏掉目標的情況。

## 本輪已完成實驗

| 試驗 | CityA val MLE (m) | 結論 |
| --- | ---: | --- |
| 固定 6×6 基準，bandwidth 25 m | 4.6677 | 原最佳 |
| 同權重，bandwidth 12 m | 4.6702 | 無改善 |
| 同權重，bandwidth 18 m | 4.6665 | 幅度極小 |
| 同權重，bandwidth 35 m | 4.6689 | 無改善 |
| 以較大步距的 train-only 路線續訓，acquisition loss weight 1 | 4.8796 | 退步 |
| 以較大步距的 train-only 路線續訓，acquisition loss weight 2 | 4.7915 | 退步 |
| 困難負例 32 張、實際訓練後的視覺權重 | 4.8481 | 退步，主要錯幀未改變 |
| 困難負例 64 張＋隨機 16 張 | 4.8777 | 退步 |
| 困難負例 64 張＋隨機 32 張 | 5.0335 | 退步 |
| 遠處錯點 margin loss 權重 0.1／0.3／0.6 | 5.0338／4.8992／5.0366 | 均退步 |
| 上述視覺權重＋acquisition 續訓權重 | 4.9622／4.9956／5.1062 | 均退步 |
| 多尺度 UAV TTA 1.0+0.95／0.90／0.80 | 4.9437／5.0117／5.7041 | 均退步 |
| 四個獨立旋轉 tracker 的座標平均（診斷） | 4.6294 | 未達標，增加四倍推論量 |
| 原特徵平均 tracker 與四 tracker 平均各佔 50%（診斷） | 4.4535 | 非原單模型架構，不能當作主模型結果 |
| 原 train-only 格網路線暖啟動，acquisition loss weight 4 | 4.6969 | capture 提高但 MLE 仍略差 |
| 從上一組再續訓，acquisition weight 8、sigma 2／4 | 4.8186／4.8318 | capture 提高到 95.2%，MLE 退步 |
| 從上一組再續訓，acquisition weight 16、sigma 4 | 4.8395 | 退步 |
| 最終 6×6 改用視覺／Kalman 中點置中 | 6.0589 | 退步 |
| 最終 6×6 改用 Kalman 置中 | 9.6284 | 顯著退步 |

較大步距資料的 audit 在 `data/official96_fixed_train_grid_wide/citya/audit.json`，6 條 episode、227 幀，未混入驗證或測試影像。結果在 `output/no_gt_6x6_20261008_wide_motion/summary.json`。MS bandwidth 結果在 `output/no_gt_6x6_20261008_ms_sweep/summary.json`；格網置中結果在 `output/no_gt_6x6_20261008_center_sweep/summary.json`；第一個視覺負例續訓結果在 `output/no_gt_6x6_20261008_visual_candidates/summary.json`。本輪保留舊基準，不以退步的權重取代。

## 本輪補充實驗與後續方向

已對同一視覺模型做 train-only 困難負例續訓，增加每張 UAV 反向傳播的衛星相似候選數，並直接評估 `candidate_epoch_last.pt`，避免訓練器保留舊權重時誤判新權重。GPU 0 的 hard32 已完成且未改善；GPU 5、6 的 hard64 變體已完成，分別得到 4.8777 m 和 5.0335 m，均未改善。GPU 0 另以原本 train-only 網格路線、現有 temporal 權重暖啟動，強化 acquisition loss（權重 4），結果在 `output/no_gt_6x6_20261008_acq_refine/`；正式四轉角 TTA MLE 為 4.6969 m，選中視窗涵蓋 GT 的比例由 88.9% 提高到 93.7%，但第 10 幀雖涵蓋 GT，局部視覺錨點仍錯 26.6 m，因此 MLE 沒有改善。記錄在 `run_no_gt6_visual_candidate_audit.py`、`run_no_gt6_visual_candidate_audit_retry16.py` 與 `output/no_gt_6x6_20261008_visual_candidates*/`。

三組 hard-negative 視覺模型都退步；僅增加每批參與反向傳播的相似衛星圖數量並不能解決目前的跨視角混淆。下一步改用訓練 GT 計算的 hard-negative 排序 loss：令最近的正確衛星格點分數高於距 GT 20 m 外最高分的錯誤格點。此 loss 不用 GT 挑選衛星影像作為推論輸入，最後 6×6 不變。三組 loss 權重 0.1／0.3／0.6 已於 GPU 0／5／6 完成，結果均退步；交叉搭配選點續訓權重仍退步。程式和結果在 `run_no_gt6_rank_loss.py`、`output/no_gt_6x6_20261008_rank_loss/` 及 `output/no_gt_6x6_20261008_rank_acq_combo/`。多尺度 TTA 的結果在 `output/no_gt_6x6_20261008_multiscale_tta/`。更強 acquisition 續訓（權重 8／16、sigma 2／4）也已完成，最高選中視窗涵蓋率雖達 95.2%，MLE 仍退步。結果目錄為 `output/no_gt_6x6_20261008_acq_refine_stage2/`。四個旋轉各自跑完整 tracker 後的座標融合結果在 `output/no_gt_6x6_20261008_rotation_ensemble/`；第 10、29 幀的四個分支都約錯 26–35 m，無法靠簡單平均解決。最後的配對實驗正在以三個新的 train-only GT-loss 視覺權重各自暖啟動重新訓練 temporal 部分，避免視覺特徵更新與舊 GRU 權重不匹配；程式 `run_no_gt6_pair_visual_temporal.py`，結果目錄 `output/no_gt_6x6_20261008_paired_visual_temporal/`。

**尚未達到 4.0 m。** 即使單條開發路線達標，仍須固定配置後檢查獨立保留測試路線，避免多次驗證調參造成過度樂觀。
