# v39 → v40 無當前幀 GT 定位：checkpoint 1（2026-09-26）

## 此 checkpoint 的意義

- CityA `val_01`、63 幀，最好的**驗證集** MLE **4.402 m**、P90 6.206 m、LSR@15 96.825%。這是 `best_no_gt_20260926/report.json` 的結果。尚未達到 4.0 m，也不是獨立測試集或 Bearing-UAV Table 6 數值。
- 無當前幀 GT 輸入：`GT_front18_oracle=false`、`GT_final_MS_center_oracle=false`、`current_frame_GT_model_input=false`。GT 僅用於訓練損失與驗證計算。固定路線的航點於測試之前宣告；這個自組 route episode 使用原始資料離線建構，因此不宜稱為完全無 GT 的原始資料建構。
- `best_no_gt_20260926/visual_retrieval_A_only.pt` 與 `best_temporal_interim.pt` 是搭配使用的權重。`code_before_fusion/` 是得到 4.402 m 當時的程式快照；`MANIFEST.json` 含來源、大小及 SHA256。原始執行設定在 `run_settings_original.json`。

## 目前 4.402 m 的推論設定

每幀由路線航點、上一幀 GRU/Kalman 狀態產生 7 個局部搜尋假設；每個假設只算前方 18 個衛星 patch，視覺相似度與局部 Soft Mean Shift 形成觀測，GRU 更新時間狀態，Kalman 融合運動與視覺觀測，再以視覺錨點為中心於 **18×18** 格網做最後 Mean Shift。這個最後格網使用 bandwidth 25 m、score temperature 0.7；候選擷取的 raw visual weight 是 20，路線參考 prior weight 是 0。UAV 影像的 0、90、180、270 度 embedding 在推論時平均。路線航點是先於當前影像宣告的外部參考，不是逐幀 GT 座標。

## 從 v39 搬過來到目前的過程

1. 保留 v39 的 MobileNetV3-Small 視覺編碼、前方 18 個衛星 patch 的 Soft Mean Shift 觀測、三幀 GRU、路線座標 Kalman、最終 Mean Shift。為 v40 接上官方 96 metadata 的 CityA 固定路線，分出訓練、`val_01` 與保留測試路線。訓練可以用 GT 計算 loss；搜尋中心、GRU、Kalman、最終 MS 在推論時不讀當前幀 GT。
2. 將前方搜尋的幾何及可見影像候選接到 causal route state，解除從受控 GT 先驗沿用的不合理手動限制。初期嚴格無 GT 驗證約 **7.202 m**，報告已另存 `baseline_20260925_report.json`。
3. 查出 `next_step_se[t]` 的監督目標方向錯位；`train_multicity_routes.py` 改成使用 `GT[t+1]-GT[t]`。後續重訓該版本，但未超過當時最佳驗證配置。
4. 視覺模型只用訓練 GT 損失，從 CityA 訓練影像做局部衛星檢索；加入 UAV 四分之一轉角增強及推論時 0、90、180、270 度影像特徵平均。對 acquisition 原始視覺證據權重、最後 MS 溫度、bandwidth、候選格網及路線先驗做 `val_01` 掃描。單輪較溫和的視覺模型加上四轉角 TTA、最後 18×18 格網，曾得到 **4.618 m**。24×24 格網得到 4.615 m，計算量更高，改進極小。
5. 從 4.618 m 的視覺權重以 seed 2047、2059 各再訓練一輪；所有 oracle 關閉。seed 2047 得到 **4.598 m**，seed 2059 得到**4.402 m**。其他已完成試驗包含較長視覺訓練 4.750 m、修正後 temporal 與視覺聯合配置 4.661 m、額外 image hypothesis 的 temporal 配置 4.747 m。完整路徑與數字索引見 `CAUSAL_EXPERIMENT_INDEX.csv`。
6. GT 擾動檢查將驗證 GT 整體偏移 `[1234,-987]` m，預測與搜尋中心變化為 **0.0 m**；原檔另存 `gt_independence_audit.json`。此檢查針對核心路線推論；TTA 僅平均影像特徵。

## 目前模型到底怎麼處理三幀？

舊版先用三個 UAV embedding 得到 `clip_mean=(z[t-2]+z[t-1]+z[t])/3`、`diff1=z[t]-z[t-1]`、`diff2=diff1-(z[t-1]-z[t-2])`。程式有宣告 `clip_mean_projection`，但 GRU **並未使用它**；實際輸入是 `z[t]`、`diff1`、`diff2` 各走一個投影 MLP。其後再加衛星上下文、視覺位置與移動、前一狀態，共 7 個 128 維區塊，送入 GRU。候選位置評分器也使用這三種各自投影的影像資訊。此細節可在 `code_before_fusion/runtime/visual_model.py` 查證。

**18 patch** 指每一個局部搜尋視窗從 6×6 幾何中只算前方 3×6 的衛星 patch。**18×18** 是 Kalman 之後最後 Mean Shift 的候選格網，會算 324 個候選位置；兩個 18 的意義不同。18×18 與 bandwidth 25 m 是驗證時的設定，並非 GRU 的輸入大小。

## 新的單 MLP 融合實驗

`V40_TEMPORAL_FUSION_MLP=1` 時，把 `clip_mean`、`diff1`、`diff2` 先串接成 1536 維，再經 **一個** Linear→GELU→LayerNorm 輸出 384 維。這 384 維切成三個 128 維區塊供既有 GRU、候選評分器和時間殘差頭使用；GRU、Kalman、前方 18 patch、最終 MS 的拓樸不變。因舊版權重不含融合層，新的 temporal 模型從隨機初始化重新訓練；視覺權重沿用本 checkpoint 已完成 GT-loss-only 訓練的最佳版本。關閉該旗標即可重現舊版結構。程式給新結構獨立名稱，舊 temporal 權重無法誤載。

在 `/yh/study/uav-sat` 執行：

```bash
python3 v40/run_fused_temporal_citya.py
```

只使用 **一個 seed（預設 2033）**，但同時啟動兩個格網配置的訓練工作：GPU 5 用 18×18（324 候選），GPU 6 用 6×6（36 候選）。兩邊都使用同一個 seed、同樣訓練資料和排程；各自每個 epoch 驗證、挑選自己的權重，再以四轉角 TTA 在 `val_01` 比較。因此表格表示「各配置重新訓練後」的結果；若權重因各自的 epoch 選擇不同，差異不完全只來自格網。比較表在 `output/fused_temporal_citya_grid_parallel_20260926/comparison.json`；驗證較好的格網才進行固定 CityA `test_01`、`test_02` 一次保留測試。要只比較驗證集可加 `--skip-heldout-test`。

Terminal 會即時顯示 GPU、格網、每條路線與 frame 進度、loss、每 epoch 驗證誤差、總進度條、已用時間和動態 ETA。完整子程序訊息同時寫入各格網的 `train.log`。若工作由其他終端或代理程式啟動，可另開 terminal 執行 `python3 v40/show_fused_temporal_progress.py` 看兩張 GPU 的即時進度。每個工作預設 4 epochs ×（13 條訓練 route＋1 次驗證）＝56 個進度單位；early stopping 可能提前完成。實際時間依 GPU 及影像快取狀態而變。

18×18 曾在先前的 CityA 驗證掃描中改善結果，因此成為 4.402 m 配置；6×6 是原始最終 MS 大小。先前小格網的成績來自不同視覺權重和設定，**不能**直接代表這次融合模型的 6×6 表現；本次共用權重比較才回答這個問題。

## 限制

目前最好的 4.402 m 是單條路線反覆挑選後的開發成績，可能高估泛化能力。失誤幀多由影像候選選錯造成；單 MLP 融合是待驗證假說，不能預先宣稱會比 4.402 m 好。保留測試結果應與驗證結果分開呈現。
