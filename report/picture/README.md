# FieldAnchor 3.759 m 重現結果與圖片

本頁集中整理這次推論的資料、數值、圖片和測速定義。下面各表也另存成獨立 `.md`，方便複製進論文。

## 實驗來源與適用範圍

- 原始推論：`/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368`。使用歷史 v39 權重重跑 inference，**沒有重新訓練**。
- 資料：四個城市、每城兩條測試路線，共 8 條路線與 1,482 筆觀測。這些路線由各張帶地理標記的 UAV 影像組成序列。
- 設定：**GT-assisted**。當前影格 GT 用於局部搜尋中心及 acquisition hypothesis 選擇，因此 3.759 m 不能寫成無 GT 推論成績，也不能直接當作開放式全圖 retrieval 的成績。
- 指標：MLE 是每筆定位誤差（公尺）的平均；LSR@3、LSR@15 是誤差不超過 3 m、15 m 的比例。Overall 先合併 1,482 筆誤差再計算。

## 八條路線定位結果

| City | Route | Samples | LSR@3 ↑ | LSR@15 ↑ | MLE (m) ↓ |
| --- | --- | --- | --- | --- | --- |
| City A | test_01 | 161 | 44.72% | 100.00% | 3.657 |
| City A | test_02 | 219 | 42.01% | 100.00% | 3.651 |
| City B | test_01 | 146 | 37.67% | 100.00% | 3.846 |
| City B | test_02 | 171 | 42.69% | 100.00% | 3.591 |
| City C | test_01 | 148 | 45.27% | 100.00% | 3.565 |
| City C | test_02 | 226 | 38.94% | 100.00% | 4.191 |
| City D | test_01 | 150 | 40.67% | 100.00% | 4.084 |
| City D | test_02 | 261 | 43.30% | 100.00% | 3.524 |
| Overall | — | 1482 | 41.90% | 100.00% | 3.759 |

Metrics use each frame’s recorded GT coordinate. Overall is pooled across 1,482 observations, not an unweighted average of routes.
Protocol: v13 synthetic 4 m routes; current-frame GT centers the local search and selects the acquisition hypothesis.

City C 的 test_02 MLE 最高；主結果 Overall 為 3.759 m、LSR@3 41.90%、LSR@15 100.00%。[獨立表格](route_wise_results.md)

## 衛星地圖上的軌跡

所有圖都直接使用各城市 **完整的 4096 × 4096 像素（1024 × 1024 m）衛星圖**，不再依路線縮小背景。八張圖的地圖物理範圍和顯示比例完全相同，且路徑全部留在圖內。

綠線是原始 waypoint 所定義的 GT 路徑，紅線依影格順序連接原始模型預測；沒有繪製每筆 GT 位置點。紅線可能與綠線重疊，請放大 PNG 檢視。四城總覽顯示各城市的 test_01；下列連結可開啟全部八條路線。

![四城市 test_01 衛星地圖軌跡](trajectories_four_cities_test01.png)

| City | test_01 | test_02 |
| --- | --- | --- |
| City A | [PNG](trajectory_citya_test_01.png) | [PNG](trajectory_citya_test_02.png) |
| City B | [PNG](trajectory_cityb_test_01.png) | [PNG](trajectory_cityb_test_02.png) |
| City C | [PNG](trajectory_cityc_test_01.png) | [PNG](trajectory_cityc_test_02.png) |
| City D | [PNG](trajectory_cityd_test_01.png) | [PNG](trajectory_cityd_test_02.png) |

[地圖裁切與路徑完整性紀錄](trajectory_crop_audit.json)；八張圖的裁切邊長一致，所有 GT waypoint 與預測點皆在圖內。

## 最後座標解碼比較

| Final decoder | LSR@3 ↑ | LSR@15 ↑ | MLE (m) ↓ | GPU-only aggregation (ms) ↓ | Single-call aggregation (ms) ↓ |
| --- | --- | --- | --- | --- | --- |
| Weighted Average (36 patches) | 42.51% | 100.00% | 3.749 | 0.003920 | 0.041984 |
| Mean Shift (1 mode) | 41.90% | 100.00% | 3.759 | 0.002640 | 0.042480 |

Both rows start from the same 36 final 6×6 candidate centers and regularized logits after identical front Mean Shift, GRU, and Kalman outputs. Weighted Average reduces all 36 softmax(logit/τ)-weighted centers. Mean Shift first iterates and merges its 36 seeds into active modes; all 1,482 frames have exactly one active mode, which its final aggregation reduces.
Probe-vs-original Full maximum XY difference: 0.000000000 m across 1482 frames. The logits include Kalman and current-frame GT reference priors.
Latency uses real probe inputs on an RTX 3090. GPU-only is CUDA-graph device time per final reduction; single-call is a synchronized CUDA-event measurement. Feature extraction, scoring, Mean Shift iterations and mode merging are excluded. The complete final Mean Shift stage takes about 7.24 ms/frame in the original replay.
Mean Shift minus Weighted Average paired MLE: +0.011 m (route bootstrap 95% interval [+0.006, +0.015] m; Mean Shift better on 0/8 routes). The observed result favors Weighted Average slightly; this experiment does not show a unique Mean Shift advantage.

兩列使用完全相同的 6 × 6＝36 個 candidate 位置與 scores，前段的 Mean Shift、GRU、Kalman 也相同。Weighted Average 直接聚合 36 個點；Mean Shift 先完成迭代與 mode 合併，再聚合最後 mode。這 1,482 筆最後都只剩 1 個 mode。**Weighted Average 的 MLE 略低**，所以這組結果不能寫成「Mean Shift 的最後聚合精度最高」。[獨立表格](final_decoder_comparison.md)

表內 GPU-only 約 0.003920 / 0.002640 ms 是最後一次座標加權聚合的 device 時間；single-call 約 0.041984 / 0.042480 ms 包含單次呼叫的 launch 間隔。兩者均不含 candidate scoring，也不含 Mean Shift 迭代與合併。完整 final Mean Shift stage 約 7.24 ms/frame。請勿把微基準數字稱為完整 decoder latency。[測速方法與 P90](final_aggregation_latency.md)

## 整體推論速度

| Item | Value |
| --- | --- |
| GPU | NVIDIA GPU 0/5/6 across parallel city evaluation |
| End-to-end mean latency | 39.72 ms/frame |
| Throughput equivalent | 25.18 FPS |
| Timing samples | 1242 |
| Final 6×6 Mean Shift latency | 7.24 ms/frame |

End-to-end timing starts at a prepared UAV tensor and excludes disk I/O, preprocessing, model loading, and satellite gallery construction. GPU models and concurrent load are in the replay environment; this is a measured replay, not a deployment benchmark.

平均 39.72 ms/frame 對應約 25.18 FPS。計時從已備妥的 UAV tensor 開始；不含磁碟 I/O、前處理、模型載入及 satellite gallery 建構。這是原始 replay 環境的量測，並非實機部署速度。[獨立表格](inference_runtime.md)

## 路線資料量

| City | train_01 | test_01 | test_02 | Route observations total |
| --- | --- | --- | --- | --- |
| City A | 198 | 161 | 219 | 578 |
| City B | 196 | 146 | 171 | 513 |
| City C | 196 | 148 | 226 | 570 |
| City D | 195 | 150 | 261 | 606 |
| Overall | 785 | 605 | 877 | 2267 |

These are counts of sampled observations on synthetic routes. Model training internally splits train_01 into train/validation segments; test_01 and test_02 are the eight evaluation routes.

這裡計的是合成路線上抽樣的觀測數。`train_01` 在訓練流程中又分為 train/validation；八條評估路線是各城市的 `test_01` 和 `test_02`。[獨立表格](dataset_route_counts.md)

## 重現檔案

- 表格和圖片生成：[`../scripts/generate_3759_additions.py`](../scripts/generate_3759_additions.py)。
- 最後聚合測速：[`../scripts/benchmark_3759_final_aggregation.py`](../scripts/benchmark_3759_final_aggregation.py)。
- 36 個候選點及分數由 `decoder_probe` 推論 CSV 記錄；主結果則取自原始 `full` 推論 CSV。probe 的 Mean Shift 座標與原始 Full 在逐影格檢查中一致。
