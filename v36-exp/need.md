<!-- V36_EXP_RESULTS_BEGIN -->

# V36 實驗自動彙整

尚未完成的工作會顯示 `PENDING`。

## 表 1：SoftMS vs Weighted Centroid

| 定位方式 | 聚合座標數 | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 | LSR@15 | 純座標聚合時間 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted Centroid | 18 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | 0.023903707 |
| SoftMS 收斂 modes 加權 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

聚合座標數是逐幀平均。時間只計算候選座標加權求和，不含圖片、backbone、matching、MeanShift、GRU 或 Kalman；不計 FPS。

## 表 2：V36 主要架構消融

| 方法 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SoftMS only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| SoftMS + 3-frame GRU | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| SoftMS + GRU + 慣性多項式 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 完整 V36（含 learned-variance Kalman） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 表 3：Forward 3×6 vs 6×6

| 搜尋方式 | 候選數 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | 端到端時間 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 完整 6×6 | 36 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Forward 3×6（causal-origin 修正版） | 18 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

3×6 的 origin backshift 固定為一個 gallery cell（4.75 m）；它由 Route-A/grid geometry 決定，沒有用 Route B/C 挑參數。

## 表 4：為什麼要三幀 GRU

| 輸入影像數量 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 幀 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 2 幀 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 3 幀（V36） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 表 5：慣性多項式實驗

| 運動預測方式 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | Progress MAE | Speed MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kalman CV（不使用 learned polynomial） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 只使用 GRU 速度 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GRU 速度 + 加速度二階多項式（V36） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

MAE = Mean Absolute Error（平均絕對誤差）；Progress MAE 是沿路徑進度誤差，Speed MAE 是每幀速度誤差。

舊版「不使用運動預測」曾把每幀位移強制設為 0，但又保留 Kalman 每幀最多修正 3 m 的限制，因而人為累積出數百公尺落後；該數據無效。修正版是不使用 learned polynomial，但保留外部 Kalman 自身的 constant-velocity prediction。

## 表 6：Kalman / Measurement Variance

| 最後輸出方式 | MLE | P90 | LSR@3 | LSR@5 | LSR@10 | LSR@15 |
|---|---:|---:|---:|---:|---:|---:|
| 不使用 Kalman | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 完整 V36 learned-variance Kalman | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 表 7：最終 Route B / Route C

| 路徑 | MLE | Median | P90 | P95 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | LSR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Route B | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Route C | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 平均（逐幀合併） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 表 8：其他論文原生協定比較

| 方法 | 原生定位協定 | MLE | Median | P90 | P95 | LSR@3 | LSR@5 | LSR@10 | LSR@15 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseUAV | Global retrieval | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Sample4Geo | Global retrieval | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Game4Loc | Global retrieval | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| InfoGeo | Global retrieval | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Bearing-UAV | Neighbor-map position/heading regression | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| V36（Ours） | GT+jitter Forward-3×6 local tracking | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

舊版約 12 m 的 local-18 adapter 結果已判定無效，不再列入正式比較；表 8 必須由各官方模型的原生 retrieval/regression 流程重新產生。
<!-- V36_EXP_RESULTS_END -->
