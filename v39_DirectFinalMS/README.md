# v39_DirectFinalMS — SELECTED FINAL METHOD

目前選定這個版本作為「直接 Kalman -> MS2 -> Final」的主方法。不要使用 v42_VisualConsistencyMS 作為目前主實驗。

最終流程：

`MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final`

## 核心設計

這個版本把第二次 Kalman Update 完全移除。

MS2 的搜尋中心由單一 Kalman posterior 決定：先找到最接近 Kalman XY 的永久 SAT lattice anchor，再開完整 6x6 = 36 candidates。

目前 predefined route reference point 不拿去更新 Kalman，也不直接當輸出，只在 MS2 scoring 內作為 spatial prior。

因此 MS2 對 36 個候選同時考慮：

1. UAV-SAT visual likelihood。
2. Kalman spatial prior。
3. Predefined reference-point spatial prior。

最後執行 Soft MeanShift #2，`MS2 XY` 本身就是 Final Position；MS2 後面沒有 Kalman、clipping 或額外濾波。

## 為什麼採用這個版本

如果 MS2 只看影像相似度，農田等重複紋理場景容易把 MeanShift 吸到錯誤 local mode。

把 Kalman posterior 與 predefined route reference point 都只作為 MS2 內部的空間先驗，可以保留老師要求的 `Kalman -> MS2 -> Output`，同時避免增加 KF2 或把 MeanShift 變成形式上的最後一層。

## 預設 MS2 設定

- Kalman prior sigma: 4.0 m
- Reference prior sigma: 4.0 m
- Kalman prior weight: 1.5
- Reference prior weight: 2.5
- MeanShift bandwidth: 5.0 m

## 已有結果

| Method | Route B MLE (m) | Route C MLE (m) | B+C weighted MLE (m) |
|---|---:|---:|---:|
| Original MobileNetV3 Kalman-final baseline | 4.7521 | 4.0711 | 4.5097 |
| v39 Direct Kalman -> reference-guided MS2 -> Final | **2.2148** | **1.7951** | **2.0654** |

相較原始 MobileNetV3 Kalman-final baseline，v39 的 B+C weighted MLE 約下降 54.2%。

Route B 其他結果：P90 4.1256 m，LSR@5 96.97%，LSR@10/15/20 100%，JumpRate 0%。

Route C 其他結果：P90 3.8423 m，LSR@5 95.15%，LSR@10/15/20 100%，JumpRate 0%。

## 執行

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

結果位置：

`v39_DirectFinalMS/output/robust_tracker_summary.json`

## 論文描述注意

這仍然屬於 controlled predefined-reference-point protocol。MS2 scoring 會使用目前 frame 的 predefined route reference point，因此論文應透明描述為 reference-guided local refinement，不應描述為完全未知目前位置的 autonomous localization。
