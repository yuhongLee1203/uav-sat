# v39_DirectFinalMS — GRU / Kalman / MS2 Paper Experiments

論文與架構圖統一只把下列三個模組視為主要 architecture：

`GRU -> Kalman Filter -> MS2 -> Final Position`

實際程式在 GRU 前仍然需要固定的 UAV-SAT 視覺候選與視覺量測生成流程，但這一段視為 **fixed visual front-end**，不列入主要 architecture block，也不放進 module ablation table。

因此後續論文實驗不再出現 MS1 欄位，也不再跑 MS1 decoder、reference robustness、reference weight 等實驗。

## 1. Main module ablation

主表只看 GRU、Kalman、MS2：

| Experiment | GRU | Kalman | MS2 |
|---|:---:|:---:|:---:|
| `abl_gru_only` | ✓ |  |  |
| `abl_gru_kalman` | ✓ | ✓ |  |
| `abl_gru_ms2` | ✓ |  | ✓ |
| `abl_kalman_ms2` |  | ✓ | ✓ |
| `full_model` | ✓ | ✓ | ✓ |

這樣可以同時做 progressive ablation 與 leave-one-module-out：

- `abl_gru_only`：最簡單 temporal estimator。
- `abl_gru_kalman`：測加入 Kalman 後的效果。
- `abl_gru_ms2`：拿掉 Kalman，確認 MS2 本身能否取代 filter。
- `abl_kalman_ms2`：拿掉 GRU，確認 temporal learned state 是否必要。
- `full_model`：完整 GRU + Kalman + MS2。

## 2. GRU motion-model design

固定完整架構，比較：

- `design_motion_none`: no learned inertial polynomial
- `design_motion_velocity`: velocity motion
- `full_model`: quadratic motion

用途：驗證 GRU 後面使用的 second-order motion prediction 是否優於較簡單設計。

## 3. Kalman uncertainty design

固定完整架構，比較：

- `design_kalman_fixed_var`: fixed measurement variance
- `full_model`: learned measurement variance

用途：驗證 learned measurement uncertainty 是否能改善 Kalman fusion。

## 4. MS2 local-window size

固定 GRU + Kalman + MS2，比較：

- `sens_ms2_grid4x4`: 4x4
- `full_model`: 6x6
- `sens_ms2_grid8x8`: 8x8

用途：分析 MS2 最後局部 refinement 的搜尋範圍大小。

## 5. MeanShift bandwidth sensitivity

固定完整架構與 6x6 MS2 window，比較：

- `sens_ms_bandwidth3`: 3 m
- `full_model`: 5 m
- `sens_ms_bandwidth7`: 7 m

用途：分析 MeanShift bandwidth 對最終定位結果的影響。

## 6. 固定、不列入 architecture ablation 的東西

以下內容在所有主要實驗中固定，不作為 architecture module：

- UAV-SAT visual localizer
- 前方 local candidate construction
- GRU 前的 visual observation generation
- predefined route reference-point protocol
- MS2 內部固定 scoring formulation

也就是論文 overview 與主消融只討論：

`GRU -> Kalman Filter -> MS2`

## 7. 已移除的實驗

不再執行：

- MS1 only / MS1 + GRU 等表格
- SoftMS vs Weighted Centroid 的 MS1 decoder 實驗
- 3x6 vs 6x6 的前端搜尋實驗
- visual + reference / visual + Kalman 等 MS2 score 拆解
- reference-point robustness
- reference prior weight sweep
- Kalman prior weight sweep

## 8. 一次跑完全部論文實驗

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
RUN_ALL_EXPERIMENTS=1 \
bash v39_DirectFinalMS/run.sh
```

GPU 配置：

- GPU 0：`full_model` + 主要 GRU/Kalman/MS2 ablation
- GPU 5：leave-one-out + GRU/Kalman design
- GPU 6：MS2 window / MeanShift bandwidth

每張 GPU 內部串行，三張 GPU 平行，並共用 backbone feature cache。

輸出：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

主要總表：

- `experiment_summary.csv`
- `experiment_summary.md`

總表只會以 GRU / Kalman / MS2 作為主要 module 欄位。

## 9. 論文主表建議

正文主消融表直接使用：

`GRU | Kalman | MS2 | MLE | P90 | LSR@5 | Jump Rate`

其中最核心比較為：

1. GRU
2. GRU + Kalman
3. GRU + Kalman + MS2

另外 `GRU + MS2` 與 `Kalman + MS2` 可作為 leave-one-module-out rows，用來證明完整三模組組合的必要性。

## 10. 單獨跑完整主方法

```bash
cd /yh/study/uav-sat && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

單次結果：

`v39_DirectFinalMS/output/robust_tracker_summary.json`
