# v39_DirectFinalMS — GRU / Kalman / MS Paper Experiments

論文與架構圖統一只把下列三個模組視為主要 architecture：

`GRU -> Kalman Filter -> MS -> Final Position`

實際程式在 GRU 前仍然需要固定的 UAV-SAT 視覺候選與視覺量測生成流程，但這一段視為 **fixed visual front-end**，不列入主要 architecture block，也不放進 module ablation table。

因此後續論文實驗不再出現 MS1，也不再使用 MS2 這個名稱。最終 MeanShift 模組一律直接稱為 **MS**。

## 1. Main module ablation

主表只看 GRU、Kalman、MS：

| Experiment | GRU | Kalman | MS |
|---|:---:|:---:|:---:|
| `abl_gru_only` | ✓ |  |  |
| `abl_gru_kalman` | ✓ | ✓ |  |
| `abl_gru_ms` | ✓ |  | ✓ |
| `abl_kalman_ms` |  | ✓ | ✓ |
| `full_model` | ✓ | ✓ | ✓ |

這樣可以同時做 progressive ablation 與 leave-one-module-out：

- `abl_gru_only`：只保留 GRU estimator。
- `abl_gru_kalman`：加入 Kalman，測 temporal prediction + filtering。
- `abl_gru_ms`：拿掉 Kalman，測 GRU + MS。
- `abl_kalman_ms`：拿掉 GRU，測 Kalman + MS。
- `full_model`：完整 GRU + Kalman + MS。

## 2. GRU motion-model design

固定完整架構，比較：

- `design_motion_none`: no learned inertial polynomial
- `design_motion_velocity`: velocity motion
- `full_model`: quadratic motion

用途：驗證 GRU 後面的 second-order motion prediction 是否優於較簡單設計。

## 3. Kalman uncertainty design

固定完整架構，比較：

- `design_kalman_fixed_var`: fixed measurement variance
- `full_model`: learned measurement variance

用途：驗證 learned measurement uncertainty 是否能改善 Kalman fusion。

## 4. MS local-window size

固定 GRU + Kalman + MS，比較：

- `sens_ms_grid4x4`: 4x4
- `full_model`: 6x6
- `sens_ms_grid8x8`: 8x8

用途：分析最後 MeanShift refinement 的局部搜尋範圍大小。

## 5. MeanShift bandwidth sensitivity

固定完整架構與 6x6 MS window，比較：

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
- MS 內部固定 scoring formulation

也就是論文 overview 與主消融只討論：

`GRU -> Kalman Filter -> MS`

## 7. 已移除的實驗

不再執行：

- MS1 only / MS1 + GRU 等表格
- MS1 decoder 實驗
- 前端 3x6 vs 6x6 搜尋實驗
- visual + reference / visual + Kalman 等 score 拆解
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

GPU 配置採最大化且避免共用 cache 競爭的方式：

- GPU 0 先跑一次 `full_model`，安全建立 shared feature cache。
- cache 建立後，GPU 0 / 5 / 6 三張卡同時進入各自的實驗 queue。
- GPU 0：主要 architecture ablation。
- GPU 5：leave-one-out + GRU/Kalman design。
- GPU 6：MS window / MeanShift bandwidth。
- 每張 GPU 內 jobs 串行，三張 GPU 彼此平行。

輸出：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

主要總表：

- `experiment_summary.csv`
- `experiment_summary.md`

總表只會以 `GRU | Kalman | MS` 作為主要 module 欄位。

## 9. 論文主表建議

正文主消融表直接使用：

`GRU | Kalman | MS | MLE | P90 | LSR@5 | Jump Rate`

其中最核心比較為：

1. GRU
2. GRU + Kalman
3. GRU + Kalman + MS

另外 `GRU + MS` 與 `Kalman + MS` 可作為 leave-one-module-out rows，用來證明完整三模組組合的必要性。

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
