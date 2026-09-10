# v39_DirectFinalMS — Architecture-Focused Paper Experiments

目前主方法維持：

`MS1 -> GRU -> Kalman Filter -> MS2 -> Final Position`

這一版的實驗設計已重新整理成論文常見的 **module ablation + architecture/design sensitivity**。不再把 reference point 當成主要 ablation module，也不再跑 reference noise robustness、reference weight sweep、visual+reference 這類不直接對應 overview block 的實驗。

Reference-point protocol 在所有實驗中保持固定，只作為既定資料/定位 protocol，不放進主要 ablation table。

## 1. Main module ablation

這是最重要的主表，直接對應 overview 的 block：

| Experiment | MS1 | GRU | Kalman | MS2 |
|---|:---:|:---:|:---:|:---:|
| `abl_ms1_only` | ✓ |  |  |  |
| `abl_ms1_gru` | ✓ | ✓ |  |  |
| `abl_ms1_gru_kalman` | ✓ | ✓ | ✓ |  |
| `full_model` | ✓ | ✓ | ✓ | ✓ |

用途：直接回答每加入一個主要模組後，定位誤差與穩定性如何變化。

## 2. MS1 decoder design

固定其他架構不變，比較：

- `full_model`: Soft MeanShift
- `design_ms1_weighted`: Weighted Centroid

用途：證明 MS1 使用 MeanShift 而不是單純 weighted aggregation 的設計是否合理。

## 3. Candidate-search design

固定其他架構不變，比較：

- `full_model`: forward 3x6
- `design_search_full6x6`: full 6x6

用途：驗證 hard forward candidate restriction 是否能避免後方候選造成錯誤匹配。

## 4. Motion-model design

固定 GRU 與其他模組，比較：

- `design_motion_none`: no learned inertial polynomial
- `design_motion_velocity`: velocity model
- `full_model`: quadratic motion model

用途：驗證 second-order motion model 是否優於較簡單的運動假設。

## 5. Kalman uncertainty design

固定其他架構不變，比較：

- `design_kalman_fixed_var`: fixed measurement variance
- `full_model`: learned measurement variance

用途：驗證 learned measurement uncertainty 是否對 Kalman fusion 有幫助。

## 6. MS2 local-window size

固定完整架構，比較：

- `sens_ms2_grid4x4`: 4x4
- `full_model`: 6x6
- `sens_ms2_grid8x8`: 8x8

用途：分析 final refinement 的局部搜尋範圍大小。

## 7. MeanShift bandwidth sensitivity

固定完整架構與 6x6 MS2 window，比較：

- `sens_ms_bandwidth3`: 3 m
- `full_model`: 5 m
- `sens_ms_bandwidth7`: 7 m

用途：確認 MeanShift bandwidth 的選擇不是只靠單一極端值。

## 8. 不再執行的實驗

以下已從 paper experiment suite 移除：

- visual only / visual + reference / visual + Kalman prior 這種 MS2 score 拆解
- reference-point noise robustness
- reference prior weight sweep
- Kalman prior weight sweep
- reference perturbation 4/8/12/16 m

原因：這些項目不是 overview 中獨立的主要 architecture block，會讓主要 ablation 難以解釋，也偏離常見的 module ablation 呈現方式。

## 9. 一次跑完全部實驗

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
RUN_ALL_EXPERIMENTS=1 \
bash v39_DirectFinalMS/run.sh
```

執行配置：

- 先在 GPU 0 執行 `full_model`，建立/預熱共用 feature cache。
- 之後 GPU 0、5、6 平行執行剩餘實驗。
- 每張 GPU 內 jobs 串行，避免同 GPU contention。
- 每個實驗有獨立 output 與 runtime directory。
- 不會覆蓋之前的結果；每次建立新的 timestamp experiment folder。

輸出：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

最重要的總表：

- `experiment_summary.csv`
- `experiment_summary.md`

總表會直接列出：MS1、GRU、Kalman、MS2 是否存在，以及 decoder、search、motion、MS2 grid、bandwidth 和 B/C 的 MLE、P90、LSR、Jump Rate。

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

## 論文呈現建議

主要 ablation table 建議只放：

`MS1 | GRU | Kalman | MS2 | MLE | P90 | LSR@5 | Jump Rate`

再用 2–3 個小表或 sensitivity figure 分別呈現：

- 3x6 vs 6x6
- SoftMS vs Weighted Centroid
- none / velocity / quadratic motion
- MS2 window 4x4 / 6x6 / 8x8
- MeanShift bandwidth 3 / 5 / 7 m

這樣會比把 reference-point 設定塞進主 ablation table 更符合 architecture-focused 的論文實驗邏輯。
