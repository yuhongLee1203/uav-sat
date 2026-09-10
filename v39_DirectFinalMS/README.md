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

## 預設 MS2 設定

- Kalman prior sigma: 4.0 m
- Reference prior sigma: 4.0 m
- Kalman prior weight: 1.5
- Reference prior weight: 2.5
- MeanShift bandwidth: 5.0 m
- MS2 reference perturbation: 0.0 m（只供 robustness experiment 使用；主方法為 0）

## 已有主結果

| Method | Route B MLE (m) | Route C MLE (m) | B+C weighted MLE (m) |
|---|---:|---:|---:|
| Original MobileNetV3 Kalman-final baseline | 4.7521 | 4.0711 | 4.5097 |
| v39 Direct Kalman -> reference-guided MS2 -> Final | **2.2148** | **1.7951** | **2.0654** |

相較原始 MobileNetV3 Kalman-final baseline，v39 的 B+C weighted MLE 約下降 54.2%。

Route B：P90 4.1256 m，LSR@5 96.97%，LSR@10/15/20 100%，JumpRate 0%。
Route C：P90 3.8423 m，LSR@5 95.15%，LSR@10/15/20 100%，JumpRate 0%。

# 論文完整實驗套件

完整實驗已整合進既有 `run.sh`，不新增另一套方法。它會先執行主方法並預熱共用 feature cache，之後使用 GPU 0、5、6 平行處理。

## A. Main comparison

- `baseline_kalman_no_ms2`：原始 Previous-State + Kalman final。直接讀既有 summary，不浪費 GPU 重跑。
- `main_full_j8`：完整 v39 = Visual + Kalman prior + Reference prior + SoftMS2。

用途：證明 `Kalman -> MS2 -> Final` 相較原始 `Kalman -> Final` 的改善。

## B. MS2 component ablation

固定 `JITTER_M=8`、bandwidth=5 m、MS2 reference perturbation=0：

- `abl_visual_only`：Visual only；KF prior=0、Reference prior=0。
- `abl_visual_kf`：Visual + Kalman prior；Reference prior=0。
- `abl_visual_reference`：Visual + Reference prior；KF prior=0。
- `main_full_j8`：Visual + Kalman prior + Reference prior。

用途：分離兩個 spatial prior 的貢獻，證明完整 MS2 為什麼有效。

## C. MS2 reference robustness

為避免把 MS1/local-search 的 jitter 和 MS2 本身混在一起，此實驗固定整個前段設定不變：

- `JITTER_M=8` 固定。
- MS1、GRU、Kalman 完全固定。
- 只在 MS2 的 predefined-reference prior 加入 deterministic zero-mean Gaussian perturbation。

測試 sigma：

- 0 m：`main_full_j8`
- 4 m：`robust_ms2ref_noise_4`
- 8 m：`robust_ms2ref_noise_8`
- 12 m：`robust_ms2ref_noise_12`
- 16 m：`robust_ms2ref_noise_16`

固定 seed 由 frame index 決定，因此每次重跑完全可重現。

用途：回答「MS2 的 reference spatial prior 不準時，定位性能如何退化」。

## D. Hyperparameter sensitivity

採 compact one-factor-at-a-time，而不是大型 grid search：

- Reference prior weight：1.5 / **2.5** / 3.5
- Kalman prior weight：0.75 / **1.5** / 2.25
- MeanShift bandwidth：4 / **5** / 6 m

粗體是主方法設定。

用途：證明結果不是只靠單一極端參數才能成立。

## E. 自動整理的論文指標

最後總表包含：

- Route B / C MLE
- B+C weighted MLE
- relative change vs main
- P90
- LSR@5
- LSR@15
- Jump Rate
- MS2 mean shift from Kalman
- Jitter / MS2 reference-noise sigma / prior weights / bandwidth

每個子實驗自己的 JSON 仍保留 MedLE、P95、P99、LSR@10/20 等完整指標。

# 一次跑完所有實驗

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
RUN_ALL_EXPERIMENTS=1 \
bash v39_DirectFinalMS/run.sh
```

執行策略：

1. GPU 0 先跑 `main_full_j8` 並預熱共用 feature cache。
2. 接著 GPU 0 / 5 / 6 同時工作。
3. 每張 GPU 內的 jobs 串行執行，避免同一 GPU 同時塞多個模型造成 contention。
4. 每個實驗使用獨立 runtime directory，不會互相覆寫程式。
5. 每次完整實驗建立新的 timestamp output folder，不覆蓋之前結果。
6. 全部完成後自動產生總表。

輸出資料夾：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

最重要的總表：

- `experiment_summary.csv`
- `experiment_summary.md`

每個子實驗另外保留自己的 `robust_tracker_summary.json`、per-frame CSV、eval log。

## 單獨執行主方法

```bash
cd /yh/study/uav-sat && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

單次結果：`v39_DirectFinalMS/output/robust_tracker_summary.json`

## 論文描述注意

這仍然屬於 controlled predefined-reference-point protocol。MS2 scoring 會使用目前 frame 的 predefined route reference point，因此論文應透明描述為 reference-guided local refinement，不應描述為完全未知目前位置的 autonomous localization。
