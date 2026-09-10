# v39_DirectFinalMS — Final Paper Experiment Suite

論文架構固定為：

`GRU -> Kalman Filter -> MS -> Final Position`

GRU 前面的 UAV-SAT visual localizer / local visual observation generation 視為固定 front-end，不列入主要 architecture block。

## Selected default configuration

根據前一輪 pilot experiment，正式實驗預設改為：

- GRU motion: quadratic
- Kalman measurement variance: **fixed**
- MS window: **6x6**
- MeanShift bandwidth: **7 m**

注意：前一輪 B/C 結果已被看過，因此這次設定屬於 exploratory model refinement。若投稿時要宣稱完全 unbiased test performance，應在設定鎖定後使用未參與選擇的 held-out test data 做最終一次評估。

## Table 1 — Progressive architecture ablation

主消融不再使用 leave-one-out 的 `Kalman + MS without GRU` 當正文主表，而是按照實際架構順序逐步加入 proposed modules：

| Setting | GRU | Kalman | MS |
|---|:---:|:---:|:---:|
| Baseline visual front-end |  |  |  |
| + GRU | ✓ |  |  |
| + GRU + Kalman | ✓ | ✓ |  |
| Full: + GRU + Kalman + MS | ✓ | ✓ | ✓ |

這張表的目的就是回答：固定 visual front-end 後，GRU、Kalman、MS 依序加入是否改善 localization accuracy / stability。

對應實驗：

- `baseline_visual`
- `abl_gru_only`
- `abl_gru_kalman`
- `full_model`

## Table 2 — GRU motion design

固定 Kalman + MS，比較 GRU motion prediction：

- `design_motion_none`
- `design_motion_velocity`
- `full_model`: quadratic

報告 MLE 與 speed MAE。

## Table 3 — Kalman measurement design

正式預設採用 pilot 中較好的 fixed variance，因此比較：

- `design_kalman_none`: no Kalman
- `design_kalman_learned`: learned measurement variance
- `full_model`: fixed measurement variance (selected)

這張表不再把 learned variance 當成必須勝出的 contribution；實驗直接比較哪種 Kalman measurement design 更適合目前資料與 MS refinement。

## Table 4 — MS window accuracy-efficiency trade-off

比較：

- `sens_ms_grid4x4`: 16 candidates
- `full_model`: 6x6 = 36 candidates
- `sens_ms_grid8x8`: 64 candidates

除了 B/C MLE，也會實際量測：

- `MS_LatencyMean_ms`
- `MS_LatencyP90_ms`
- `MS_ThroughputFPS`

計時範圍只包含：

`Kalman position -> candidate construction/scoring -> MeanShift -> final MS coordinate`

GPU 每幀同步，前 30 幀 warm-up 不納入統計。

如果 8x8 只帶來極小精度增益但 latency 明顯上升，即可合理選擇 6x6 作為 accuracy-efficiency balance。

## Table 5 — MeanShift bandwidth accuracy-efficiency trade-off

比較：

- `sens_ms_bandwidth3`: 3 m
- `sens_ms_bandwidth5`: 5 m
- `full_model`: **7 m**

同時列 MLE 與 MS latency/FPS。

Bandwidth 在固定 candidate count 與固定 iteration 下通常不會像 window size 那樣大幅改變計算量。因此如果 7 m 維持近似 latency 且 accuracy 最佳，論文應直接說選擇 7 m 是因為它提供較低定位誤差而幾乎沒有額外 runtime cost，而不是硬說 5 m 是中間值。

## GPU parallelization

執行 `RUN_ALL_EXPERIMENTS=1` 時：

1. GPU 0 先執行 `full_model`，安全建立 shared feature cache。
2. cache 完成後 GPU 0 / 5 / 6 同時執行各自 queue。
3. 每張 GPU 內部串行，三張 GPU 彼此平行，避免同卡 contention。

分配：

- GPU 0：progressive architecture + learned variance
- GPU 5：GRU motion + no-Kalman + 4x4
- GPU 6：8x8 + bandwidth 3/5

## One-command execution

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
RUN_ALL_EXPERIMENTS=1 \
bash v39_DirectFinalMS/run.sh
```

輸出：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

會自動產生：

- `experiment_summary.csv`
- `paper_tables.md`
- `experiment_summary.md`

`paper_tables.md` 會直接整理 Table 1–5，包括 accuracy、jump rate、MS latency 與 MS FPS。

## Single full-model run

```bash
cd /yh/study/uav-sat && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

預設即為：quadratic GRU + fixed-variance Kalman + 6x6 MS + bandwidth 7 m。
