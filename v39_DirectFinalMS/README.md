# v39_DirectFinalMS — GRU / Kalman / MS Paper Experiments

論文與架構圖統一：

`GRU -> Kalman Filter -> MS -> Final Position`

GRU 前方的 UAV-SAT 視覺量測流程視為固定 front-end，不列入主要 architecture block。

## 1. 先釐清：Table 2 不是 1/2/3 張圖片實驗

目前所有正式 GRU 實驗都固定使用 **3 張連續 UAV 影像**。

程式中的 `velocity` 與 `quadratic` 並不是影像張數：

- `velocity` = **Constant Velocity**：GRU 仍然使用 3 幀，但 Kalman motion prediction 只使用 GRU 預測的速度，不加入 acceleration。
- `quadratic` = **Velocity + Acceleration**：GRU 同樣使用 3 幀，motion prediction 使用速度與 acceleration。
- `none` = **No learned motion**：不使用 GRU 所提供的 learned motion step，Kalman 只保留自己的 previous velocity。

因為先前實驗中 Constant Velocity 與 Velocity + Acceleration 表現非常接近，而且 Constant Velocity 略佳、也更容易說明，所以目前正式預設改為：

`3-frame GRU + Constant Velocity`

不要在論文中把 Constant Velocity 寫成「兩張圖片」。如果真的要比較 1/2/3 張圖片，應該分別重新訓練對應的 temporal model，不能直接拿目前以 3-frame feature 訓練的 checkpoint 做公平比較。

## 2. Table 1 — Progressive architecture ablation

正文只從論文架構的第一個主要 block 開始：

| Setting | GRU | Kalman | MS |
|---|:---:|:---:|:---:|
| GRU | ✓ |  |  |
| + Kalman | ✓ | ✓ |  |
| + MS | ✓ | ✓ | ✓ |

固定 visual front-end 不列入 Table 1。

## 3. Table 2 — Motion prediction model

所有 row 都固定使用 3-frame GRU：

- No learned motion
- Constant Velocity — current selected setting
- Velocity + Acceleration

這張表比較的是 **motion equation**，不是影像數量。

## 4. Table 3 — Kalman measurement design

比較：

- No Kalman
- Learned measurement variance
- Fixed measurement variance — current selected setting

## 5. Table 4 — MS local-window accuracy / efficiency

現在完整測試：

- 4x4 = 16 candidates
- 5x5 = 25 candidates
- 6x6 = 36 candidates
- 7x7 = 49 candidates
- 8x8 = 64 candidates

全部在 **同一張 GPU 5** 依序執行，避免先前不同 GPU 負載造成 latency 不公平。

每個設定同時記錄：

- Route B / C MLE
- weighted B+C MLE
- MS latency
- MS FPS

自動 selection rule：

> 在 B+C MLE 距離最佳結果 **0.5% 以內** 的 window 中，選擇 MS latency 最低者。

因此如果 6x6 已經與 7x7/8x8 幾乎一樣準，但運算明顯較快，就可以正式使用「accuracy-efficiency balance」解釋為什麼選 6x6。

## 6. Table 5 — MeanShift bandwidth sensitivity

Bandwidth 不直接改變 MeanShift 的 tensor size 或 iteration 數，因此不應使用 latency 作為選擇 bandwidth 的主要理由。

目前 SAT lattice stride = 32 px；資料解析度為 0.14 m/px，因此相鄰 SAT candidate center 約相距：

`32 x 0.14 = 4.48 m`

Bandwidth 控制 Gaussian kernel 的空間平滑尺度：

- bandwidth 遠小於 4.48 m：各候選較接近獨立 mode，精修較接近離散候選。
- bandwidth 約一到兩個 candidate spacing：會融合附近一致的 mode。
- bandwidth 很大：越來越接近對整個 local window 做廣域平滑，可能造成 over-smoothing。

因此不再只測 3 / 5 / 7 m，而是完整測試：

`1, 2, 3, ..., 14 m`

全部在 **同一張 GPU 6** 依序執行。

1 m 明顯小於一個 candidate spacing；14 m 已接近 6x6 window 中心到外圍的空間尺度，因此這個範圍足以看出 bandwidth 從窄 kernel 到強 smoothing 的完整趨勢。

Table 5 主要呈現：

- B MLE
- C MLE
- B+C MLE
- B+C P90
- B+C LSR@5

不再把 bandwidth latency 當主要比較項目。

## 7. GPU 配置

為確保同類型實驗在相同 GPU 上比較：

- **GPU 0**：architecture ablation + motion model + Kalman design
- **GPU 5**：MS window 4x4 / 5x5 / 6x6 / 7x7 / 8x8
- **GPU 6**：MeanShift bandwidth 1–14 m

三張 GPU 同時工作，但每一類 sweep 都固定在同一張 GPU 內串行執行。

## 8. Current selected defaults

目前根據前一輪 pilot：

- GRU input: 3 frames
- Motion: Constant Velocity
- Kalman: Fixed measurement variance
- MS window: 6x6
- MeanShift bandwidth: 7 m

本次完整 4–8 window sweep 與 1–14 m bandwidth sweep 是用來確認這些選擇，而不是假設它們一定是最佳值。

## 9. 一次跑完全部實驗

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

其中：

- `experiment_summary.csv`：所有 raw experiment metrics
- `paper_tables.md`：Table 1–5
- `experiment_summary.md`：同 paper tables
- `selection_summary.json`：自動整理 window balance point 與 bandwidth 最佳測試值

## 10. 單獨跑目前完整方法

```bash
cd /yh/study/uav-sat && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

目前預設單次架構：

`3-frame GRU (Constant Velocity) -> Kalman (Fixed Variance) -> MS (6x6, BW 7 m)`
