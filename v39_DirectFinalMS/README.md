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

## 已有主結果

| Method | Route B MLE (m) | Route C MLE (m) | B+C weighted MLE (m) |
|---|---:|---:|---:|
| Original MobileNetV3 Kalman-final baseline | 4.7521 | 4.0711 | 4.5097 |
| v39 Direct Kalman -> reference-guided MS2 -> Final | **2.2148** | **1.7951** | **2.0654** |

相較原始 MobileNetV3 Kalman-final baseline，v39 的 B+C weighted MLE 約下降 54.2%。

Route B 其他結果：P90 4.1256 m，LSR@5 96.97%，LSR@10/15/20 100%，JumpRate 0%。

Route C 其他結果：P90 3.8423 m，LSR@5 95.15%，LSR@10/15/20 100%，JumpRate 0%。

## 論文完整實驗設計

完整實驗套件已整合進既有的 `run.sh`，不需要另外建立實驗腳本。它會先重跑一次主方法並預熱共用 feature cache，之後使用 GPU 0、5、6 平行執行。

### A. Main comparison / final contribution

- `baseline_kalman_no_ms2`: 原始 Previous-State + Kalman final，作為沒有 MS2 的基線；直接讀取既有 baseline summary，不浪費 GPU 重跑。
- `main_full_j8`: 完整 v39，Visual + Kalman prior + Reference prior + SoftMS2。

這一組回答：最後加入 MS2 是否真的優於原本的 Kalman final。

### B. MS2 component ablation

固定 jitter=8 m、bandwidth=5 m：

- `abl_visual_only`: Visual only，KF prior=0，Reference prior=0。
- `abl_visual_kf`: Visual + Kalman prior，Reference prior=0。
- `abl_visual_reference`: Visual + Reference prior，KF prior=0。
- `main_full_j8`: Visual + Kalman prior + Reference prior。

這一組是最重要的消融，可以直接證明兩種 spatial prior 各自的作用，以及完整 MS2 為什麼有效。

### C. Reference-point robustness

完整 v39，其餘參數固定，只改 reference-point perturbation：

- 0 m
- 4 m
- 8 m（main）
- 12 m
- 16 m

這一組回答：方法是否只在很準的參考點下有效，以及參考點誤差增加後定位性能如何退化。

### D. Hyperparameter sensitivity

只做 compact one-factor-at-a-time，不進行沒有必要的大型 grid search：

- Reference prior weight: 1.5 / **2.5** / 3.5
- Kalman prior weight: 0.75 / **1.5** / 2.25
- MeanShift bandwidth: 4 / **5** / 6 m

粗體是主方法設定。這一組回答主結果是否依賴單一極端超參數。

### E. 自動輸出指標

總表會整理：

- Route B / C MLE
- B+C weighted MLE
- P90
- LSR@5
- LSR@15
- Jump Rate
- MS2 mean shift from Kalman
- 各實驗相對 main 的 MLE 變化百分比

完整 per-route summary 仍保留 MedLE、P95、P99、LSR@10/20 等原始欄位。

## 一次跑完所有論文實驗

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
RUN_ALL_EXPERIMENTS=1 \
bash v39_DirectFinalMS/run.sh
```

執行順序：

1. GPU 0 先執行 `main_full_j8`，並預熱共用 feature cache。
2. 預熱完成後，GPU 0 / 5 / 6 各自執行獨立 runtime directory，因此不會互相覆寫程式。
3. 三張 GPU 各自串行處理自己的 queue、GPU 間平行處理，以避免同一張 GPU 同時塞多個模型造成反而變慢。
4. 全部完成後自動產生總表。

每次完整實驗會建立新的 timestamp 資料夾，例如：

`v39_DirectFinalMS/experiments_YYYYMMDD_HHMMSS/`

最後最重要的兩個檔案：

- `experiment_summary.csv`
- `experiment_summary.md`

每個子實驗也各自保留：

- `robust_tracker_summary.json`
- per-frame CSV
- eval log

## 單獨執行主方法

原本單次執行方式仍然保留：

```bash
cd /yh/study/uav-sat && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v39_DirectFinalMS/run.sh
```

單次結果位置：

`v39_DirectFinalMS/output/robust_tracker_summary.json`

## 論文描述注意

這仍然屬於 controlled predefined-reference-point protocol。MS2 scoring 會使用目前 frame 的 predefined route reference point，因此論文應透明描述為 reference-guided local refinement，不應描述為完全未知目前位置的 autonomous localization。
