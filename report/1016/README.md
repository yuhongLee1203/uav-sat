# 2026-10-16：3.759 m 版本指標核對與超參數分析

使用原始 v39 3.759 m GT-assisted 版本，City A/B/C/D、8條合成路線、1,482 observations。GT 在該版本推論中提供 local reference 與 acquisition hypothesis；不是目前的無 GT 版本。所有新 GPU 計算只使用 GPU 0，最終 MS 始終為6×6。

| Method／協定 | 四 tile 座標衍生 R@1 ↑ | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE ↓ | SR@20 | SPL | NE |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| FieldAnchor Full，GT-assisted route replay | 93.66% | 41.90% | 73.75% | 100.00% | 3.759 m | N/A | N/A | N/A |

**這裡 R@1 的意思：**把 final XY 轉成已知 RSB 的四鄰接 tile 判定，與 Bearing-UAV 程式的 sign rule 對齊；不是 InfoGeo 的配對影像 Top-1 retrieval。GT-assisted 協定與評估子集也不同，因此這張表不應直接混入它們的 published benchmark comparison 而省略說明。

**Navigation 的意思：**SR@20 看 UAV 是否真的到達目標20m內；SPL 同時看是否成功及實際飛行路徑是否有效率；NE 看 episode 結束時真實位置距目標多遠。固定影像 replay 沒有由模型動作造成的新觀測、實際飛行軌跡與終止決策，所以本次沒有硬填數字。架構可作定位模組，但需要接上 closed-loop 環境／controller 與有效 heading 來源才可評估 navigation。

## 可以直接閱讀的表格

| 檔案 | 內容 |
| --- | --- |
| [01_main_localization.md](01_main_localization.md) | Full，8條路線與Overall，含座標衍生四tile R@1 |
| [02_ablation_localization.md](02_ablation_localization.md) | Component、temporal、window variants 的原始逐幀重算指標 |
| [03_bandwidth.md](03_bandwidth.md) | 固定候選的 final MS bandwidth sensitivity |
| [04_temperature.md](04_temperature.md) | 固定 regularized logits 的 decoder temperature sensitivity |
| [05_iterations.md](05_iterations.md) | final MS 迭代次數與實測 decoder latency |
| [06_mode_beta.md](06_mode_beta.md) | mode-density β sensitivity，結果基本持平 |
| [07_metric_applicability.md](07_metric_applicability.md) | 為什麼兩篇都有 R@1、你可以算哪一種、SR/SPL/NE公式與目前限制 |
| [08_paper_hyperparameters.md](08_paper_hyperparameters.md) | InfoGeo實際掃哪些參數，與這次FieldAnchor實驗的差異 |
| [09_full_tracker_bandwidth.md](09_full_tracker_bandwidth.md) | 已完成四城市／八條route的完整 tracker bandwidth 重播 |

## 圖片

- [Bandwidth](03_bandwidth.png)：太窄的空間 kernel 會降低精度，7 m後趨於飽和。
- [Temperature](04_temperature.png)：此條件式分析中 τ=0.5 的 MLE 約3.740 m，原τ=0.3約3.759 m，差異很小。
- [Iterations：accuracy／latency](05_iterations_latency.png)：3次後準確度改善很小，但時間增加。
- [Iterations：MLE／LSR@5](05_iterations.png)、[β](06_mode_beta.png)。

以上03–06為 **條件式 final-decoder sensitivity**：使用相同36個候選與原regularized logits，只重算final decoder；不將新結果回灌後續tracker狀態。23列設定代表20個不同組合。不能把最小誤差列當成已完成完整tracker評估的新主模型；09另做完整tracker重播以核對狀態回饋效應。所有test上的掃參皆是探索性分析，主模型更改應先經train/validation選定，再測未用於選參的test。

InfoGeo 不是 Mean Shift 方法；它掃 slot數K、融合係數α和頻譜維度r。FieldAnchor本次選的是自身真正存在的解碼參數，沒有修改架構或搜尋大小。

## 原始數據與稽核

- [source_audit.json](source_audit.json)：原始CSV與保存候選／分數逐幀完全一致，Full位置差異0。
- [sensitivity_results.json](sensitivity_results.json)：完整數字、GPU型號、baseline重現差異。
- `sensitivity_predictions.npz`：20個組合對1,482 observations的解碼位置。
- `localization_frame_audit.json`：每幀四tile判定與原始位置誤差。
- `full_tracker_bandwidth/`：完整重播的各城市logs、原始逐幀CSV、manifest；僅讀取原始checkpoint與prepared資料。
- 原始保存run：`/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368`。
- no_kalman保存replay的LSR@5是73.82%，與先前稿件73.68%不同；此處使用原始CSV，沒有為符合稿件而改值。

重算表格與條件式敏感度：`python3 report/1016/run_analysis.py`。完整tracker bandwidth重播：`python3 report/1016/run_full_bandwidth.py`。兩者均限GPU0。核對候選來源：`python3 report/1016/audit_sources.py`。

## 論文來源

- [Bearing-UAV：Beyond Matching to Tiles，Appendix A.2／Section 4.3.6](https://arxiv.org/html/2603.22153v1)
- [InfoGeo，Section 5.1／5.4](https://arxiv.org/pdf/2605.07099)
- [SPL 原始定義，Section 4](https://arxiv.org/pdf/1807.06757)

## 完整 tracker bandwidth 實測結果

| Bandwidth (m) | MLE (m) | LSR@3 (%) | LSR@5 (%) |
| --- | ---: | ---: | ---: |
| 4 | 3.8045 | 40.89 | 73.01 |
| 7 | 3.7591 | 41.90 | 73.75 |
| 14 | 3.7506 | 42.44 | 74.16 |

[完整重播曲線](09_full_tracker_bandwidth.png)。7 m baseline重現原3.759m；14 m相對改善僅約公分等級，不據此宣稱新設定顯著更好。這三組均重新執行完整tracker，與03–06固定候選分析範圍不同。

## 三種解碼耗時與固定權重

- [Argmax／Mean Shift／Weighted Average 實測表](decoder_latency/comparison.md)：GPU0、最終座標運算，排除MS收斂及合併，另列單次呼叫耗時。
- [原始權重固定參考](WEIGHTS_REFERENCE.md)、[權重SHA-256清單](decoder_latency/weights_manifest.json)。
- [完整量測結果](decoder_latency/results.json)、[原始逐次耗時](decoder_latency/timing_raw.csv)。

本次GPU純運算實測為Mean Shift < Argmax < Weighted Average；MLE為Weighted Average < Mean Shift < Argmax。MS皆只剩1個active mode，計時範圍不包含收斂，不能推論完整MS decoder較快。

## 2026-10-17：公平計時重跑（優先引用）

[共同36點、權重備妥→XY的比較](decoder_latency_fair/comparison.md)，[原始結果](decoder_latency_fair/results.json)。MS直接加總36個收斂seed，沒有先合併成1點再計時；全部1,482筆輸出與原版MS完全一致。GPU純運算：Argmax約0.003472ms、MS約0.003936ms、WA約0.003928ms，後兩者應視為相近。補充表另列MS密度／權重計算及原碼合併的成本。舊decoder_latency/是合併後1點加總，仍保留稽核用，不能視為同一計時範圍。

## 2026-10-17：原碼剩餘解碼流程重跑

[原實作三方法比較](decoder_latency_native/comparison.md)：僅排除MS的設定迭代，保留density／seed權重／原Python群合併到XY；WA包含softmax。三方法皆同步wall clock。GPU0各1,200次，Argmax median=0.074228ms、WA=0.093070ms、MS=3.025765ms。這個口徑是實際剩餘程式延遲，與前面prepared-weights CUDA graph加總不同。若描述「MS迭代後的原碼耗時」，應引用本表；前面各表保留其各自明確範圍。
