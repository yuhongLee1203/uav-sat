# InfoGeo 的超參數與 FieldAnchor 適合做的分析

InfoGeo 的 sensitivity 是 slot 數 K、融合權重 α，另比較頻譜維度 r；並非 Mean Shift bandwidth。它的 Top-1 是配對影像 retrieval。這些模組不在 FieldAnchor，不能把其 K／α／r 改名後硬套。來源：[InfoGeo Section 5.4，Table 6、Figure 7](https://arxiv.org/pdf/2605.07099)。

| 本次參數 | 掃描值 | 原始值 | 研究問題 | 本次是否重新訓練 |
| --- | --- | --- | --- | --- |
| Final MS bandwidth（m） | 2、4、5、7、10、14、20 | 7 | 空間 kernel 太窄或太寬會如何影響解碼？ | 否，固定 final candidates 重算 decoder |
| Final score temperature τ | 0.1、0.2、0.3、0.5、0.7、1.0 | 0.3 | 尖銳／平坦的候選分布對定位影響？ | 否，原 regularized logits 固定，只改 decoder softmax |
| MS iterations | 1、2、3、5、8 | 3 | 收斂精度和計算成本的差別？ | 否 |
| MS mode-density β | 0、4、8、12、20 | 12 | 密度對種子權重的偏好有多大影響？ | 否 |

四個參數各自一維掃描，其餘維持原始值，最後搜尋大小固定 6×6。本次共有 23 列設定，去掉重複的 baseline 為 20 個不同 decoder 設定。

這些結果是 **conditional final-decoder sensitivity**：保存的 36 個候選與分數來自原 3.759 m、GT-assisted replay。候選建構與 score priors 不變，也不把新的 final 輸出回灌 tracker 跑下一幀。它能回答 decoder 對超參數的敏感度，不能直接當成完整 tracker 重新測試的 MLE，更不能宣稱更佳設定已通過獨立 test 選擇。

若要改變論文主設定，應先在 train/validation 選定參數，再對完整 tracker 與未用於選參的 test 執行一次。訓練超參數如 hidden dimension、learning rate、loss weight 則必須分別訓練 checkpoint，本次沒有將 inference sweep 冒充重新訓練實驗。
