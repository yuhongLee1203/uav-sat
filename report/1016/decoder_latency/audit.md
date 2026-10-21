# 解碼計時複核（2026-10-17）

原先數字可重現，但「Mean Shift」欄僅代表合併後1個mode的最後加總，不能解讀為Mean Shift演算法較Argmax快。原回答的名稱與排序描述過於簡略，這裡澄清。

GPU0 RTX3090、同一筆真實路線中間幀、batch=1，三輪交錯順序量測。下表為各輪GPU純運算中位數，再取三輪中位數。

| 操作 | GPU運算 (ms) | GPU kernel數 |
| --- | ---: | ---: |
| Argmax: search 36 + gather XY | 0.00344800 | 2 |
| MS: final aggregation after merging (1 mode) | 0.00262400 | 2 |
| Weighted Average: aggregate 36 candidates | 0.00392000 | 2 |
| MS control: aggregate 36 converged seeds BEFORE merging | 0.00391200 | 2 |
| Argmax control: search 36 only (NO XY gather) | 0.00195200 | 1 |

Profiler確認Argmax實際有兩個GPU kernel：`argmax`找36個分數的最大索引，然後`index_select`取XY。合併後MS也有兩個kernel，但只是1個點的乘權重與加總。這個特定PyTorch實作與計時起點下，1點MS尾端加總確實較快。

只測Argmax、不取XY的對照約0.00195ms，比1點MS加總更快；但它只輸出索引，不是最終座標，所以不能拿來替換正式Argmax列。另一個對照：將36個已完成迭代、尚未合併的MS seeds加權加總約0.00391ms；它不包含合併，只用於確認輸入數目／計時切點的影響，不能當完整MS結果。

結論：原始量測沒有證據顯示計時器算錯；不過排除收斂「及合併」後的1點加總，範圍比泛稱「MS解碼」窄得多。若需要從剛迭代完成、尚未合併的點開始計時，應額外納入basin consolidation，再重測。完整MS或不同實作的排序，不能由此推定。

完整逐輪結果與kernel名稱：[audit_results.json](audit_results.json)。重跑：`python3 report/1016/audit_decoder_latency.py`。
