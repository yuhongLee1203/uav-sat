| Value | Default | MLE (m) | LSR@3 (%) | LSR@5 (%) | LSR@15 (%) | Decoder median latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 |  | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3415 |
| 4 |  | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3369 |
| 8 |  | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3275 |
| 12 | yes | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3396 |
| 20 |  | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3758 |

條件式 final-decoder sensitivity：只重算保存的同一組 6×6／36 candidates 與 regularized logits。前段模型、GRU、Kalman、候選位置及後續時序狀態固定；不是完整 tracker 重播，也未重新訓練。Logits 含 GT-conditioned reference/Kalman priors。一次只變一個參數；baseline τ=0.30、bandwidth=7 m、iterations=3、β=12。測試集最佳列僅作探索性分析，不作新模型選擇。Latency：GPU 0、batch=1、8條 route 各取中間一幀，重複3次共24次，CUDA同步，原 decoder 含 basin consolidation 的中位數；排除 candidate scoring。
