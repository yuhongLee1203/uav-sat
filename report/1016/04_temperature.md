| Value | Default | MLE (m) | LSR@3 (%) | LSR@5 (%) | LSR@15 (%) | Decoder median latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 |  | 3.9374 | 39.5412 | 70.3104 | 100.0000 | 4.3434 |
| 0.2 |  | 3.8126 | 40.8232 | 72.4696 | 100.0000 | 4.3478 |
| 0.3 | yes | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3396 |
| 0.5 |  | 3.7400 | 42.3752 | 74.6964 | 100.0000 | 4.3337 |
| 0.7 |  | 3.7572 | 41.7679 | 74.4939 | 100.0000 | 4.3419 |
| 1.0 |  | 3.8090 | 40.2834 | 73.6167 | 100.0000 | 4.3414 |

條件式 final-decoder sensitivity：只重算保存的同一組 6×6／36 candidates 與 regularized logits。前段模型、GRU、Kalman、候選位置及後續時序狀態固定；不是完整 tracker 重播，也未重新訓練。Logits 含 GT-conditioned reference/Kalman priors。一次只變一個參數；baseline τ=0.30、bandwidth=7 m、iterations=3、β=12。測試集最佳列僅作探索性分析，不作新模型選擇。Latency：GPU 0、batch=1、8條 route 各取中間一幀，重複3次共24次，CUDA同步，原 decoder 含 basin consolidation 的中位數；排除 candidate scoring。
