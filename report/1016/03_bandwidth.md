| Value | Default | MLE (m) | LSR@3 (%) | LSR@5 (%) | LSR@15 (%) | Decoder median latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 |  | 4.4240 | 35.0202 | 64.5749 | 99.6626 | 4.4617 |
| 4 |  | 3.8045 | 40.8907 | 73.0094 | 100.0000 | 4.3463 |
| 5 |  | 3.7765 | 41.3630 | 73.3468 | 100.0000 | 4.3361 |
| 7 | yes | 3.7591 | 41.9028 | 73.7517 | 100.0000 | 4.3396 |
| 10 |  | 3.7529 | 42.2402 | 74.0216 | 100.0000 | 4.3443 |
| 14 |  | 3.7506 | 42.4426 | 74.1565 | 100.0000 | 4.3394 |
| 20 |  | 3.7495 | 42.4426 | 74.1565 | 100.0000 | 4.3329 |

條件式 final-decoder sensitivity：只重算保存的同一組 6×6／36 candidates 與 regularized logits。前段模型、GRU、Kalman、候選位置及後續時序狀態固定；不是完整 tracker 重播，也未重新訓練。Logits 含 GT-conditioned reference/Kalman priors。一次只變一個參數；baseline τ=0.30、bandwidth=7 m、iterations=3、β=12。測試集最佳列僅作探索性分析，不作新模型選擇。Latency：GPU 0、batch=1、8條 route 各取中間一幀，重複3次共24次，CUDA同步，原 decoder 含 basin consolidation 的中位數；排除 candidate scoring。
