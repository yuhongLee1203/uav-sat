# Final Weighted-Centroid Ablation Tables

Main method: **Front SoftMS(18) -> 3-frame GRU -> fixed-R external Kalman -> one final MeanShift**.

Constant Velocity is fixed for every experiment; there is no separate motion-model table. The Kalman is not trained; its contribution is tested only by the progressive architecture ablation.

## Table 1. Progressive architecture ablation

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Front SoftMS(18) | 5.271 | 4.525 | 5.005 | 54.84% | 2.330/2.387% |
| + GRU | 5.284 | 4.532 | 5.016 | 55.21% | 2.374/2.307% |
| + Kalman | 4.183 | 3.513 | 3.945 | 69.16% | 0.044/0.000% |
| + final MS | 1.942 | 1.621 | 1.828 | 97.96% | 0.000/0.000% |

## Table 2. Temporal input-frame ablation

Each row is trained separately on Route A with the same original v39 training settings.

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 1.911 | 1.591 | 1.797 | 3.477 |
| 2 | current + previous; first difference | 1.939 | 1.581 | 1.812 | 3.556 |
| 3 | current + previous two; first + second difference | 1.942 | 1.621 | 1.828 | 3.483 |

## Table 3. Final MeanShift window sensitivity and PURE decoder latency

Latency starts only after the final candidate centers and regularized logits are ready. It measures **one soft_mean_shift call through metric XY output only**; candidate search/indexing, SAT projection, similarity scoring and prior-logit construction are excluded.

| Window | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 1.942 | 3.331 ms | 300.3 |
| 5x5 | 25 | 1.827 | 4.582 ms | 218.2 |
| 6x6 (main) | 36 | 1.828 | 6.173 ms | 162.0 |
| 7x7 | 49 | 1.827 | 7.711 ms | 129.7 |
| 8x8 | 64 | 1.827 | 9.433 ms | 106.0 |

## Table 4. Online end-to-end runtime

Prepared UAV tensor -> backbone -> Front SoftMS(18) -> 3-frame GRU -> fixed-R Kalman -> one final MS -> XY.

- Route B: 29.940 ms / 33.4 FPS
- Route C: 30.026 ms / 33.3 FPS
- Weighted B+C: **29.970 ms / 33.4 FPS**
