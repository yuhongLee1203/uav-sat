# Final Weighted-Centroid Ablation Tables

Main method: **Front SoftMS(18) -> 3-frame GRU -> fixed-R external Kalman -> one final MeanShift**.

Constant Velocity is fixed for every experiment; there is no separate motion-model table. The Kalman is not trained; its contribution is tested only by the progressive architecture ablation.

## Table 1. Progressive architecture ablation

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Front SoftMS(18) | 5.271 | 4.525 | 5.005 | 54.84% | 2.330/2.387% |
| + GRU | 5.529 | 4.977 | 5.332 | 51.10% | 2.769/2.705% |
| + Kalman | 4.495 | 3.972 | 4.309 | 63.61% | 0.044/0.000% |
| + final MS | 2.071 | 1.737 | 1.952 | 97.26% | 0.000/0.000% |

## Table 2. Temporal input-frame ablation

Each row is trained separately on Route A with the same original v39 training settings.

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 2.024 | 1.699 | 1.909 | 3.650 |
| 2 | current + previous; first difference | 2.075 | 1.733 | 1.953 | 3.721 |
| 3 | current + previous two; first + second difference | 2.071 | 1.737 | 1.952 | 3.713 |

## Table 3. Final MeanShift window sensitivity and PURE decoder latency

Latency starts only after the final candidate centers and regularized logits are ready. It measures **one soft_mean_shift call through metric XY output only**; candidate search/indexing, SAT projection, similarity scoring and prior-logit construction are excluded.

| Window | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.087 | 3.321 ms | 301.1 |
| 5x5 | 25 | 1.952 | 4.350 ms | 229.9 |
| 6x6 (main) | 36 | 1.952 | 8.209 ms | 121.8 |
| 7x7 | 49 | 1.951 | 7.692 ms | 130.0 |
| 8x8 | 64 | 1.951 | 11.538 ms | 86.7 |

## Table 4. Online end-to-end runtime

Prepared UAV tensor -> backbone -> Front SoftMS(18) -> 3-frame GRU -> fixed-R Kalman -> one final MS -> XY.

- Route B: 30.391 ms / 32.9 FPS
- Route C: 29.900 ms / 33.4 FPS
- Weighted B+C: **30.216 ms / 33.1 FPS**
