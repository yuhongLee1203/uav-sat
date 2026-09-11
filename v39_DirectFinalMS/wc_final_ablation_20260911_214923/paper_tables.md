# Final Weighted-Centroid Ablation Tables

Main method: **Weighted Centroid -> 3-frame GRU -> fixed-R external Kalman -> one final MeanShift**.

Constant Velocity is fixed for every experiment; there is no separate motion-model table. The Kalman is not trained; its contribution is tested only by the progressive architecture ablation.

## Table 1. Progressive architecture ablation

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Weighted Centroid | 6.070 | 5.122 | 5.732 | 45.36% | 4.527/3.580% |
| + GRU | 6.352 | 5.543 | 6.064 | 42.02% | 6.066/4.375% |
| + Kalman | 4.805 | 4.124 | 4.563 | 60.27% | 0.044/0.159% |
| + final MS | 2.190 | 1.789 | 2.047 | 95.53% | 0.000/0.000% |

## Table 2. Temporal input-frame ablation

Each row is trained separately on Route A with the same original v39 training settings.

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 2.212 | 1.822 | 2.073 | 4.133 |
| 2 | current + previous; first difference | 2.227 | 1.822 | 2.083 | 4.143 |
| 3 | current + previous two; first + second difference | 2.190 | 1.789 | 2.047 | 4.077 |

## Table 3. Final MeanShift window sensitivity and PURE decoder latency

Latency starts only after the final candidate centers and regularized logits are ready. It measures **one soft_mean_shift call through metric XY output only**; candidate search/indexing, SAT projection, similarity scoring and prior-logit construction are excluded.

| Window | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.216 | 4.014 ms | 249.1 |
| 5x5 | 25 | 2.047 | 5.044 ms | 198.3 |
| 6x6 (main) | 36 | 2.047 | 7.024 ms | 142.4 |
| 7x7 | 49 | 2.046 | 10.125 ms | 98.8 |
| 8x8 | 64 | 2.046 | 10.368 ms | 96.4 |

## Table 4. Online end-to-end runtime

Prepared UAV tensor -> backbone -> Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> one final MS -> XY.

- Route B: 34.543 ms / 28.9 FPS
- Route C: 33.144 ms / 30.2 FPS
- Weighted B+C: **34.045 ms / 29.4 FPS**
