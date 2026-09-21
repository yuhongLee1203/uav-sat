# Final Weighted-Centroid Ablation Tables

Main method: **Front SoftMS(18) -> 3-frame GRU -> fixed-R external Kalman -> one final MeanShift**.

Constant Velocity is fixed for every experiment; there is no separate motion-model table. The Kalman is not trained; its contribution is tested only by the progressive architecture ablation.

## Table 1. Progressive architecture ablation

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| Front SoftMS(18) | 5.271 | 4.525 | 5.005 | 54.84% | 2.330/2.387% |
| + GRU | 5.504 | 4.949 | 5.306 | 51.78% | 2.769/2.705% |
| + Kalman | 4.485 | 3.965 | 4.300 | 64.18% | 0.044/0.000% |
| + final MS | 2.067 | 1.752 | 1.955 | 97.51% | 0.000/0.000% |

## Table 2. Temporal input-frame ablation

Each row is trained separately on Route A with the same original v39 training settings.

| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |
|---:|---|---:|---:|---:|---:|
| 1 | current frame only | 2.025 | 1.745 | 1.925 | 3.592 |
| 2 | current + previous; first difference | 2.045 | 1.759 | 1.943 | 3.664 |
| 3 | current + previous two; first + second difference | 2.067 | 1.752 | 1.955 | 3.688 |

## Table 3. Final MeanShift window sensitivity and PURE decoder latency

Latency starts only after the final candidate centers and regularized logits are ready. It measures **one soft_mean_shift call through metric XY output only**; candidate search/indexing, SAT projection, similarity scoring and prior-logit construction are excluded.

| Window | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.080 | 3.440 ms | 290.7 |
| 5x5 | 25 | 1.955 | 4.447 ms | 224.9 |
| 6x6 (main) | 36 | 1.955 | 6.505 ms | 153.7 |
| 7x7 | 49 | 1.954 | 10.129 ms | 98.7 |
| 8x8 | 64 | 1.954 | 9.093 ms | 110.0 |

## Table 4. Online end-to-end runtime

Prepared UAV tensor -> backbone -> Front SoftMS(18) -> 3-frame GRU -> fixed-R Kalman -> one final MS -> XY.

- Route B: 30.831 ms / 32.4 FPS
- Route C: 30.635 ms / 32.6 FPS
- Weighted B+C: **30.761 ms / 32.5 FPS**
