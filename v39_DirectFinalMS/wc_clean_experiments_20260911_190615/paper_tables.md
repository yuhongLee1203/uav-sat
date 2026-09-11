# Clean v39 Weighted-Centroid Ablation Tables

**Only methodological change:** front SoftMS/MS1 is replaced by posterior Weighted Centroid with posterior spatial variance. The original v39 temporal checkpoint and all remaining inference settings are unchanged.

## Table 1. Progressive architecture ablation

| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |
|---|---:|---:|---:|---:|---:|
| WC + GRU | 6.284 | 5.504 | 6.006 | 42.28% | 5.846/4.535% |
| WC + GRU + Kalman | 4.724 | 4.046 | 4.482 | 60.95% | 0.044/0.159% |
| WC + GRU + Kalman + MS | 2.162 | 1.747 | 2.014 | 95.67% | 0.000/0.000% |

## Table 2. Motion prediction model (all use the original 3-frame GRU)

| Motion | B MLE | C MLE | B+C MLE |
|---|---:|---:|---:|
| No learned motion | 2.476 | 2.264 | 2.401 |
| Constant Velocity | 2.162 | 1.747 | 2.014 |
| Velocity + Acceleration | 2.163 | 1.748 | 2.015 |

## Table 3. Kalman design

| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |
|---|---:|---:|---:|---:|
| No Kalman | 2.787 | 2.339 | 2.628 | 1.231/0.796% |
| Learned variance | 2.218 | 1.752 | 2.052 | 0.000/0.000% |
| Fixed variance | 2.162 | 1.747 | 2.014 | 0.000/0.000% |

## Table 4. Final-MS window sensitivity and correctly isolated stage latency

All rows were run sequentially on GPU 5. Latency is Kalman output → candidate indexing/scoring → exactly one final MeanShift → XY.

| Window | Candidates | B+C MLE | MS latency | MS FPS |
|---|---:|---:|---:|---:|
| 4x4 | 16 | 2.184 | 7.122 ms | 140.4 |
| 5x5 | 25 | 2.014 | 8.005 ms | 124.9 |
| 6x6 (fixed v39 operating point) | 36 | 2.014 | 10.041 ms | 99.6 |
| 7x7 | 49 | 2.012 | 11.019 ms | 90.8 |
| 8x8 | 64 | 2.013 | 12.951 ms | 77.2 |

## Table 5. MeanShift bandwidth sensitivity

Final-MS window remains fixed at 6x6. B/C results are reported as sensitivity only; they are not used to retune the operating point.

| Bandwidth | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|
| 1 m | 19.291 | 51.283 | 36.53% |
| 2 m | 2.701 | 5.077 | 89.59% |
| 3 m | 2.195 | 4.306 | 94.62% |
| 4 m | 2.094 | 4.123 | 95.30% |
| 5 m | 2.047 | 4.058 | 95.56% |
| 6 m | 2.026 | 4.008 | 95.67% |
| 7 m (fixed v39 operating point) | 2.014 | 3.979 | 95.67% |
| 8 m | 2.008 | 3.971 | 95.64% |
| 9 m | 2.004 | 3.963 | 95.59% |
| 10 m | 2.001 | 3.956 | 95.56% |
| 11 m | 1.999 | 3.952 | 95.59% |
| 12 m | 1.998 | 3.951 | 95.61% |
| 13 m | 1.997 | 3.952 | 95.64% |
| 14 m | 1.996 | 3.950 | 95.67% |

## Table 6. End-to-end runtime

Prepared UAV tensor → backbone → Weighted Centroid → original GRU → original Kalman → exactly one final MS → XY.

- Route B: 34.308 ms / 29.1 FPS
- Route C: 33.480 ms / 29.9 FPS
- Weighted B+C: **34.013 ms / 29.4 FPS**
