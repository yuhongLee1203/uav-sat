# Final MeanShift Window Accuracy–Latency Study

Latency definition: Kalman XY center -> nearest permanent SAT lattice anchor -> NxN candidate extraction -> SAT projection/scoring -> spatial-prior regularized logits -> MeanShift -> final metric XY.

Each configuration is measured five times on physical GPU6 (RTX 3090). The reported latency is the median of the five B+C pooled run-level mean latencies.

| Final MS Window | Candidates | MLE (m) ↓ | LSR@3 (%) ↑ | LSR@5 (%) ↑ | Final-MS Stage Latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.214 | 75.27 | 92.05 | 5.757 |
| 5x5 | 25 | 2.045 | 77.56 | 95.76 | 6.956 |
| 6x6 | 36 | 2.045 | 77.56 | 95.76 | 9.244 |
| 7x7 | 49 | 2.043 | 77.56 | 95.81 | 11.034 |
| 8x8 | 64 | 2.043 | 77.56 | 95.81 | 12.461 |
