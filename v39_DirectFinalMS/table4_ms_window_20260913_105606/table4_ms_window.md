# Table 4. Accuracy-Efficiency Trade-off of Final MeanShift Window Size

| Final MS Window | Candidates | MLE (m) ↓ | LSR@3 (%) ↑ | LSR@5 (%) ↑ | E2E Latency (ms) ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.214 | 75.27 | 92.02 | 28.304 | 35.3 |
| **5x5** | 25 | **2.045** | 77.56 | 95.73 | **28.432** | **35.2** |
| 6x6 | 36 | 2.045 | 77.56 | 95.73 | 30.314 | 33.0 |
| 7x7 | 49 | 2.043 | 77.56 | 95.78 | 32.190 | 31.1 |
| 8x8 | 64 | 2.043 | 77.56 | 95.78 | 33.502 | 29.8 |

Best MLE: 2.043 m
0.5% MLE threshold: 2.053 m
Selected window: **5x5**

Selection rule: among configurations whose MLE is within 0.5% of the best MLE, select the configuration with the lowest end-to-end latency.
