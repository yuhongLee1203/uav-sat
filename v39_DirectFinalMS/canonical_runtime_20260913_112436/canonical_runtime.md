# Canonical Runtime Results

Runtime protocol: isolated physical GPU6 (RTX 3090), three repetitions, 30-frame warm-up per route. Reported latency is the median of the three B+C run-level mean latencies. FPS = 1000 / canonical latency.

| Final MS Window | Candidates | MLE (m) ↓ | LSR@3 (%) ↑ | LSR@5 (%) ↑ | E2E Latency (ms) ↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.214 | 75.27 | 92.02 | 34.315 | 29.1 |
| 5x5 | 25 | 2.045 | 77.56 | 95.73 | 35.627 | 28.1 |
| 6x6 | 36 | 2.045 | 77.56 | 95.73 | 36.468 | 27.4 |
| 7x7 | 49 | 2.043 | 77.56 | 95.78 | 33.061 | 30.2 |
| 8x8 | 64 | 2.043 | 77.56 | 95.78 | 34.464 | 29.0 |

## Final Proposed Configuration

- Final MS window: **5x5**
- Canonical E2E latency: **35.627 ms**
- Canonical FPS: **28.1 FPS**

Use this exact 5x5 latency/FPS in both Table 1 and Table 4.
