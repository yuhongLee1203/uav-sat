# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYA; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.301 | 7.324 | 31.34% | 65.12% | 98.64% |
| w/o Kalman | 3.963 | 6.841 | 38.69% | 70.30% | 98.91% |
| w/o MeanShift | 6.948 | 12.195 | 15.53% | 37.60% | 77.38% |
| Full | 3.962 | 6.839 | 38.42% | 70.03% | 98.91% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.976 | 6.858 | 37.60% | 70.57% |
| 2 frames | 3.935 | 6.820 | 38.69% | 72.21% |
| Full | 3.962 | 6.839 | 38.42% | 70.03% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.503 | 8.158 | 62.40% | 11.027 |
| 5x5 | 25 | 3.962 | 6.838 | 70.03% | 11.595 |
| 6x6 | 36 | 3.962 | 6.839 | 70.03% | 9.464 |
| 7x7 | 49 | 3.930 | 6.808 | 71.39% | 17.718 |
| 8x8 | 64 | 3.930 | 6.809 | 71.39% | 21.898 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
