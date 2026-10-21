# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYA; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.274 | 7.303 | 32.15% | 65.67% | 98.64% |
| w/o Kalman | 3.946 | 6.843 | 38.69% | 71.12% | 98.91% |
| w/o MeanShift | 6.823 | 12.111 | 16.89% | 38.15% | 79.29% |
| Full | 3.943 | 6.805 | 38.42% | 71.39% | 99.18% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.941 | 6.806 | 38.96% | 72.21% |
| 2 frames | 3.967 | 6.799 | 38.42% | 70.84% |
| Full | 3.943 | 6.805 | 38.42% | 71.39% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.462 | 8.149 | 63.76% | 3.838 |
| 5x5 | 25 | 3.943 | 6.805 | 71.39% | 3.778 |
| 6x6 | 36 | 3.943 | 6.805 | 71.39% | 5.093 |
| 7x7 | 49 | 3.899 | 6.765 | 72.48% | 6.350 |
| 8x8 | 64 | 3.900 | 6.766 | 72.48% | 7.797 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
