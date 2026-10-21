# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYC; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.084 | 7.063 | 34.76% | 68.18% | 98.66% |
| w/o Kalman | 3.604 | 6.397 | 44.92% | 75.40% | 99.20% |
| w/o MeanShift | 6.413 | 11.220 | 18.18% | 33.16% | 83.42% |
| Full | 3.611 | 6.295 | 44.39% | 74.87% | 99.47% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.601 | 6.344 | 43.32% | 75.67% |
| 2 frames | 3.623 | 6.352 | 44.12% | 75.13% |
| Full | 3.611 | 6.295 | 44.39% | 74.87% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 3.973 | 7.016 | 69.52% | 2.853 |
| 5x5 | 25 | 3.610 | 6.296 | 74.87% | 3.726 |
| 6x6 | 36 | 3.611 | 6.295 | 74.87% | 5.042 |
| 7x7 | 49 | 3.597 | 6.295 | 75.67% | 6.521 |
| 8x8 | 64 | 3.597 | 6.295 | 75.67% | 10.317 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
