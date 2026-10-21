# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYC; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.153 | 7.166 | 33.42% | 67.11% | 98.66% |
| w/o Kalman | 3.641 | 6.388 | 43.85% | 75.67% | 99.20% |
| w/o MeanShift | 6.541 | 11.216 | 17.38% | 35.03% | 83.96% |
| Full | 3.646 | 6.372 | 44.12% | 75.67% | 99.20% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.705 | 6.516 | 43.05% | 74.60% |
| 2 frames | 3.688 | 6.481 | 43.32% | 74.33% |
| Full | 3.646 | 6.372 | 44.12% | 75.67% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.082 | 7.204 | 68.72% | 2.901 |
| 5x5 | 25 | 3.645 | 6.373 | 75.67% | 3.783 |
| 6x6 | 36 | 3.646 | 6.372 | 75.67% | 5.160 |
| 7x7 | 49 | 3.633 | 6.372 | 75.94% | 6.407 |
| 8x8 | 64 | 3.634 | 6.372 | 75.94% | 7.671 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
