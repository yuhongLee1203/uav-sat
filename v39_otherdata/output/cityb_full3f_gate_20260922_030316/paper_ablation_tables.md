# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYB; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.212 | 7.288 | 33.75% | 68.77% | 98.11% |
| w/o Kalman | 3.725 | 6.696 | 42.59% | 74.45% | 99.37% |
| w/o MeanShift | 6.297 | 11.119 | 18.30% | 36.28% | 85.49% |
| Full | 3.711 | 6.668 | 41.64% | 74.45% | 99.37% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.763 | 6.685 | 41.01% | 73.82% |
| 2 frames | 3.734 | 6.663 | 42.59% | 74.13% |
| Full | 3.711 | 6.668 | 41.64% | 74.45% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 3.952 | 7.133 | 71.92% | 2.882 |
| 5x5 | 25 | 3.717 | 6.666 | 74.13% | 4.417 |
| 6x6 | 36 | 3.711 | 6.668 | 74.45% | 5.105 |
| 7x7 | 49 | 3.708 | 6.666 | 74.45% | 6.718 |
| 8x8 | 64 | 3.708 | 6.668 | 74.45% | 7.750 |

## Integrity audit

`FULL_TREND_CHECK=PASS`

Full is numerically best on the predeclared primary metrics; inspect paired confidence intervals before claiming significance.
