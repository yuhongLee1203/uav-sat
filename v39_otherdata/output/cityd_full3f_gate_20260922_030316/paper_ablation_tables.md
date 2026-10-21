# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYD; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.283 | 7.136 | 29.68% | 65.21% | 98.05% |
| w/o Kalman | 4.068 | 7.097 | 34.31% | 72.51% | 98.05% |
| w/o MeanShift | 7.752 | 12.563 | 10.95% | 25.55% | 70.32% |
| Full | 4.091 | 7.151 | 33.82% | 72.26% | 97.32% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 4.068 | 7.064 | 34.31% | 72.26% |
| 2 frames | 3.982 | 6.854 | 36.50% | 71.78% |
| Full | 4.091 | 7.151 | 33.82% | 72.26% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.688 | 8.480 | 58.64% | 2.788 |
| 5x5 | 25 | 4.096 | 7.150 | 71.53% | 3.815 |
| 6x6 | 36 | 4.091 | 7.151 | 72.26% | 5.066 |
| 7x7 | 49 | 4.049 | 6.911 | 72.99% | 6.418 |
| 8x8 | 64 | 4.050 | 6.911 | 72.99% | 10.283 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
