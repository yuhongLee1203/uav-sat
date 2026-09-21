# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYC; held-out navigation trajectories are reported as nav50/nav51.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.091 | 7.109 | 36.01% | 69.95% | 98.45% |
| w/o Kalman | 3.677 | 6.647 | 44.56% | 76.68% | 98.19% |
| w/o MeanShift | 6.863 | 11.782 | 15.28% | 35.49% | 80.57% |
| Full | 3.667 | 6.631 | 44.82% | 76.68% | 98.19% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.661 | 6.639 | 44.30% | 76.42% |
| 2 frames | 3.661 | 6.628 | 44.56% | 76.42% |
| Full | 3.667 | 6.631 | 44.82% | 76.68% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.204 | 7.864 | 69.17% | 2.815 |
| 5x5 | 25 | 3.666 | 6.630 | 76.68% | 3.725 |
| 6x6 | 36 | 3.667 | 6.631 | 76.68% | 5.059 |
| 7x7 | 49 | 3.643 | 6.603 | 77.46% | 6.524 |
| 8x8 | 64 | 3.643 | 6.606 | 77.46% | 7.940 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
