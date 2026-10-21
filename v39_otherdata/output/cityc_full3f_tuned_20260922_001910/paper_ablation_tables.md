# Bearing-UAV single-city ICLR ablation (measured)

Dataset domain(s): CITYC; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.169 | 7.173 | 33.69% | 67.38% | 98.40% |
| w/o Kalman | 3.609 | 6.406 | 44.12% | 75.67% | 99.20% |
| w/o MeanShift | 6.422 | 11.126 | 17.65% | 36.36% | 84.49% |
| Full | 3.611 | 6.400 | 44.12% | 75.94% | 99.20% |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.768 | 6.541 | 40.37% | 73.80% |
| 2 frames | 3.667 | 6.407 | 42.78% | 75.13% |
| Full | 3.611 | 6.400 | 44.12% | 75.94% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.021 | 7.195 | 69.79% | 2.802 |
| 5x5 | 25 | 3.610 | 6.399 | 75.94% | 3.594 |
| 6x6 | 36 | 3.611 | 6.400 | 75.94% | 5.122 |
| 7x7 | 49 | 3.600 | 6.399 | 76.47% | 6.817 |
| 8x8 | 64 | 3.600 | 6.400 | 76.47% | 7.667 |

## Integrity audit

`FULL_TREND_CHECK=FAIL`

At least one ablation is better on a primary metric. Do not claim every component improves accuracy; revise only through training/validation data, not held-out navigation results.
