# Bearing-UAV 4-city ICLR ablation (measured)

Dataset domain(s): CITYA, CITYB, CITYC, CITYD; held-out sequences are reported as test_01/test_02.
The local-search protocol is reported separately from the official Bearing-UAV global-regression benchmark.

## Component removal

| Variant | MLE (m) | P90 (m) | LSR@5 | Jump rate | Speed error (m/frame) |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 4.275 | 7.419 | 65.92% | 0.475% | 5.921 |
| w/o Kalman | 3.763 | 6.818 | 73.82% | 0.610% | 2.569 |
| w/o MeanShift | 6.785 | 11.511 | 33.40% | 5.631% | 2.546 |
| Full | 3.759 | 6.837 | 73.75% | 0.610% | 2.546 |

## Temporal input

| Input | MLE (m) | P90 (m) | LSR@3 | LSR@5 |
|---|---:|---:|---:|---:|
| 1 frame | 3.883 | 6.987 | 39.81% | 72.47% |
| 2 frames | 3.921 | 7.071 | 39.14% | 71.79% |
| Full | 3.759 | 6.837 | 41.90% | 73.75% |

## Final MeanShift window

| Window | Candidates | MLE (m) | P90 (m) | LSR@5 | MS latency (ms) |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 | 4.168 | 7.660 | 67.14% | 3.013 |
| 5x5 | 25 | 3.763 | 6.867 | 73.75% | 3.962 |
| 6x6 | 36 | 3.759 | 6.837 | 73.75% | 5.615 |
| 7x7 | 49 | 3.738 | 6.821 | 73.95% | 6.342 |
| 8x8 | 64 | 3.738 | 6.822 | 74.09% | 8.542 |

## Heading diagnostic

Heading is reported as a diagnostic because the current minimal-loss run sets the heading-loss weight to zero.

| Variant | HSR@15 | MHE (deg) |
|---|---:|---:|
| w/o GRU | 39.54% | 37.05 |
| w/o Kalman | 39.61% | 36.81 |
| w/o MeanShift | 39.20% | 37.75 |
| Full | 39.61% | 36.87 |

## Interpretation

- Full vs. w/o Kalman: MLE changes by -0.004 m, while jump rate changes from 0.610% to 0.610%.
- Three frames have the best MLE/P90 (3.759/6.837 m); the LSR@5 difference from one frame is not statistically resolved by the paired bootstrap.
- 6x6 is the balance point: only +0.021 m MLE behind 7x7, with 11.5% lower MeanShift latency.

## Integrity audit

`MLE_PRIMARY_TREND_CHECK=PASS`

`STRICT_ALL_METRICS_FULL_BEST=FAIL`

Full has the best MLE in both component and temporal ablations, but does not win every secondary metric; report the measured trade-offs without claiming universal dominance.
