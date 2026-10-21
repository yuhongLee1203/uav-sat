# Bearing-UAV 4-city ablation (measured)

Dataset domains: CITYA, CITYB, CITYC, CITYD; held-out sequences: test_01/test_02.
R@1* uses Bearing-UAV's same-quadrant/sign rule on the tracker's continuous final XY.

## Component removal

| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 92.11% | 99.87% | 39.54% | 4.275 | 37.05 |
| w/o Kalman | 93.66% | 100.00% | 39.61% | 3.763 | 36.81 |
| w/o MeanShift | 86.77% | 98.04% | 39.20% | 6.785 | 37.75 |
| Full | 93.66% | 100.00% | 39.61% | 3.759 | 36.87 |

## Temporal input

| Input | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| 1 frame | 93.32% | 100.00% | 39.68% | 3.883 | 36.90 |
| 2 frames | 93.59% | 100.00% | 37.85% | 3.921 | 37.01 |
| Full | 93.66% | 100.00% | 39.61% | 3.759 | 36.87 |

## Final MeanShift window

| Window | Candidates | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ | MS latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 92.98% | 99.93% | 39.47% | 4.168 | 36.96 | 3.013 |
| 5x5 | 25 | 93.66% | 100.00% | 39.61% | 3.763 | 36.87 | 3.962 |
| 6x6 | 36 | 93.66% | 100.00% | 39.61% | 3.759 | 36.87 | 6.896 |
| 7x7 | 49 | 93.72% | 100.00% | 39.68% | 3.738 | 36.85 | 6.342 |
| 8x8 | 64 | 93.72% | 100.00% | 39.68% | 3.738 | 36.85 | 8.542 |

## Notes

- P90/P95/P99/CVaR and jump-rate are not included in paper-facing tables.
- HSR@15 is the percentage of frames with heading error <= 15 degrees.
- MHE is mean absolute heading error in degrees.

## Integrity audit

`LOCALIZATION_TREND_CHECK=PASS`

Full is numerically best on MLE and LSR@15 across the checked component/temporal ablations; inspect confidence intervals before claiming significance.
