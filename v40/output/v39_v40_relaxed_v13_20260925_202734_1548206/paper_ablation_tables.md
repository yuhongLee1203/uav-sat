# Bearing-UAV 4-city ablation (measured)

Dataset domains: CITYA, CITYB, CITYC, CITYD; held-out sequences: test_01/test_02.
R@1* uses Bearing-UAV's same-quadrant/sign rule on the tracker's continuous final XY.
HSR/MHE use predicted heading with shared causal state-direction fusion alpha=0.00 for every compared row.

## Component removal

| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 92.17% | 100.00% | 43.12% | 4.334 | 34.59 |
| w/o Kalman | 93.45% | 100.00% | 39.95% | 4.068 | 35.10 |
| w/o MeanShift | 84.89% | 95.28% | 39.27% | 7.920 | 36.30 |
| Full | 93.39% | 100.00% | 39.95% | 4.076 | 35.11 |

## Temporal input

| Input | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| 1 frame | 93.05% | 100.00% | 32.12% | 3.883 | 37.47 |
| 2 frames | 93.52% | 100.00% | 31.98% | 3.931 | 37.65 |
| Full | 93.39% | 100.00% | 39.95% | 4.076 | 35.11 |

## Final MeanShift window

| Window | Candidates | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ | MS latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 92.38% | 99.60% | 39.81% | 4.760 | 35.47 | 4.204 |
| 5x5 | 25 | 93.39% | 100.00% | 39.95% | 4.081 | 35.11 | 5.097 |
| 6x6 | 36 | 93.39% | 100.00% | 39.95% | 4.076 | 35.11 | 7.118 |
| 7x7 | 49 | 93.45% | 100.00% | 39.88% | 4.001 | 35.03 | 8.778 |
| 8x8 | 64 | 93.45% | 100.00% | 39.88% | 4.001 | 35.03 | 11.371 |

## Notes

- P90/P95/P99/CVaR and jump-rate are not included in paper-facing tables.
- HSR@15 is the percentage of frames with heading error <= 15 degrees.
- MHE is mean absolute heading error in degrees.
- Heading fusion uses only predicted recurrent heading and causal estimator-state displacement; GT is used only to score the final heading prediction.

## Integrity audit

`LOCALIZATION_TREND_CHECK=FAIL`

At least one measured ablation is better on MLE or LSR@15; report that trade-off rather than forcing a preferred ranking.
