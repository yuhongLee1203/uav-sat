# V39 Forward-18 + Final-5x5 SoftMS Ablation Results

Candidate pipeline: **6x6 local geometry -> heading-forward 18 -> front SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final 5x5 SoftMS (BW=7m)**.

Forward 6x6->18 is fixed in all component and final-MS-window rows.

## Table 1. Component ablation

| Setting | GRU | Kalman | Final MS | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Full | yes | fixed | yes | 2.128 | 1.731 | 1.986 | 3.846 | 96.69% |
| w/o GRU | no | fixed | yes | 1.976 | 1.593 | 1.840 | 3.638 | 97.31% |
| w/o Kalman | yes | none | yes | 2.613 | 2.179 | 2.458 | 4.494 | 93.35% |
| w/o Final MS | yes | fixed | no | 4.629 | 3.975 | 4.396 | 8.138 | 61.49% |

## Table 2. Temporal frames

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.116 | 1.722 | 1.976 | 3.866 | 96.77% |
| 2 | 2.127 | 1.730 | 1.986 | 3.872 | 96.75% |
| 3 | 2.128 | 1.731 | 1.986 | 3.852 | 96.69% |

## Table 3. Final MeanShift window

| Window | Candidates | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.279 | 1.884 | 2.138 | 4.294 | 93.72% | 3.413 ms | 292.99 |
| 5x5 | 25 | 2.128 | 1.731 | 1.986 | 3.852 | 96.69% | 4.352 ms | 229.77 |
| 6x6 | 36 | 2.128 | 1.731 | 1.987 | 3.852 | 96.69% | 8.298 ms | 120.51 |
| 7x7 | 49 | 2.126 | 1.729 | 1.985 | 3.849 | 96.77% | 7.909 ms | 126.44 |
| 8x8 | 64 | 2.127 | 1.729 | 1.985 | 3.846 | 96.77% | 9.594 ms | 104.23 |

## Full-model E2E runtime

- Mean: **31.101 ms/frame**
- FPS: **32.15**
