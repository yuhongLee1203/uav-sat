# V39 5x5/15 SoftMS Ablation Results

Main candidate pipeline: **5x5 geometry -> forward 15 -> front SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final 5x5 SoftMS**.

Forward 5x5->15 is fixed in all component and final-MS-window rows.

## Table 1. Component ablation

| Setting | GRU | Kalman | Final MS | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Full | yes | fixed | yes | 1.968 | 1.728 | 1.882 | 3.489 | 97.57% |
| w/o GRU | no | fixed | yes | 1.789 | 1.558 | 1.706 | 3.385 | 97.82% |
| w/o Kalman | yes | none | yes | 2.258 | 2.001 | 2.167 | 4.041 | 96.15% |
| w/o Final MS | yes | fixed | no | 4.300 | 3.713 | 4.091 | 7.266 | 68.22% |

## Table 2. Temporal frames

| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.929 | 1.673 | 1.838 | 3.535 | 97.65% |
| 2 | 1.927 | 1.664 | 1.834 | 3.472 | 97.62% |
| 3 | 1.968 | 1.728 | 1.882 | 3.489 | 97.57% |

## Table 3. Final MeanShift window

| Window | Candidates | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.015 | 1.804 | 1.940 | 3.608 | 96.21% | 4.257 ms | 234.91 |
| 5x5 | 25 | 1.968 | 1.728 | 1.882 | 3.489 | 97.57% | 4.637 ms | 215.67 |
| 6x6 | 36 | 1.967 | 1.728 | 1.882 | 3.488 | 97.57% | 6.336 ms | 157.84 |
| 7x7 | 49 | 1.967 | 1.727 | 1.881 | 3.489 | 97.57% | 7.649 ms | 130.74 |
| 8x8 | 64 | 1.967 | 1.728 | 1.881 | 3.488 | 97.57% | 9.583 ms | 104.35 |

## Full-model E2E runtime

- Mean: **33.311 ms/frame**
- FPS: **30.02**
