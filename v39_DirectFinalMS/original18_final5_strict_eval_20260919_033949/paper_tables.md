# V39 SoftMS Core Ablation Results

Main pipeline: **Forward 3x6 SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final SoftMS**.

**Forward 3x6 is fixed in all core component ablations.** A full-6x6 replacement is intentionally excluded because it changes directionality, candidate count, compute budget, and search semantics at the same time.

All B+C metrics are recomputed from concatenated raw per-frame errors.

## Table 1. Core component removal (w/o)

| Setting | Front | GRU | Kalman | Final MS | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Full | Forward 3x6 SoftMS | yes | fixed | yes | 2.128 | 1.731 | 1.986 | 3.846 | 96.69% |
| w/o GRU | Forward 3x6 SoftMS | no | fixed | yes | 1.976 | 1.593 | 1.840 | 3.638 | 97.31% |
| w/o Kalman | Forward 3x6 SoftMS | yes | none | yes | 2.613 | 2.179 | 2.458 | 4.494 | 93.35% |
| w/o Final MS | Forward 3x6 SoftMS | yes | fixed | no | 4.629 | 3.975 | 4.396 | 8.138 | 61.49% |

## Table 2. Temporal input frames

Each 1/2/3-frame row is freshly trained on Route A with identical settings. Forward 3x6 SoftMS is fixed.

| UAV frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.116 | 1.722 | 1.976 | 3.866 | 96.77% |
| 2 | 2.127 | 1.730 | 1.986 | 3.872 | 96.75% |
| 3 | 2.128 | 1.731 | 1.986 | 3.852 | 96.69% |

## Table 3. Final MeanShift window sensitivity

Front Forward-3x6 SoftMS is fixed. Pure final-MS latency starts after final candidate centers/logits are ready.

| Window | Candidates | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency | Pure final-MS FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.279 | 1.884 | 2.138 | 4.294 | 93.72% | 3.453 ms | 289.62 |
| 5x5 | 25 | 2.128 | 1.731 | 1.986 | 3.852 | 96.69% | 4.665 ms | 214.34 |
| 6x6 | 36 | 2.128 | 1.731 | 1.987 | 3.852 | 96.69% | 6.525 ms | 153.25 |
| 7x7 | 49 | 2.126 | 1.729 | 1.985 | 3.849 | 96.77% | 10.269 ms | 97.38 |
| 8x8 | 64 | 2.127 | 1.729 | 1.985 | 3.846 | 96.77% | 9.743 ms | 102.64 |

## Full-model end-to-end runtime

- Mean: **32.166 ms/frame**
- FPS: **31.09**
