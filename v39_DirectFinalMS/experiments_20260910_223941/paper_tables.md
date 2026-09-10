# v39 Paper Tables

All GRU variants use three UAV image frames. 'Constant Velocity' and 'Velocity + Acceleration' refer to the downstream motion equation, not the number of input images.

## Table 1. Progressive architecture ablation

| Setting | GRU | Kalman | MS | B MLE | C MLE | B+C MLE | B LSR@5 | C LSR@5 | B/C Jump |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| GRU | yes | no | no | 5.769 | 5.028 | 5.505 | 44.46% | 53.97% | 3.473/3.182% |
| + Kalman | yes | yes | no | 4.499 | 3.886 | 4.281 | 59.89% | 69.48% | 0.044/0.000% |
| + MS | yes | yes | yes | 2.082 | 1.697 | 1.945 | 97.93% | 95.47% | 0.000/0.000% |

## Table 2. Motion prediction model (all use 3-frame GRU input)

| Motion prediction | Meaning | B MLE | C MLE | B+C MLE |
|---|---|---:|---:|---:|
| No learned motion | Kalman keeps its own previous velocity | 2.407 | 1.944 | 2.242 |
| Constant Velocity (selected) | GRU velocity; acceleration term is not used | 2.082 | 1.697 | 1.945 |
| Velocity + Acceleration | GRU velocity plus acceleration term | 2.085 | 1.704 | 1.950 |

## Table 3. Kalman measurement design

| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |
|---|---:|---:|---:|---:|
| No Kalman | 2.581 | 2.165 | 2.433 | 0.659/0.636% |
| Learned variance | 2.194 | 1.764 | 2.041 | 0.000/0.000% |
| Fixed variance (selected) | 2.082 | 1.697 | 1.945 | 0.000/0.000% |

## Table 4. MS local-window accuracy-efficiency trade-off

All rows are measured sequentially on GPU 5. Selection rule: lowest latency among settings within 0.5% of the best B+C MLE.

| Window | Candidates | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.224 | 1.828 | 2.083 | 90.584 | 11.0 |
| 5x5 (selected) | 25 | 2.081 | 1.696 | 1.944 | 93.582 | 10.7 |
| 6x6 | 36 | 2.082 | 1.697 | 1.945 | 99.613 | 10.0 |
| 7x7 | 49 | 2.080 | 1.695 | 1.943 | 104.005 | 9.6 |
| 8x8 | 64 | 2.081 | 1.696 | 1.944 | 113.677 | 8.8 |

## Table 5. MeanShift bandwidth sensitivity

Adjacent SAT candidate centers are approximately 4.48 m apart. Bandwidth changes the spatial smoothing scale, not the number of MeanShift operations, so latency is intentionally omitted.

| Bandwidth | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |
|---:|---:|---:|---:|---:|---:|
| 1 m | 24.480 | 10.073 | 19.352 | 51.454 | 37.35% |
| 2 m | 2.895 | 2.173 | 2.638 | 4.898 | 90.61% |
| 3 m | 2.260 | 1.885 | 2.127 | 4.068 | 95.93% |
| 4 m | 2.160 | 1.779 | 2.025 | 3.924 | 96.66% |
| 5 m | 2.115 | 1.730 | 1.978 | 3.831 | 96.77% |
| 6 m | 2.093 | 1.708 | 1.956 | 3.813 | 96.94% |
| 7 m | 2.082 | 1.697 | 1.945 | 3.787 | 97.06% |
| 8 m | 2.075 | 1.691 | 1.938 | 3.781 | 97.06% |
| 9 m | 2.071 | 1.687 | 1.934 | 3.778 | 97.06% |
| 10 m | 2.068 | 1.684 | 1.931 | 3.776 | 97.03% |
| 11 m | 2.066 | 1.682 | 1.929 | 3.772 | 97.03% |
| 12 m | 2.065 | 1.681 | 1.928 | 3.768 | 97.06% |
| 13 m | 2.064 | 1.680 | 1.927 | 3.765 | 97.06% |
| 14 m (best) | 2.063 | 1.679 | 1.926 | 3.765 | 97.06% |

## Automatic selection summary

- MS window selected by the predefined accuracy-efficiency rule: **5x5**.
- Best tested MeanShift bandwidth by B+C MLE: **14 m**.
