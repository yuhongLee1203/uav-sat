# v39 Paper Tables

## Table 1. Progressive architecture ablation

| Setting | GRU | Kalman | MS | B MLE | C MLE | B+C MLE | B LSR@5 | C LSR@5 | B/C Jump |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| Baseline | no | no | no | 5.597 | 4.645 | 5.258 | 47.23% | 58.51% | 2.857/2.864% |
| + GRU | yes | no | no | 5.769 | 5.027 | 5.505 | 44.46% | 53.97% | 3.473/3.182% |
| + GRU + Kalman | yes | yes | no | 4.481 | 3.883 | 4.268 | 60.76% | 68.92% | 0.044/0.000% |
| + GRU + Kalman + MS | yes | yes | yes | 2.085 | 1.704 | 1.950 | 97.67% | 95.79% | 0.000/0.000% |

## Table 2. GRU motion-model design

| Motion | B MLE | C MLE | B+C MLE | B Speed MAE | C Speed MAE |
|---|---:|---:|---:|---:|---:|
| None | 2.407 | 1.944 | 2.242 | 0.649 | 1.188 |
| Velocity | 2.082 | 1.697 | 1.945 | 0.649 | 1.188 |
| Quadratic | 2.085 | 1.704 | 1.950 | 0.649 | 1.188 |

## Table 3. Kalman measurement design

| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |
|---|---:|---:|---:|---:|
| No Kalman | 2.581 | 2.165 | 2.433 | 0.659/0.636% |
| Learned variance | 2.183 | 1.764 | 2.034 | 0.000/0.000% |
| Fixed variance (selected) | 2.085 | 1.704 | 1.950 | 0.000/0.000% |

## Table 4. MS window accuracy-efficiency trade-off

| Window | Candidates | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 16 | 2.227 | 1.830 | 2.086 | 10.495 | 95.3 |
| 6x6 | 36 | 2.085 | 1.704 | 1.950 | 20.538 | 48.7 |
| 8x8 | 64 | 2.085 | 1.703 | 1.949 | 106.638 | 9.4 |

## Table 5. MeanShift bandwidth accuracy-efficiency trade-off

| Bandwidth | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |
|---:|---:|---:|---:|---:|---:|
| 3 m | 2.264 | 1.891 | 2.131 | 97.705 | 10.2 |
| 5 m | 2.119 | 1.737 | 1.983 | 73.945 | 13.5 |
| 7 m (selected) | 2.085 | 1.704 | 1.950 | 20.538 | 48.7 |
