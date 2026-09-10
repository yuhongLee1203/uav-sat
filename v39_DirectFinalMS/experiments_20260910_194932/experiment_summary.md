# v39 GRU-Kalman-MS Experiment Summary

| Experiment | GRU | Kalman | MS | Motion | MS Grid | BW | B MLE | C MLE | B+C MLE | Delta vs Full | B LSR@5 | C LSR@5 | B Jump | C Jump |
|---|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abl_gru_only | yes | no | no | quadratic | - | - | 5.769 | 5.027 | 5.505 | 166.53% | 44.46% | 53.97% | 3.473% | 3.182% |
| abl_gru_kalman | yes | yes | no | quadratic | - | - | 4.752 | 4.071 | 4.510 | 118.35% | 56.02% | 67.09% | 0.044% | 0.080% |
| abl_gru_ms | yes | no | yes | quadratic | 6 | 5.0 | 2.614 | 2.199 | 2.466 | 19.39% | 92.62% | 94.12% | 0.659% | 0.636% |
| abl_kalman_ms | no | yes | yes | quadratic | 6 | 5.0 | 2.151 | 1.698 | 1.990 | -3.67% | 97.72% | 93.96% | 0.000% | 0.000% |
| full_model | yes | yes | yes | quadratic | 6 | 5.0 | 2.215 | 1.795 | 2.065 | 0.00% | 96.97% | 95.15% | 0.000% | 0.000% |
| design_motion_none | yes | yes | yes | none | 6 | 5.0 | 2.483 | 2.021 | 2.319 | 12.26% | 95.69% | 93.16% | 0.000% | 0.000% |
| design_motion_velocity | yes | yes | yes | velocity | 6 | 5.0 | 2.226 | 1.795 | 2.073 | 0.37% | 97.06% | 94.99% | 0.000% | 0.000% |
| design_kalman_fixed_var | yes | yes | yes | quadratic | 6 | 5.0 | 2.119 | 1.737 | 1.983 | -3.99% | 97.50% | 95.63% | 0.000% | 0.000% |
| sens_ms_grid4x4 | yes | yes | yes | quadratic | 4 | 5.0 | 2.353 | 1.926 | 2.201 | 6.56% | 93.37% | 92.37% | 0.000% | 0.000% |
| sens_ms_grid8x8 | yes | yes | yes | quadratic | 8 | 5.0 | 2.212 | 1.792 | 2.063 | -0.13% | 97.01% | 95.39% | 0.000% | 0.000% |
| sens_ms_bandwidth3 | yes | yes | yes | quadratic | 6 | 3.0 | 2.359 | 1.946 | 2.212 | 7.07% | 95.78% | 94.12% | 0.000% | 0.000% |
| sens_ms_bandwidth7 | yes | yes | yes | quadratic | 6 | 7.0 | 2.183 | 1.764 | 2.034 | -1.53% | 97.19% | 95.15% | 0.000% | 0.000% |
