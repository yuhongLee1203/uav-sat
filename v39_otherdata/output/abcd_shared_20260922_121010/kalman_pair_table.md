# Full vs w/o Kalman (held-out, measured)

| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| w/o Kalman | 84.55% | 96.36% | 38.93% | 7.475 | 37.12 |
| Full | 84.89% | 96.36% | 38.87% | 7.425 | 37.17 |

Full − w/o Kalman MLE: -0.0498 m
Paired MLE 95% CI: [-0.0860, -0.0180] m

R@1* is the Bearing-UAV same-quadrant/sign criterion derived from continuous final XY.
