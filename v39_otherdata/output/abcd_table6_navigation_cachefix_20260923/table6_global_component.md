# Bearing-UAV Table-6-style ABCD component ablation

One shared model/profile; temporal state resets only at independent city/route boundaries.

| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 30.03% | 2.43% | 20.58% | 319.285 | 56.62 |
| w/o Kalman | 30.03% | 2.90% | 20.45% | 307.008 | 56.05 |
| w/o MeanShift | 30.30% | 3.44% | 20.72% | 307.375 | 55.82 |
| Full | 30.30% | 3.24% | 20.72% | 307.414 | 55.83 |

Full strictly best on all five metrics vs every component removal: False
