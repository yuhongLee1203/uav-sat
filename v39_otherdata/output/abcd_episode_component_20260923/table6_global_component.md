# v39 temporal ABCD route component ablation

Independent navigation routes from four cities are pooled. This is not Bearing-UAV's official rsi_id=96 static Table 6 protocol.
Training: 4 epochs; city episode order is shuffled from the recorded seed. GRU/Kalman state resets for each route; weights and optimizer persist.


| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |
|---|---:|---:|---:|---:|---:|
| w/o GRU | 31.92% | 4.25% | 19.16% | 254.266 | 56.44 |
| w/o Kalman | 39.00% | 11.20% | 18.96% | 162.312 | 60.37 |
| w/o MeanShift | 38.39% | 12.28% | 18.56% | 165.002 | 61.60 |
| Full | 38.53% | 12.01% | 18.56% | 165.310 | 61.64 |

Full strictly best on all five metrics vs every component removal: False
