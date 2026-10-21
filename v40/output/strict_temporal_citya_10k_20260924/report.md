# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `random_temporal`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 79.022 | 142.875 | 7.45% | 7.45% |
| citya | test_02 | 219 | 80.905 | 129.949 | 5.48% | 3.20% |
