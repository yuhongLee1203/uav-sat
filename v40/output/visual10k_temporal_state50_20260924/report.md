# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 26.531 | 44.350 | 30.43% | 28.57% |
| citya | test_02 | 219 | 33.463 | 72.529 | 27.40% | 26.94% |
