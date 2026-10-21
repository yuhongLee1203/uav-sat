# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `random_temporal`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 137.960 | 242.847 | 6.83% | 4.35% |
| citya | test_02 | 219 | 233.398 | 454.946 | 4.57% | 2.28% |
