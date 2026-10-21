# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `random_temporal`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 16.327 | 30.138 | 50.93% | 77.64% |
| citya | test_02 | 219 | 10.051 | 21.959 | 80.82% | 86.30% |
