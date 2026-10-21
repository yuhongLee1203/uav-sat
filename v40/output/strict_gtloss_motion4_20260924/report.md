# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 9.313 | 16.105 | 87.58% | 83.85% |
| citya | test_02 | 219 | 25.653 | 48.983 | 35.16% | 36.53% |
