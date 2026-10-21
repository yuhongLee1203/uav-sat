# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 21.812 | 40.140 | 33.54% | 33.54% |
| citya | test_02 | 219 | 35.549 | 73.210 | 22.83% | 21.46% |
| cityb | test_01 | 146 | 15.446 | 29.005 | 54.11% | 54.11% |
| cityb | test_02 | 171 | 17.614 | 40.604 | 56.14% | 56.14% |
