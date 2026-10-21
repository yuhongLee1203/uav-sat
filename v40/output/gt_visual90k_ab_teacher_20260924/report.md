# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 21.001 | 37.067 | 37.89% | 35.40% |
| citya | test_02 | 219 | 43.621 | 84.040 | 18.72% | 19.63% |
| cityb | test_01 | 146 | 12.765 | 23.672 | 66.44% | 67.81% |
| cityb | test_02 | 171 | 21.427 | 46.533 | 43.86% | 41.52% |
