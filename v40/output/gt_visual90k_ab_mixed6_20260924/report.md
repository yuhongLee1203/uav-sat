# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 122.116 | 219.710 | 6.21% | 1.86% |
| citya | test_02 | 219 | 134.710 | 201.973 | 3.20% | 1.37% |
| cityb | test_01 | 146 | 130.131 | 221.833 | 2.05% | 2.05% |
| cityb | test_02 | 171 | 173.497 | 264.298 | 2.92% | 2.34% |
