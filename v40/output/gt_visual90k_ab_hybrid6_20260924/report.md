# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 86.787 | 137.958 | 6.83% | 4.35% |
| citya | test_02 | 219 | 84.556 | 126.491 | 2.74% | 1.83% |
| cityb | test_01 | 146 | 90.164 | 151.036 | 2.05% | 2.05% |
| cityb | test_02 | 171 | 113.116 | 165.556 | 2.92% | 2.92% |
