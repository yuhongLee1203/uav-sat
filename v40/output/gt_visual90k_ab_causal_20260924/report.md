# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 88.421 | 149.693 | 6.83% | 4.35% |
| citya | test_02 | 219 | 90.088 | 137.693 | 2.74% | 1.37% |
| cityb | test_01 | 146 | 88.685 | 150.322 | 2.05% | 2.05% |
| cityb | test_02 | 171 | 107.746 | 166.261 | 2.92% | 2.92% |
