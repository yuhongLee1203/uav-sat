# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 13.193 | 25.597 | 65.22% | 62.73% |
| citya | test_02 | 219 | 25.720 | 50.412 | 31.51% | 34.25% |
| cityb | test_01 | 146 | 14.687 | 24.814 | 58.22% | 58.22% |
| cityb | test_02 | 171 | 32.795 | 45.312 | 2.92% | 2.92% |
