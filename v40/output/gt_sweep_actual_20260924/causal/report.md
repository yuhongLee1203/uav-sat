# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 17.906 | 30.466 | 40.37% | 40.99% |
| citya | test_02 | 219 | 23.331 | 46.784 | 34.70% | 38.81% |
| cityb | test_01 | 146 | 27.408 | 43.205 | 23.29% | 24.66% |
| cityb | test_02 | 171 | 25.889 | 45.071 | 21.64% | 21.05% |
