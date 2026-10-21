# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 160.115 | 255.148 | 4.97% | 2.48% |
| citya | test_02 | 219 | 250.616 | 404.397 | 3.20% | 2.28% |
| cityb | test_01 | 146 | 140.055 | 215.035 | 2.05% | 2.05% |
| cityb | test_02 | 171 | 259.203 | 407.002 | 2.92% | 2.34% |
