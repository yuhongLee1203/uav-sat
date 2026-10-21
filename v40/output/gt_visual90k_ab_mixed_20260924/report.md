# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 10.980 | 17.083 | 80.75% | 72.05% |
| citya | test_02 | 219 | 20.867 | 40.991 | 36.53% | 40.18% |
| cityb | test_01 | 146 | 25.161 | 40.399 | 22.60% | 22.60% |
| cityb | test_02 | 171 | 44.832 | 58.272 | 2.92% | 2.92% |
