# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `random_temporal`.
Inference does not read the current-frame GT column. Held-out waypoints are predeclared independently of selected GT; GT was used offline to assemble synthetic episodes.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 64 | 10.276 | 17.851 | 84.38% | 75.00% |
| citya | test_02 | 89 | 9.858 | 17.643 | 82.02% | 76.40% |
