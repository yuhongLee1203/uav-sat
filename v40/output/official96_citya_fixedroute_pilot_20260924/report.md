# Bearing-UAV prepared-route model evaluation

Training supervision: `gt_train`; temporal initialization: `random_temporal`.
Inference does not read the current-frame GT column. Held-out waypoints are predeclared independently of selected GT; GT was used offline to assemble synthetic episodes.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 64 | 10.336 | 17.893 | 81.25% | 75.00% |
| citya | test_02 | 89 | 8.891 | 16.287 | 85.39% | 75.28% |
