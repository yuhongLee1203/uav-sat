# Bearing-UAV prepared-route model evaluation

Training supervision: `image_only`; temporal initialization: `shared`.
Testing uses images + map/waypoints, no current-frame GT or route-duration pace prior.
This is a custom route subset of Bearing-UAV 90K, not official Table 6.

| City | Route | Frames | MLE (m) | P90 (m) | LSR@15 | Capture |
|---|---|---:|---:|---:|---:|---:|
| citya | test_01 | 161 | 92.597 | 158.008 | 6.83% | 6.21% |
| citya | test_02 | 219 | 191.829 | 252.432 | 3.65% | 4.11% |
| cityb | test_01 | 146 | 126.706 | 225.447 | 3.42% | 3.42% |
| cityb | test_02 | 171 | 129.851 | 210.178 | 2.92% | 4.09% |
