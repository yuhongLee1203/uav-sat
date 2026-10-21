# Original 3.759 m replay: valid GT-assisted localization metrics

Historical proximity gate: **PASS**; exact printed-value match: **NO**.

City A/B/C/D, eight synthetic routes, 1,482 held-out images. Current-frame GT centers search and selects the acquisition hypothesis.

## Main architecture

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| Full (6x6) | 41.90% | 73.75% | 100.00% | 3.759 |

## Component removal

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| w/o GRU | 32.66% | 65.92% | 99.87% | 4.275 |
| w/o Kalman | 41.30% | 73.82% | 100.00% | 3.763 |
| w/o final Mean Shift | 15.45% | 33.40% | 98.04% | 6.785 |
| Full (6x6) | 41.90% | 73.75% | 100.00% | 3.759 |

## Temporal input

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| 1 frame | 39.81% | 72.47% | 100.00% | 3.883 |
| 2 frames | 39.14% | 71.79% | 100.00% | 3.921 |
| Full (6x6) | 41.90% | 73.75% | 100.00% | 3.759 |

## Final MeanShift window

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| 4x4 | 37.65% | 67.14% | 99.93% | 4.168 |
| 5x5 | 41.90% | 73.75% | 100.00% | 3.763 |
| Full (6x6) | 41.90% | 73.75% | 100.00% | 3.759 |
| 7x7 | 42.04% | 73.95% | 100.00% | 3.738 |
| 8x8 | 42.04% | 74.09% | 100.00% | 3.738 |

## Excluded metrics

HSR@15/MHE are not reported as Bearing-UAV heading metrics: the saved heading head was trained against direction between neighboring points on the synthetic route, not the UAV image's yaw_deg. Recall@1 is excluded because no official four-tile ranking was produced. SR@20/SPL/NE are excluded because there is no closed-loop navigation rollout.

The legacy heading table, if present, is retained as route_motion_heading_diagnostic.md for debugging only.

## Historical printed-value mismatches

- no_kalman: MLE 3.763/3.763; LSR@5 73.82%/73.68%
