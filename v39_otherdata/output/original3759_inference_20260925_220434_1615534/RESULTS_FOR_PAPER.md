# Original v13 4 m ABCD experiment: reportable results

## Evaluation scope

This experiment evaluates GT-assisted localization on the original v13 synthetic 4 m routes for City A/B/C/D. The held-out evaluation contains eight routes and 1,482 frames. At every test frame, the current GT position centers the local search and selects the acquisition hypothesis. These results therefore describe an oracle/GT-assisted setting, not no-GT deployment.

## Main result

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| Full, 3 frames, 6×6 final window | 41.90% | 73.75% | 100.00% | 3.759 |

## Component ablation

| Variant | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| w/o GRU | 32.66% | 65.92% | 99.87% | 4.275 |
| w/o Kalman | 41.30% | 73.82% | 100.00% | 3.763 |
| w/o Mean Shift | 15.45% | 33.40% | 98.04% | 6.785 |
| Full | 41.90% | 73.75% | 100.00% | 3.759 |

The recurrent and final Mean Shift components improve the measured localization results relative to their removal. The Kalman variant is effectively tied with Full on MLE: 3.763 m versus 3.759 m. Its LSR@5 is 0.07 percentage points higher in this replay (73.82% versus 73.75%). The paired route-level 95% interval for Full minus w/o Kalman MLE is [−0.0116, +0.0048] m, so this test does not establish a measurable accuracy gain from Kalman.

## Temporal input

| Input | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ |
|---|---:|---:|---:|---:|
| 1 frame | 39.81% | 72.47% | 100.00% | 3.883 |
| 2 frames | 39.14% | 71.79% | 100.00% | 3.921 |
| 3 frames | 41.90% | 73.75% | 100.00% | 3.759 |

Three frames has the lowest MLE of these separately trained temporal variants. Its MLE is lower than one frame by 0.124 m (route-level 95% interval [−0.1544, −0.0983]) and lower than two frames by 0.162 m ([−0.1870, −0.1342]).

## Final Mean Shift window

| Window | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | MLE (m) ↓ | Mean Shift time (ms) |
|---|---:|---:|---:|---:|---:|
| 4×4 | 37.65% | 67.14% | 99.93% | 4.168 | 4.043 |
| 5×5 | 41.90% | 73.75% | 100.00% | 3.763 | 4.998 |
| 6×6 (Full) | 41.90% | 73.75% | 100.00% | 3.759 | 7.516 |
| 7×7 | 42.04% | 73.95% | 100.00% | 3.738 | 9.499 |
| 8×8 | 42.04% | 74.09% | 100.00% | 3.738 | 10.269 |

The 7×7 and 8×8 settings have lower MLE than the original 6×6 setting on all eight routes, at higher measured latency. Keep the 6×6 row identified as the original Full configuration; report the window comparison as measured.

## Why HSR@15 and MHE are omitted

HSR@15 measures the share of predicted headings within 15° of heading ground truth; MHE is the mean absolute wrapped angular error. The model has a heading output, but this checkpoint was trained against the direction between neighboring points in the synthetic route. The dataset also records image yaw as yaw_deg; it is a different target. In this evaluation the two target directions differ by about 90° on average. The saved HSR@15/MHE values (39.61% / 36.87°) therefore measure agreement with the constructed route direction, not visual UAV heading. They should not appear as Bearing-UAV heading results.

It is acceptable to omit HSR/MHE when the paper is scoped to localization and says heading was not evaluated. Do not claim joint localization-and-heading or compare this result to a benchmark table that requires heading. A valid heading result needs a heading predictor trained for the chosen heading target and an evaluation using that same target.

Official four-tile Recall@1 was not produced. SR@20, SPL, and NE require a closed-loop navigation rollout, which this synthetic-route evaluation does not contain. These metrics are omitted.

## Result integrity

The historical MLE/LSR@5 proximity check passed. The replay does not exactly match every printed historical value: w/o Kalman LSR@5 is 73.82% here versus 73.68% in the cited result. The current-frame GT-assisted protocol and this mismatch should be disclosed with the table.
