# Six-architecture M/G/K ablation

This branch implements the six architectures drawn on PPT pages 2–4: `MKG`, `MGK`, `GMK`, `GKM`, `KGM`, `KMG`.

## Why PPT page 1 / old v36 is different

The old flow is effectively `MS1 -> Kalman` in parallel with a motion-GRU, then `MS2 -> final_position`, followed by `final_position + GRU polynomial delta -> next search center`. That makes the GRU a future-motion generator and couples the next visual search to the previous final localization. The old Kalman implementation also overwrites its positional prior with `previous_final_xy` and inserts the previous GRU delta into velocity before update.

The six new flows remove that feedback semantics. GRU is now a **current-frame position refiner**, Kalman is a **self-contained recursive estimator**, and MeanShift is a **visual localization/refinement block** whose location in the pipeline changes according to the architecture order.

## Common GRU semantics

Input: incoming current-stage XY, visual variance `[sigma_x^2, sigma_y^2]`, temporal mean of projected UAV embeddings, first embedding difference, previous hidden state.

Output: current-frame `correction_xy`, current-frame `corrected_xy`, new hidden state. There is no heading head, motion head, polynomial, previous-final-position input, or future search-center delta.

## Common Kalman semantics

Input: incoming current-stage XY as measurement, visual posterior variance as measurement covariance, and the Kalman filter's own predicted state/covariance from its previous posterior.

Output: posterior `[x, y, vx, vy]` and covariance. The filter is not reset to the previous final localization and does not consume a GRU delta.

## Fair experiment protocol

Route A trains each architecture's GRU. Route B and Route C are evaluation-only. The initial local search support is identical across all six variants: a 6x6 lattice is opened around the predefined route reference point and only the strict forward 3x6 half is kept. When M is first it performs Forward MeanShift there; when M appears later it performs a centered 6x6 MeanShift around the incoming stage coordinate.

## Expected ranking before experiment

`MGK` is the leading hypothesis because visual mode selection happens first, the GRU then removes systematic current-frame residual error, and Kalman is last to preserve temporal smoothness. `GMK` is the next strongest theoretical candidate. The actual ranking must be determined by the Route-B/Route-C metrics.
