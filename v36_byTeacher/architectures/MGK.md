# MGK: Forward MeanShift → GRU → Kalman

## Frame-level flow
1. Initial 6×6 local gallery is opened around the predefined route reference point; strict forward 3×6 (18 candidates) is retained.
2. **M**: Forward MeanShift outputs current visual XY and visual variance `[σ²x, σ²y]`.
3. **G**: GRU refines the current MeanShift XY using visual uncertainty, temporal UAV embedding mean, first embedding difference, and previous hidden state.
4. **K**: Kalman uses the GRU-corrected current XY as measurement; measurement covariance remains derived from the visual posterior variance.
5. Final position is the Kalman posterior `[x,y]`.

## GRU input / output
**Input:** MeanShift current XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** current-frame residual `correction_xy`, `corrected_xy`, and new hidden state. No previous/final XY is fed to G, and no future motion or polynomial delta is produced.

## Kalman input / output
**Input:** GRU-corrected current XY, visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and posterior covariance.

## Expected behavior
This is the strongest theoretical candidate: MeanShift selects the local visual mode, GRU removes systematic/current-frame visual residual error, and Kalman is last so the final output keeps temporal smoothness and uncertainty-aware filtering. I expect MGK to be the most likely winner before running the experiment.
