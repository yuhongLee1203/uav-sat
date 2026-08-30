# GKM: GRU → Kalman → Center MeanShift

## Frame-level flow
1. Initial local support is strict forward 3×6 (18 candidates) from the 6×6 local gallery.
2. Since M is not first, raw visual XY is the softmax-posterior weighted centroid and visual variance is computed from the same posterior.
3. **G**: GRU performs current-frame residual correction on the raw visual XY.
4. **K**: Kalman filters the GRU-corrected XY using visual covariance.
5. **M**: A centered 6×6 gallery is opened around the Kalman XY and Center MeanShift becomes the final localization output.

## GRU input / output
**Input:** raw current visual XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** current-frame `correction_xy`, `corrected_xy`, and new hidden state. No previous/final position and no future motion delta are used.

## Kalman input / output
**Input:** GRU-corrected XY, visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and covariance; posterior XY becomes the Center MeanShift search center.

## Expected behavior
The Kalman center should stabilize the final local visual search, but because MeanShift is last, an ambiguous visual posterior can pull the final estimate away from the smoother Kalman result. This makes GKM useful for testing whether the final visual refinement helps or hurts.
