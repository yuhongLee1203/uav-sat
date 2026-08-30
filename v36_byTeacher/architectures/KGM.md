# KGM: Kalman → GRU → Center MeanShift

## Frame-level flow
1. Initial local support is strict forward 3×6 (18 candidates) from the 6×6 local gallery.
2. Since M is not first, raw visual XY is the softmax-posterior weighted centroid and visual variance is computed from the same posterior.
3. **K**: Kalman first filters the raw visual XY using the visual covariance.
4. **G**: GRU performs current-frame residual correction on the Kalman XY using temporal visual features and uncertainty.
5. **M**: A centered 6×6 gallery is opened around the GRU-corrected XY and Center MeanShift produces the final position.

## GRU input / output
**Input:** current Kalman XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** current-frame `correction_xy`, `corrected_xy`, and new hidden state. It does not receive previous/final localization XY and does not generate a polynomial or future-motion displacement.

## Kalman input / output
**Input:** raw visual XY, visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and covariance; posterior XY is passed to G.

## Expected behavior
Early Kalman filtering gives G a stable coordinate, but the final MeanShift can still overwrite some of that temporal smoothing if the local visual posterior is multimodal. KGM tests whether a final visual snap is beneficial after temporal correction.
