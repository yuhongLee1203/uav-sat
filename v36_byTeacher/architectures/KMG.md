# KMG: Kalman → Center MeanShift → GRU

## Frame-level flow
1. Initial local support is strict forward 3×6 (18 candidates) from the 6×6 local gallery.
2. Since M is not first, raw visual XY is the softmax-posterior weighted centroid and visual variance is computed from that posterior.
3. **K**: Kalman first filters the raw visual XY using the visual covariance.
4. **M**: A centered 6×6 gallery is opened around the Kalman XY and Center MeanShift produces a refined visual XY.
5. **G**: GRU performs the last current-frame residual correction.
6. Final position is the GRU-corrected XY.

## GRU input / output
**Input:** centered MeanShift XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** current-frame `correction_xy`, `corrected_xy`, and new hidden state. No previous/final XY is given to G and no future polynomial displacement is produced.

## Kalman input / output
**Input:** raw visual XY, visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and covariance; posterior XY becomes the Center MeanShift search center.

## Expected behavior
G receives a coordinate already stabilized by K and visually re-centered by M, so its input quality can be high. However, because G is the final block, its residual is not smoothed afterward. It may rank well if the GRU learns small corrections, but MGK is still the safer expected winner.
