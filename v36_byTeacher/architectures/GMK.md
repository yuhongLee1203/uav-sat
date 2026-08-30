# GMK: GRU → Center MeanShift → Kalman

## Frame-level flow
1. Initial 6×6 local gallery is opened around the predefined route reference point; strict forward 3×6 (18 candidates) is retained.
2. Because M is not first, the initial visual position is the softmax-posterior weighted centroid of those 18 candidates, with visual variance `[σ²x, σ²y]`.
3. **G**: GRU refines this raw current visual XY from temporal embedding information and uncertainty.
4. **M**: A centered 6×6 gallery is opened around the GRU-corrected XY and Center MeanShift produces a new visual XY.
5. **K**: Kalman takes the centered MeanShift XY as its measurement.
6. Final position is the Kalman posterior `[x,y]`.

## GRU input / output
**Input:** raw current visual XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** current-frame residual `correction_xy`, `corrected_xy`, new hidden state. It does not consume previous/final position and does not generate a future displacement.

## Kalman input / output
**Input:** centered MeanShift XY, corresponding visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and covariance.

## Expected behavior
The final Kalman is advantageous, but the GRU must work on a noisier pre-MeanShift observation. The later Center MeanShift can recover the local visual mode, so GMK is a plausible strong alternative to MGK.
