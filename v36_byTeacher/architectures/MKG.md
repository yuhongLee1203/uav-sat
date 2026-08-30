# MKG: Forward MeanShift → Kalman → GRU

## Frame-level flow
1. Initial 6×6 local gallery is opened around the predefined route reference point; strict forward 3×6 (18 candidates) is retained.
2. **M**: Forward MeanShift outputs current visual XY and visual variance `[σ²x, σ²y]`.
3. **K**: Kalman takes the MeanShift XY as measurement and the visual variance as `R_t`; its prior comes from its own previous posterior state/covariance.
4. **G**: GRU receives the Kalman current XY, visual variance, temporal UAV embedding mean, first embedding difference, and previous hidden state.
5. Final position is the GRU-corrected current-frame XY.

## GRU input / output
**Input:** current Kalman XY, `[σ²x, σ²y]`, temporal mean, first difference, previous hidden state. **Output:** `correction_xy`, `corrected_xy`, new hidden state. The GRU does not receive previous/final XY and does not output velocity, acceleration, heading, polynomial motion, or a future search-center delta.

## Kalman input / output
**Input:** MeanShift XY measurement, visual covariance, previous Kalman posterior state/covariance. **Output:** posterior `[x,y,vx,vy]` and posterior covariance; `[x,y]` is passed to G.

## Expected behavior
Strong visual mode first and temporal filtering second, but the final GRU residual can partially undo Kalman smoothness. It is a useful comparison but is not my first expected winner.
