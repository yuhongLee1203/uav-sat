# MKG — MeanShift → Kalman → GRU

## Data flow

`Forward 3×6 cosine evidence → M → K → G → Final Position`

### M input / output

Input: cosine logits and candidate coordinates from the common forward 3×6 support.  
Output: MeanShift visual position `x_M` and axis-wise variance `R_M`.

### K input / output

Input: current measurement `x_M`, measurement covariance `R_M`, previous Kalman state `[x, y, vx, vy]`, and previous covariance `P`.  
Output: posterior current position `x_K` and posterior covariance `P_K`.

### G input / output

Input: `x_K`, `diag(P_K)`, temporal mean of the current/previous UAV embeddings, first embedding difference, and previous GRU hidden state.  
Output: refined current-frame position `x_G` and new hidden state. `x_G` is the final position.

## Important correction

The GRU does not predict a future polynomial displacement, and no final position is added to a GRU delta. The GRU only removes residual error from the current Kalman estimate.

## Expected behavior

Strong candidate. M gives K a clean visual mode; K removes frame-level jitter; G is last and can correct residual temporal bias.
