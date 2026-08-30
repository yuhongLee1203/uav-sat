# KMG — Kalman → MeanShift → GRU

## Data flow

`Forward 3×6 raw cosine observation → K → centered M → G → Final Position`

### K input / output

Input: raw cosine centroid `x_raw`, raw posterior variance `R_raw`, previous Kalman state/covariance.  
Output: stable posterior center `x_K` and covariance `P_K`.

### M input / output

Input: `x_K` as the center of a new full 6×6 visual search.  
Output: centered MeanShift position `x_M` and visual variance `R_M`.

### G input / output

Input: `x_M`, `R_M`, temporal mean, first embedding difference, previous hidden state.  
Output: refined current-frame `x_G`, which is the final position, and new hidden state.

## Expected behavior

This is the leading hypothesis before measurement. K first suppresses raw cosine noise, M gets a stable search center and restores visual precision, and G is last so it can correct remaining temporal/systematic error without being overwritten afterward.
