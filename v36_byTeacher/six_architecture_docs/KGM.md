# KGM — Kalman → GRU → MeanShift

## Data flow

`Forward 3×6 raw cosine observation → K → G → centered M → Final Position`

### K input / output

Input: raw cosine centroid `x_raw`, raw posterior variance `R_raw`, previous Kalman state/covariance.  
Output: posterior `x_K` and covariance `P_K`.

### G input / output

Input: `x_K`, `diag(P_K)`, temporal mean, first embedding difference, previous hidden state.  
Output: refined current position `x_G` and new hidden state.

### M input / output

Input: `x_G` as center of the full centered 6×6 visual search.  
Output: MeanShift `x_M` as the final position and visual variance.

## Expected behavior

K and G cooperate to produce a stable center before visual re-localization. Risk: the last MeanShift can undo the GRU correction when the local visual response is multi-modal.
