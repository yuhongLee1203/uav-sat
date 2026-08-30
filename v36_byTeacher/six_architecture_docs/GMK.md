# GMK — GRU → MeanShift → Kalman

## Data flow

`Forward 3×6 raw cosine observation → G → centered M → K → Final Position`

### G input / output

Input: raw cosine posterior centroid `x_raw`, raw posterior variance `R_raw`, temporal mean, first embedding difference, previous hidden state.  
Output: refined current position `x_G` and new hidden state.

### M input / output

Input: `x_G` as the center of a new full 6×6 visual search.  
Output: centered MeanShift position `x_M` and new visual variance `R_M`.

### K input / output

Input: `x_M`, `R_M`, previous Kalman state and covariance.  
Output: posterior `x_K` as final position and posterior covariance.

## Expected behavior

G can move the search center before the second visual localization, while K smooths the resulting MeanShift measurement. Risk: an early GRU error may move the centered search into the wrong visual neighborhood.
