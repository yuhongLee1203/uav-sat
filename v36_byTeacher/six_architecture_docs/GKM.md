# GKM — GRU → Kalman → MeanShift

## Data flow

`Forward 3×6 raw cosine observation → G → K → centered M → Final Position`

### G input / output

Input: raw visual centroid/variance, temporal mean, first embedding difference, previous hidden state.  
Output: refined current position `x_G` and new hidden state.

### K input / output

Input: `x_G` as measurement, propagated measurement variance, previous Kalman state/covariance.  
Output: posterior position `x_K` and posterior covariance.

### M input / output

Input: `x_K` as the center of a new full 6×6 visual search.  
Output: MeanShift `x_M`, which is the final position, and its visual variance.

## Expected behavior

K provides a stable center for the final visual re-localization. Risk: because M is last, an ambiguous visual mode can overwrite the temporally consistent G/K estimate.
