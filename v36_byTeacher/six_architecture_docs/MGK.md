# MGK — MeanShift → GRU → Kalman

## Data flow

`Forward 3×6 cosine evidence → M → G → K → Final Position`

### M input / output

Input: forward 3×6 cosine logits and candidate coordinates.  
Output: MeanShift position `x_M` and visual variance `R_M`.

### G input / output

Input: `x_M`, `R_M`, temporal mean, first embedding difference, and previous hidden state.  
Output: refined current-frame position `x_G` and new hidden state. The variance is propagated to K because the PPT does not define a separate GRU uncertainty head.

### K input / output

Input: `x_G` as the current measurement, propagated measurement variance, previous Kalman state/covariance.  
Output: posterior `x_K`, which is the final position, plus posterior covariance.

## Expected behavior

Potentially very smooth because K is final, but the final Kalman step can attenuate useful GRU corrections and introduce temporal lag if measurement uncertainty is large.
