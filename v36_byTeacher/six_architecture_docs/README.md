# Six-order MeanShift / GRU / Kalman ablation

This directory documents the six serial orders requested in the PPT: **MKG, MGK, GMK, GKM, KGM, KMG**.

## Common definitions

- **M — MeanShift localization**
  - If M is the first operator, it runs on the common **forward 3×6** visual candidate set.
  - If M is not first, it opens a **centered 6×6** search around the incoming stage position and runs MeanShift there.
  - Output: current-frame visual position and axis-wise visual variance.
- **G — GRU current-position refiner**
  - Input: incoming current-frame position, incoming variance, temporal mean of UAV embeddings, first embedding difference, previous hidden state.
  - Output: refined **current-frame** position and new hidden state.
  - It does **not** output future speed/acceleration/heading/polynomial motion for this ablation.
  - There is no `Final Position + GRU Delta` feedback path.
- **K — Kalman filter**
  - Input: incoming current-frame position as measurement, its axis-wise variance as measurement covariance, and Kalman's own previous state/covariance.
  - Output: posterior current-frame position and posterior covariance.
  - Kalman owns its constant-velocity recurrence; no GRU motion delta is injected into its state.

## Common visual evidence

Before the first M/G/K operator, all six variants use the same frozen UAV/SAT retrieval model and the same predefined-route local support. Cosine scores over the forward 3×6 candidates produce a raw posterior centroid and variance. This raw observation is the input when the first operator is G or K; if M is first, MeanShift is applied to the same forward 3×6 scores.

## Experimental fairness

The visual retrieval checkpoint is shared and frozen. Each order has its own GRU checkpoint. Training uses Route A; checkpoint selection uses Route C; final evaluation reports Route B and Route C separately. The predefined route reference point is used only to open/orient the common local candidate window in this controlled comparison.

## Initial hypothesis

Before running the experiments, **KMG** is the strongest candidate: Kalman first stabilizes the raw cosine observation, centered MeanShift then re-localizes around a temporally consistent center, and GRU is last so it can remove remaining systematic/temporal error without a later non-learned module overwriting its correction. **MKG** is the next candidate. This is only a hypothesis; the ranking must be decided from Route B/C MLE, P90 and LSR results.
