# CityA causal localization progress (2026-09-26)

The current no-current-frame-GT validation best is **4.402 m MLE** on the 63-frame CityA `val_01` episode. The goal of 4.0 m has not yet been reached. This is a development result, not an independent test or official Table 6 score.

The reproducible code and best weights are in [`checkpoint1/`](checkpoint1/HISTORY.md), with a detailed v39→v40 history, the exact 18 patch versus 18×18 grid explanation, original run settings, and the new single-MLP three-frame experiment command. Earlier causal reports are indexed in `checkpoint1/CAUSAL_EXPERIMENT_INDEX.csv`; inferior causal artifacts were removed after archival.
