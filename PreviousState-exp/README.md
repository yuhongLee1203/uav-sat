# PreviousState-exp

This top-level experiment folder is intentionally separate from `v36_GvsK/`.

All variants use the current **Previous-State-Only** temporal architecture:

- Main GRU current input = temporal mean + first difference + second difference + Previous State.
- Previous State raw = `[previous v_s, previous v_e, previous heading residual, previous turn rate]`.
- Main GRU does **not** receive satellite context, response variance, visual innovation, or the previous Kalman final localization position.
- Previous GRU hidden state remains the GRU recurrent hidden input and is not concatenated into the 512-D current input.
- Forward local search, quadratic motion and learned route-coordinate Kalman remain enabled unless noted below.

## Variants

### 1. `mobileclip_current`
Preserves the already-trained ~3.9 m Previous-State-Only MobileCLIP2-S2 model. The runner copies the current source snapshot out of `v36_GvsK/previous_state_only/` into this folder and reuses the existing Previous-State-Only visual + temporal checkpoints for evaluation only.

### 2. `mobilenetv3_prevstate`
Same Previous-State-Only architecture, but changes the visual backbone to `mobilenet_v3_small`.

For this experiment, the **first-stage forward 3x6 visual decoder is Weighted Centroid instead of Soft MeanShift**. The 18 forward candidates are scored exactly as before; their softmax-normalized similarity weights are used to compute the weighted center coordinate. That weighted center is then used by the recurrent/Kalman pipeline as the current visual observation.

This variant reuses the packaged forNX Route-A-only MobileNetV3 visual checkpoint and retrains only the temporal Previous-State-Only GRU on Route A, then evaluates Routes B/C. The normal Kalman posterior is the reported final localization.

### 3. `mobilenetv3_postkalman6x6`
Uses the **same front-stage Weighted Centroid decoder** as variant 2 and reuses the exact MobileNetV3 temporal checkpoint from variant 2.

After the normal forward-3x6 Weighted-Centroid -> GRU -> Kalman posterior is produced for frame `t`, the second-stage local gallery is **not centered on the Kalman posterior**. Instead, the controlled current-frame reference point that defines the local search location is converted to the corresponding satellite search center. A **full centered 6x6 = 36** satellite patch window is opened around that reference-point-aligned center, and the current UAV feature is matched against those 36 patches.

The second-stage decoder is **Soft MeanShift**, not Weighted Centroid. Its MeanShift coordinate becomes the reported `END_MS` final localization for that frame. This second-stage result is output-only and is not fed back into the Kalman state or GRU state.

Thus the two MobileNetV3 variants are:

- `mobilenetv3_prevstate`: forward 3x6 **Weighted Centroid** -> GRU -> Kalman -> final.
- `mobilenetv3_postkalman6x6`: forward 3x6 **Weighted Centroid** -> GRU -> Kalman -> reference-point-centered full 6x6 -> **Soft MeanShift** -> END_MS final.

## Outputs

- `output/mobileclip_current/`
- `output/mobilenetv3_prevstate/`
- `output/mobilenetv3_postkalman6x6/`

Each result directory contains `robust_tracker_summary.json` and route CSVs.
