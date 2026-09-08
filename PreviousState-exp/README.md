# PreviousState-exp

This top-level experiment folder is intentionally separate from `v36_GvsK/`.

All three variants use the current **Previous-State-Only** temporal architecture:

- Main GRU current input = temporal mean + first difference + second difference + Previous State.
- Previous State raw = `[previous v_s, previous v_e, previous heading residual, previous turn rate]`.
- Main GRU does **not** receive satellite context, response variance, visual innovation, or the previous Kalman final localization position.
- Previous GRU hidden state remains the GRU recurrent hidden input and is not concatenated into the 512-D current input.
- SoftMS, forward 3x6 first-stage local search, quadratic motion and learned route-coordinate Kalman remain enabled unless noted below.

## Variants

### 1. `mobileclip_current`
Preserves the already-trained ~3.9 m Previous-State-Only MobileCLIP2-S2 model. The runner copies the current source snapshot out of `v36_GvsK/previous_state_only/` into this folder and reuses the existing Previous-State-Only visual + temporal checkpoints for evaluation only.

### 2. `mobilenetv3_prevstate`
Same Previous-State-Only architecture, but changes the visual backbone to `mobilenet_v3_small`. It reuses the packaged forNX Route-A-only MobileNetV3 visual checkpoint and retrains only the temporal Previous-State-Only GRU on Route A, then evaluates Routes B/C.

### 3. `mobilenetv3_postkalman6x6`
Reuses **the exact MobileNetV3 temporal checkpoint from variant 2**. After the normal forward-3x6 -> GRU -> Kalman posterior is produced for a frame, the current UAV feature is matched again against a **full 6x6 = 36** satellite patch window centered on that Kalman posterior position. This second stage uses `FrozenVisualLocalizer.candidate_batch(..., grid_size=6)` and SoftMS; it does not apply the forward selector.

The post-Kalman 6x6 result is an **output-only refinement**: it becomes that frame's reported final localization, but it is not written back into the Kalman state. This isolates the effect of an extra local visual refinement and avoids counting the same frame's visual evidence twice in Kalman.

## Outputs

- `output/mobileclip_current/`
- `output/mobilenetv3_prevstate/`
- `output/mobilenetv3_postkalman6x6/`

Each result directory contains `robust_tracker_summary.json` and route CSVs.
