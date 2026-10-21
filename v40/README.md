# v40: City A image-only training experiment

`run_citya_image_only.sh` trains three **unlabeled City A adaptations** of the
existing supervised shared-ABCD checkpoint, one each on GPU 0, 5 and 6. The fixed architecture is:
current UAV image through frozen MobileNet backbone, previous two cached UAV
features, forward-facing 3x6 (18) satellite patches, 3-frame GRU, external
Kalman filter, final MeanShift, XY output. Training resets recurrent/filter
state at the start of each route episode. `test_01` and `test_02` are held out.

The training loader reads only `frame_id` and `image_path`. It never uses
`x_m`, `y_m`, yaw or per-frame route reference. Route waypoints and satellite
map are *known priors*, not frame labels. The three objectives are visual
similarity pseudo-labeling; pseudo-labels plus image-derived temporal motion;
and confidence-gated pseudo-labeling/motion. They can be noisy or collapse.
The current frame always passes through the backbone; a sanitized feature
cache (no GT fields) is built once for the previous two frames. An existing
feature cache is reused if the image paths match. GT is read only during
held-out evaluation to compute metrics. The final MeanShift GT-center leak in
the copied v39 runtime is corrected for route-reference mode.

**Important limitation:** Both the visual retrieval checkpoint and the GRU
checkpoint were originally trained with GT. Therefore this experiment tests
whether further training works *without new GT labels*, not whether the entire
model can learn absolute location from unlabelled raw images from scratch.
Satellite/map correspondence and an origin prior are mathematically needed to
identify absolute XY; pure UAV images alone do not uniquely define map XY.
Kalman and MeanShift are non-differentiable decoders and are tested in the full
inference path, not optimized by gradient descent.

## Findings and recommended recheck

The old +/-60 m acquisition bank let a false visual match jump to a distant
route point; the old Kalman update then bypassed its bounded posterior and
snapped to that false measurement. v40 now keeps seven local hypotheses within
15 m of the causal estimate, drops the uncalibrated raw-peak bonus, and keeps
the bounded Kalman update. This does **not** use current-frame GT or the known
number of test frames to predict progress. The test GT is used only *after*
prediction for metrics. The old `capture=0%` in no-GT reports was also a
diagnostic bug; the corrected capture audit cannot affect selection.

Use this command to evaluate the existing supervised warm start with those
causal fixes, without retraining:

```bash
bash /yh/study/uav-sat/v40/run_citya_causal_recheck.sh
```

On the current prepared City A routes, that checkpoint yielded test_01 MLE
11.995 m and test_02 MLE 30.749 m. These are exploratory results: the test
routes were inspected while debugging and are **not** untouched paper-final
validation. A route-duration-only diagnostic baseline gives 12.069 m and
22.162 m; it uses full route duration and is not an image-only method.

The unlabeled adaptations remain experiments, not recommended checkpoints:
under the same causal search their MLEs were 47.258/88.304 m,
213.354/342.086 m and 262.662/351.565 m. Lower pseudo-label loss did not
mean better localization. `train_01` mean actual step is 5.36 m, while both
test routes are about 5.77 m. More importantly, over 97% of adjacent samples
have non-consecutive source indices and the median adjacent UAV yaw change is
roughly 85–93 degrees. These are spatially arranged pseudo-routes, not a
continuous video from which GRU motion can reliably be learned without labels.

## Shared multi-city pilot (new model checkpoint)

`run_bearinguav_multicity_pilot.sh` updates one shared temporal model and one
optimizer across selected city episodes; it resets GRU/Kalman at each route
boundary. Each city loads its own Bearing-UAV 90K UAV images, satellite map,
waypoints and gallery. It evaluates `test_01` and `test_02` separately for
each requested city. The model always uses current image -> backbone, previous
two cached image features -> front-18 -> GRU -> Kalman -> final MeanShift.
The test-time search center is causal and never reads frame GT. The script
does not use known test-route duration as a pace prior.

Two **different** training protocols are offered; do not mix their claims:

- `V40_SUPERVISION=image_only` (default): no frame GT in the new temporal
  training loss, but both visual and temporal models start from an older
  GT-supervised shared-ABCD checkpoint. This does not test from-scratch learning.
- `V40_SUPERVISION=gt_train`: frame GT is used in the temporal training loss,
  but never as a test-time model input. This is the scientifically ordinary
  supervised train / GT-free test setting.

For a short A+B pilot without new GT supervision:

```bash
V40_TRAIN_CITIES="citya cityb" V40_EVAL_CITIES="citya cityb" \
  V40_SUPERVISION=image_only V40_EPOCHS=2 V40_MAX_TRAIN_FRAMES=136 \
  bash /yh/study/uav-sat/v40/run_bearinguav_multicity_pilot.sh
```

After the pilot, if the protocol and result are acceptable, set both city
lists to `"citya cityb cityc cityd"` to run one shared ABCD temporal model.
The source images are in `/yh/study/cvpr_data/Bearing_UAV_90K`; the current
prepared custom routes use about 150–260 images per train/test episode, not
all 90K rows or the official paper's full merged-index split. Result files
explicitly record that distinction. A three-frame A+B image-only smoke test
passed, but it is not an accuracy claim.

Run from any directory:

```bash
bash /yh/study/uav-sat/v40/run_citya_image_only.sh
```

Outputs include per-design `train.log`, `temporal_image_only.pt`, `result.json`
and `summary.md`. For a quick local smoke run on GPU 0:

```bash
cd /yh/study/uav-sat
CUDA_VISIBLE_DEVICES=0 UAVSAT_BACKBONE=mobilenet_v3_small \
  UAVSAT_REFERENCE_PROTOCOL=route_reference MS_REFERENCE_PRIOR_WEIGHT=0 \
  python3 v40/image_only_citya.py --design visual_pseudo \
  --output v40/output/smoke/visual_pseudo --smoke
```
