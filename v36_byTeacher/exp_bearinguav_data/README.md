# BearingUAV variable-step actual-pose training for v36_byTeacher

This experiment adapts BearingUAV samples to the sequential `vi/` +
`sensor_with_yaw.json` format used by v36 without assigning synthetic route
coordinates to the UAV images.

Key rules of the corrected adapter:

- A planned polyline is used only to discover a spatially ordered sequence.
- Every frame position label is the selected BearingUAV UAV sample's own
  metadata position.
- Consecutive output frames use variable actual displacement; they are not
  forced to a constant 4.5 m step.
- Default accepted native-frame displacement is 1.5--5.0 m, so stride-2 is much
  closer to the motion scale used by the original v36 Route-A training.
- Repeated source images are not used inside one generated route.
- Waypoints are attached to real selected sample positions near route turns.
- `generation_summary.json` reports step mean/std/quantiles and explicitly
  reports zero image-label position mismatch by construction.

Current split:

- `train_1`, `train_2`, `train_3`: city-A
- `val_1`: held-out city-B
- `test_1`: held-out city-C

The v36 forward 3x6 search, SoftMS, GRU, polynomial motion, learned variance,
Kalman update, and native+stride-2 temporal training code are otherwise reused.

Prepare only:

```bash
python3 prepare_bearinguav_routes.py \
  --query-spacing-m 1.5 \
  --min-step-m 1.5 \
  --max-step-m 5.0 \
  --max-query-error-m 8.0 \
  --rebuild
```

Inspect `generated_routes_3train_1val_1test/generation_summary.json` before
training.  The important sanity checks are:

- `image_label_error_mean_m = 0.0`
- `image_label_error_p90_m = 0.0`
- `step_std_m > 0`
- `step_p10_m`, `step_p50_m`, and `step_p90_m` are not identical
- mean native step is in the same rough range as the original Route-A data

Full training:

```bash
CUDA_VISIBLE_DEVICES=5 ./run_train.sh \
  --visual-epochs 30 --temporal-epochs 90 --patience 15 --jitter-m 8
```
