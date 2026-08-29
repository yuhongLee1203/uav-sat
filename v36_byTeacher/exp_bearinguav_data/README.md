# BearingUAV 1000-frame multi-turn actual-pose training for v36_byTeacher

This experiment adapts BearingUAV samples to the sequential `vi/` +
`sensor_with_yaw.json` format used by v36 without assigning synthetic route
coordinates to the UAV images.

Important data semantics:

- BearingUAV-90K provides independent cross-view samples rather than a recorded
  video trajectory. The generated data is therefore a spatial pseudo-sequence.
- A planned polyline is used only to order nearby real samples.
- Every frame position label is the selected BearingUAV sample's own metadata
  position; source coordinates are never moved to an artificial route point.
- Each generated route contains straight legs connected by many 90-degree
  turns. Every traversed turn is written as a waypoint.
- `train_1`, `train_2`, and `train_3` each default to exactly 1000 frames.
- `val_1` and `test_1` also default to 1000 frames for balanced inspection.
- Effective displacement per pseudo-frame is deliberately different across
  routes and also varies within each route. The requested ordering is:
  `train_1 < val_1 < train_2 < test_1 < train_3`.
- The effective step rate is a spatial pseudo-motion rate, not a measured UAV
  velocity, because the original data is not a temporal video sequence.
- Repeated source images are not used inside one generated route.
- `generation_summary.json` reports frame count, turn-waypoint count, target and
  actual step statistics, corridor statistics, and zero image/label mismatch.

Current split:

- `train_1`: city-A, slow multi-L lawnmower route
- `train_2`: city-A, medium multi-L lawnmower route
- `train_3`: city-A, fast orthogonal spiral route
- `val_1`: held-out city-B, speed between training rates
- `test_1`: held-out city-C, speed between training rates

The v36 forward 3x6 search, SoftMS, GRU, polynomial motion, learned variance,
Kalman update, and temporal training code are otherwise reused.

Prepare only:

```bash
python3 prepare_bearinguav_routes.py \
  --corridor-m 14 \
  --min-step-m 0.8 \
  --max-step-m 13 \
  --hard-max-step-m 22 \
  --lookahead-m 32 \
  --target-train-frames 1000 \
  --target-eval-frames 1000 \
  --rebuild
```

Generate full-satellite route figures:

```bash
python3 plot_bearinguav_route.py --all
```

Inspect `generated_routes_3train_1val_1test/generation_summary.json` before
training. The key checks are: each train route has 1000 frames, each route has
multiple turn waypoints, image-label error is zero, step standard deviation is
non-zero, and actual mean step follows the requested interleaved speed order.

Full training on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 UAVSAT_RUN_TAG=bearinguav_multiturn_v4 ./run_train.sh \
  --visual-epochs 30 --temporal-epochs 90 --patience 15 --jitter-m 8
```
