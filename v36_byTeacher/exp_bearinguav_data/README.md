# BearingUAV same-scene irregular 2-train / 1-validation protocol

This experiment adapts BearingUAV samples to the sequential `vi/` +
`sensor_with_yaw.json` format used by v36 while preserving every selected UAV
image's own source position label.

The protocol uses **one satellite scene only: city-A**. There is no city-B or
city-C route in this experiment.

Split:

- `train_1`: city-A irregular sparse route, slower variable pseudo-motion.
- `train_2`: city-A irregular sparse route, faster variable pseudo-motion.
- `val_1`: city-A irregular sparse route, intermediate variable pseudo-motion.
- no separate test route in this data experiment.

The three paths are no longer stacked as upper/middle/lower horizontal bands.
Each path is a deterministic polyline made of long straight segments placed
across the full satellite image. Horizontal, vertical and diagonal legs are all
allowed, and different routes may cross. Every traversed planned segment
junction is written as a waypoint.

Each route is capped at **600 frames**. The generator requests up to 600 frames
and, if the route is sparse, can accept a shorter valid chain down to the
configured minimum (default 400) instead of failing solely because exactly 600
samples cannot be chained. Exact source images are not reused across routes.

The effective displacement target remains approximately:

`train_1 < val_1 < train_2`.

Because BearingUAV-90K is not a recorded temporal video, this is effective
spatial displacement per pseudo-frame, not measured physical UAV velocity.

Prepare only:

```bash
python3 prepare_bearinguav_routes.py \
  --corridor-m 18 \
  --min-step-m 0.8 \
  --max-step-m 13 \
  --hard-max-step-m 22 \
  --lookahead-m 34 \
  --target-train-frames 600 \
  --target-eval-frames 600 \
  --min-accepted-frames 400 \
  --rebuild
```

Visualize all three routes on the same city-A satellite image:

```bash
python3 plot_bearinguav_route.py --all
```

Generated summary:

`generated_routes_2train_1val_samecity/generation_summary.json`

Before training, confirm every route is at most 600 frames, contains multiple
segment-junction waypoints, has zero image/label mismatch, and has a non-fixed
step distribution.

Full training on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 \
UAVSAT_RUN_TAG=bearinguav_samecity_irregular_v6 \
./run_train.sh \
  --visual-epochs 30 \
  --temporal-epochs 90 \
  --patience 15 \
  --jitter-m 8
```
