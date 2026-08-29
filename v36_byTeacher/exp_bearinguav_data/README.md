# BearingUAV same-scene 2-train / 1-validation protocol

This experiment adapts BearingUAV samples to the sequential `vi/` +
`sensor_with_yaw.json` format used by v36 while preserving every selected UAV
image's own source position label.

The protocol now uses **one satellite scene only: city-A**. There is no city-B
or city-C route in this experiment.

Split:

- `train_1`: city-A upper band, 1000 frames, slower variable pseudo-motion.
- `train_2`: city-A lower band, 1000 frames, faster variable pseudo-motion.
- `val_1`: city-A middle band, 1000 frames, intermediate variable pseudo-motion.
- no separate test route in this data experiment.

All three routes are long chains of straight segments connected by 90-degree
turns. Every traversed planned turn is written as a waypoint. The spatial bands
are separated on the same satellite image so validation sees the same scene
type/map while following a different route region.

The requested effective displacement order is:

`train_1 < val_1 < train_2`.

Because BearingUAV-90K is not a recorded temporal video, this is effective
spatial displacement per pseudo-frame, not measured physical UAV velocity.

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

Visualize all three routes on the same city-A satellite image:

```bash
python3 plot_bearinguav_route.py --all
```

Generated summary:

`generated_routes_2train_1val_samecity/generation_summary.json`

Before training, confirm all three routes have 1000 frames, multiple turn
waypoints, zero image/label mismatch, non-fixed step distributions, and actual
mean step satisfying `train_1 < val_1 < train_2`.

Full training on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 \
UAVSAT_RUN_TAG=bearinguav_samecity_v5 \
./run_train.sh \
  --visual-epochs 30 \
  --temporal-epochs 90 \
  --patience 15 \
  --jitter-m 8
```
