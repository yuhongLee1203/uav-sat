# BearingUAV same-scene exact-reference 2-train / 1-validation protocol

This experiment adapts BearingUAV samples to the sequential `vi/` +
`sensor_with_yaw.json` format used by v36 while preserving every selected UAV
image's own source position label.

The protocol uses **one satellite scene only: city-A**. There is no city-B or
city-C route in this experiment.

Split:

- `train_1`: irregular city-A route, slower variable pseudo-motion.
- `train_2`: irregular city-A route, faster variable pseudo-motion.
- `val_1`: irregular city-A route, intermediate variable pseudo-motion.
- no separate test route in this data experiment.

## Important geometry distinction

The route/reference line and selected BearingUAV samples are NOT the same
object.

- The route/reference geometry is an exact piecewise-straight polyline made of
  long horizontal, vertical, and diagonal legs. Different routes may cross.
- Every route waypoint lies exactly on that planned polyline.
- Real BearingUAV source samples are selected near the route and keep their own
  source positions. They are not snapped onto the route, so image/position
  labels stay correct.
- The shared v36 controlled protocol applies its normal deterministic smooth
  jitter at runtime. Jitter is a local-search prior perturbation; it is not
  baked into the source labels or route geometry.
- The plotting tool no longer connects independent source samples with a
  jagged line. Source samples are displayed as points only.

Each route is capped at **600 frames**. The generator requests up to 600 frames
and can accept a shorter valid chain down to the configured minimum (default
400) if the public-data sample geometry is sparse.

Because BearingUAV-90K is not a recorded temporal video, reported effective
speed is spatial displacement per pseudo-frame, not measured physical UAV
velocity.

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

Visualize all exact route lines on the same satellite image:

```bash
python3 plot_bearinguav_route.py --all --show
```

Visualize one route and preview the same 8 m smooth-jitter pattern used by v36:

```bash
python3 plot_bearinguav_route.py --route train_1 --show-jitter --jitter-m 8 --show
```

Generated summary:

`generated_routes_2train_1val_samecity/generation_summary.json`

Full training on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 \
UAVSAT_RUN_TAG=bearinguav_samecity_exactref_v7 \
./run_train.sh \
  --visual-epochs 30 \
  --temporal-epochs 90 \
  --patience 15 \
  --jitter-m 8
```
