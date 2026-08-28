# BearingUAV ordered multi-route training for v36_byTeacher

The BearingUAV independent poses are materialized as waypoint polylines with
ordered reference points every approximately 4.5 metres. Each reference point
is attached to a nearby Bearing UAV view and written in the same sequential
`vi/` + `sensor_with_yaw.json` format consumed by the original v36 pipeline.

Split:

- `train_1` (city-A): 2,047 frames, 12 waypoints
- `train_2` (city-A): 2,047 frames, 12 waypoints
- `train_3` (city-A): 2,250 frames, 14 waypoints
- `val_1` (held-out city-B): 2,047 frames, 12 waypoints
- `test_1` (held-out city-C): 2,047 frames, 12 waypoints

Every route has a mean step of approximately 4.49 metres. Validation and test
use different UAV images and satellite cities from training.

The v36 forward 3x6 search, SoftMS, teacher feedback, GRU, polynomial motion,
Kalman update, loss, and native+stride-2 protocol are unchanged. Temporal
epochs rotate through one complete training route at a time; one three-epoch
cycle visits all three routes and resets recurrent/Kalman state at route
boundaries.

Measured on GPU 6:

- visual epoch: 194.7 seconds
- temporal epoch including full held-out validation: 165.4 seconds

```bash
CUDA_VISIBLE_DEVICES=6 ./run_train.sh \
  --visual-epochs 30 --temporal-epochs 90 --patience 15 --jitter-m 8
```
