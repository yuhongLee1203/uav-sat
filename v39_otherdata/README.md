# v39_otherdata — Bearing-UAV-90K adapter

This folder keeps the **v36 main architecture** (`config.py`, `visual_model.py`, `visual_localizer.py`, `robust_tracker.py`) and adds a dataset adapter for Bearing-UAV-90K.

## Experiment design

The first external-data experiment uses **City B = 36bc = Taipei** for both training and inference. This deliberately avoids mixing a city/terrain domain shift into the route-generalization test.

Routes:

- Train: `train_01`, `train_02`, `train_03`
- Held-out inference: `test_01`, `test_02`
- `route_A` is only a visual-training union alias required by the inherited v36 visual trainer. It concatenates the three training routes and is not an inference route.

All five routes are irregular polylines with alternating turns and unequal leg lengths, following the style of Bearing-UAV's released navigation waypoint JSONs rather than straight lines. Train/test UAV image samples are forced to be disjoint.

## Important scientific caveat

Bearing-UAV-90K contains independently sampled UAV observations, not a recorded frame-continuous flight. v39 therefore constructs **pseudo-flight sequences**: each planned polyline is sampled at 25 m intervals and the nearest unused UAV observation is selected, with a soft preference for a view heading compatible with the route tangent. The metadata heading is used only offline to make the sequence coherent; it is **not passed into the localization model**.

Global UAV pixel position follows the official Bearing-UAV conversion:

```text
global_x = block_x * 256 + 256 + x_norm * 256
global_y = block_y * 256 + 256 + y_norm * 256
```

The full RSI resolution is 0.25 m/px. The copied v36 local gallery keeps 320-pixel satellite crops and 32-pixel stride, so the candidate-center stride is 8 m on Bearing-UAV.

Inference uses the v36 `route_reference` protocol: the current frame's GT coordinate and GT-derived progress cap are disabled. The model pipeline remains UAV image -> MobileCLIP visual retrieval -> forward candidate selection / SoftMS -> 3-frame GRU -> polynomial motion prediction -> external Kalman.

## Two commands

Run from the repository root on the machine containing the dataset.

### 1. Prepare routes and render the full satellite route map

```bash
python3 v39_otherdata/bearing_prepare.py \
  --dataset-root /yh/study/cvpr_data/Bearing_UAV_90K \
  --city cityb
```

Preview output:

```text
v39_otherdata/generated/cityb/route_plan_full_satellite.jpg
```

The command also writes route manifests, route waypoints, split statistics, and the Bearing satellite metadata adapter.

### 2. Train and automatically run held-out inference

```bash
python3 v39_otherdata/bearing_runner.py \
  --dataset-root /yh/study/cvpr_data/Bearing_UAV_90K \
  --city cityb \
  --gpu 0 \
  --visual-epochs 30 \
  --epochs-per-route 20
```

This trains the visual head on the union of the three training routes, continues one temporal model through `train_01 -> train_02 -> train_03`, then runs inference on `test_01` and `test_02`.

Outputs are under:

```text
v39_otherdata/generated/cityb/v39_output/
```
