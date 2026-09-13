# v39_otherdata — Bearing-UAV-90K adapter for canonical v39

This folder is an **external-dataset adapter** for the selected `v39_DirectFinalMS` model. The Bearing-specific code is limited to route/data preparation and the dataset adapter. The training runner rebuilds its model runtime from:

```text
v39_DirectFinalMS/base_src
+ v39_DirectFinalMS/patch_direct_finalms.py
+ the Context-GRU / velocity-fusion patch used by v39_DirectFinalMS/run.sh
```

This is intentional: `v39_otherdata` must not maintain an independent copied tracker that can silently drift away from the selected v39 architecture.

## Correct model/training definition

The corrected Bearing run uses the same selected v39 settings as the canonical runner:

```text
6x6 geometry -> causal forward 3x6 visual scoring
posterior Weighted Centroid visual observation
-> 3-frame Context GRU
-> Constant-Velocity motion
-> fixed-R external Kalman
-> one final 5x5 MeanShift (bandwidth 7 m)
-> final position
```

The Context GRU uses the canonical five projected inputs. The selected training objective is also preserved instead of switching to the older route-reference objective:

```text
LOSS_ACQUISITION = 0.0
LOSS_MEASUREMENT = 1.0
LOSS_NEXT_STEP = 3.0
LOSS_VELOCITY = 0.25
LOSS_HEADING = 1.25
LOSS_VARIANCE_NLL = 0.05
```

The default backbone matches the current canonical `v39_DirectFinalMS/run.sh`: `mobilenet_v3_small`.

Before training, `bearing_runner.py` performs a static audit of these settings and writes:

```text
v39_otherdata/generated/cityb/v39_output_corrected/v39_bearing_training_audit.json
```

If any core v39 setting or runtime patch is missing, the run stops instead of silently training a different architecture.

## Experiment design

The default external-data experiment uses **City B / 36bc** for both training and inference so route generalization is not mixed with a city/domain shift.

Routes:

- Train episodes: `train_01`, `train_02`, `train_03`
- Held-out inference: `test_01`, `test_02`
- `route_A` is a union of the three training routes used only by the inherited visual-head trainer.

All five paths are irregular polylines with alternating turns and unequal leg lengths. Their style is based on the released Bearing-UAV navigation waypoint files rather than a single straight or periodic path. Train/test UAV samples are forced to be disjoint.

## Important dataset caveat

Bearing-UAV-90K contains independently sampled UAV observations rather than one recorded frame-continuous flight. Therefore this adapter constructs **pseudo-flight sequences** by sampling each planned route and selecting nearby unused UAV observations. Metadata heading is used only offline to choose a more route-compatible observation; it is not passed to the localization network.

The old adapter sampled every 25 m and then enlarged the v39 speed/acceleration limits. That changed the model dynamics. The corrected default is **8 m per pseudo frame**, which keeps the data cadence within the original v39 motion envelope and leaves the canonical temporal caps unchanged.

Global UAV pixel position follows the Bearing-UAV metadata conversion used by this adapter:

```text
global_x = block_x * 256 + 256 + x_norm * 256
global_y = block_y * 256 + 256 + y_norm * 256
```

The city RSI is 0.25 m/px. With the unchanged v39 satellite stride of 32 px, neighboring SAT gallery centers are 8 m apart.

## Controlled-protocol caveat

The purpose of this corrected port is to answer: **does Bearing-UAV run through the same selected v39 model/training mechanism?** The answer is now yes.

However, the exact canonical `v39_DirectFinalMS` final refinement is a controlled experimental protocol: its canonical final-MS implementation uses the predefined current-frame reference as one of the final spatial priors. Therefore results from this exact reproduction must not be described as fully autonomous/no-reference Bearing navigation. A strict no-reference external benchmark would require a separate evaluation adaptation and would no longer be an exact reproduction of the selected controlled v39 protocol.

## Two commands

Run from the repository root.

### 1. Rebuild routes and render the full satellite route map

```bash
python3 v39_otherdata/bearing_prepare.py \
  --dataset-root /yh/study/cvpr_data/Bearing_UAV_90K \
  --city cityb \
  --step-m 8 \
  --max-sample-distance-m 15
```

Preview:

```text
v39_otherdata/generated/cityb/route_plan_full_satellite.jpg
```

This command also regenerates all route manifests, route waypoints, split statistics, `route_A`, and Bearing satellite metadata.

### 2. Train from scratch and automatically run held-out inference

```bash
python3 v39_otherdata/bearing_runner.py \
  --dataset-root /yh/study/cvpr_data/Bearing_UAV_90K \
  --city cityb \
  --gpu 0 \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 20 \
  --patience 5 \
  --jitter-m 8 \
  --step-m 8 \
  --reprepare
```

The runner trains the visual task head on `route_A`, then treats `train_01`, `train_02`, and `train_03` as separate temporal episodes. Hidden state resets at a route boundary, while the temporal model and optimizer continue across stages. With the default 20 epochs per route, the cumulative schedule is 20 -> 40 -> 60 epochs. It then evaluates the held-out `test_01` and `test_02` routes.

Corrected outputs are intentionally separated from the previous invalid run:

```text
v39_otherdata/generated/cityb/v39_output_corrected/
```

Key files:

```text
v39_bearing_training_audit.json
bearing_v39_summary.json
<test route>_*_frames.csv
```

Do not use the previous `generated/cityb/v39_output/` numbers as canonical-v39 Bearing results; that run used a different route-reference objective/configuration and did not contain the selected DirectFinalMS runtime.
