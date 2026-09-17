# v39_otherdata — Canonical Bearing-UAV DirectFinalMS workflow

This folder now keeps the current external-dataset workflow for the selected `v39_DirectFinalMS` method.

## Canonical method

```text
Bearing-UAV official route
-> local 6x6 candidate geometry / forward 3x6 scoring
-> posterior Weighted Centroid visual observation
-> 3-frame Context-GRU
-> Constant-Velocity motion
-> fixed-R constrained Kalman
-> one final 6x6 MeanShift (bandwidth 7 m)
-> final position
```

The model runtime is derived from `v39_DirectFinalMS/base_src` plus `v39_DirectFinalMS/patch_direct_finalms.py`; this folder is an external-data adapter, not a second independent tracker implementation.

## Current City B experiment

Training:

```text
train_01 only -> 60 temporal epochs
```

Held-out official Bearing-UAV navigation routes:

```text
test_01 -> wps36bc_50.json
test_02 -> wps36bc_51.json
```

Current committed results:

| Route | MLE | MedLE | P90 | LSR@15 |
|---|---:|---:|---:|---:|
| test_01 | 3.486 m | 2.880 m | 6.867 m | 100% |
| test_02 | 3.841 m | 3.385 m | 6.941 m | 100% |
| aggregate | 3.677 m | 3.221 m | 6.880 m | 100% |

These results use the selected controlled local-prior v39 protocol. MLE / MedLE / LSR@15 use the same metric definitions as Bearing-UAV, but this temporal controlled-prior protocol is not identical to Bearing-UAV's single-frame four-RST pose-regression benchmark. State that distinction explicitly in paper tables/text.

## Final paper figure definition

GT display:

```text
routes/<test route>/waypoints.json
-> sort by waypoint_order
-> connect only those sparse official waypoints
-> GREEN SOLID route polyline
```

Per-frame `gt_x/gt_y` observations are not joined for the display GT.

Prediction display:

```text
raw inference CSV final_x/final_y
-> frame-order RED SOLID polyline
-> no smoothing / interpolation / spline / resampling / denoising
```

Paper-specific figures:

```text
v39_otherdata/generated/cityb/v39_output_bearing_adapted/paper_figures_waypoint_gt/
  test_01_waypoint_gt_green.jpg
  test_02_waypoint_gt_green.jpg
  plot_source_audit.json
```

## Main commands

Full City B run:

```bash
CITY=cityb \
GPU=0 \
DATASET_ROOT=/yh/study/cvpr_data/Bearing_UAV_90K \
bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh
```

Plot only, without training or inference:

```bash
CITY=cityb bash v39_otherdata/rerender_bearing_waypoint_gt.sh
```

Preview cleanup:

```bash
CITY=cityb bash v39_otherdata/cleanup_v39_otherdata_latest.sh --dry-run
```

Apply cleanup after reviewing the preview:

```bash
CITY=cityb bash v39_otherdata/cleanup_v39_otherdata_latest.sh --apply
```

The cleanup script first moves legacy files to a timestamped backup directory outside this Git repository. It then leaves only the current DirectFinalMS workflow and latest selected-city results under `v39_otherdata/`.

## Canonical files kept by cleanup

```text
README.md
bearing_prepare.py
bearing_prepare_sequence_v3.py
bearing_multicity_routes.py
bearing_prepare_multicity.py
bearing_runner.py
bearing_runner_exact_v39.py
bearing_runner_multicity_v39.py
bearing_plot_final_vs_gt.py
bearing_paper_metrics.py
run_bearing_v39_sequence_fixed.sh
run_bearing_v39_directfinalms_official_routes.sh
rerender_bearing_waypoint_gt.sh
cleanup_v39_otherdata_latest.sh
generated/cityb/...
```

Do not use legacy plotting scripts or older `v39_output*` directories as the final paper result.
