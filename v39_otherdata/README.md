# v39_otherdata — Canonical Bearing-UAV DirectFinalMS paper workflow

This folder contains the current external-dataset workflow for the selected `v39_DirectFinalMS` method.

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

The model runtime is derived from `v39_DirectFinalMS/base_src` plus `v39_DirectFinalMS/patch_direct_finalms.py`; this folder is the Bearing-UAV adapter, not a second independent tracker implementation.

## Paper main experiment: all four cities

The canonical paper run evaluates all four Bearing-UAV cities with the same code path and settings:

```text
citya -> two official test routes
cityb -> two official test routes
cityc -> two official test routes
cityd -> two official test routes
```

Each city performs its own Route-A training and held-out inference on its two official Bearing-UAV navigation routes. Results remain separated by city under `generated/<city>/`.

Run the complete four-city experiment from the repository root:

```bash
GPU=0 \
DATASET_ROOT=/yh/study/cvpr_data/Bearing_UAV_90K \
bash v39_otherdata/run_bearing_all4_cities.sh
```

This sequentially runs:

```text
citya -> cityb -> cityc -> cityd
```

and writes the final 8-route aggregate tables to:

```text
v39_otherdata/generated/paper_all4_summary.json
v39_otherdata/generated/paper_all4_summary.csv
```

The aggregate metrics are recomputed from the raw per-frame `error_final_m` values rather than averaging city-level medians or percentiles.

## Current committed City B reference result

| Route | MLE | MedLE | P90 | LSR@15 |
|---|---:|---:|---:|---:|
| test_01 | 3.486 m | 2.880 m | 6.867 m | 100% |
| test_02 | 3.841 m | 3.385 m | 6.941 m | 100% |
| aggregate | 3.677 m | 3.221 m | 6.880 m | 100% |

## Final paper figure definition

Displayed reference route:

```text
routes/<test route>/waypoints.json
-> sort by waypoint_order
-> connect only those sparse official waypoints
-> GREEN SOLID route polyline
```

The final prediction figure uses:

```text
raw inference CSV final_x/final_y
-> frame-order RED SOLID polyline
-> no smoothing / interpolation / spline / resampling / denoising
```

Every city produces:

```text
v39_otherdata/generated/<city>/v39_output_bearing_adapted/paper_figures_waypoint_gt/
  test_01_waypoint_gt_green.jpg
  test_02_waypoint_gt_green.jpg
  plot_source_audit.json
```

## Single-city run

To rerun only one city:

```bash
CITY=cityb \
GPU=0 \
DATASET_ROOT=/yh/study/cvpr_data/Bearing_UAV_90K \
bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh
```

Use `citya`, `cityb`, `cityc`, or `cityd`.

## Plot only

Without training or inference:

```bash
CITY=cityb bash v39_otherdata/rerender_bearing_waypoint_gt.sh
```

## Cleanup

The cleanup workflow is paper-safe and preserves all existing `citya/cityb/cityc/cityd` result packages plus `paper_all4_summary.*`.

Preview only:

```bash
bash v39_otherdata/cleanup_v39_otherdata_latest.sh --dry-run
```

Apply after reviewing the preview:

```bash
bash v39_otherdata/cleanup_v39_otherdata_latest.sh --apply
```

Legacy files are moved to a timestamped backup directory outside the Git repository rather than permanently deleted.

## Canonical files

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
run_bearing_all4_cities.sh
rerender_bearing_waypoint_gt.sh
cleanup_v39_otherdata_latest.sh
generated/citya/...
generated/cityb/...
generated/cityc/...
generated/cityd/...
generated/paper_all4_summary.json
generated/paper_all4_summary.csv
```

For scientific reporting, describe the information available to the inference procedure consistently with the actual implementation and distinguish this temporal local-reference experiment from Bearing-UAV's single-frame four-RST benchmark when making direct comparisons.
