# V36 G vs K

This directory restores the original V36-byTeacher source snapshot associated with the historical 3-frame / four-head architecture and adds one isolated paired evaluation.

## Provenance

- Restored snapshot: `e732045cacc6d2bff152663e8b5966ee1b49b98b`
- Direct architecture parent: `fe3ff2329fad0d43bfbe0a8650bbb675fd19d339`
- Original model: `ThreeFrameRouteStateGRU`
- Heads: `correction_head`, `variance_head`, `motion_head`, `heading_head`
- Original flow is unchanged in the copied files.

The historical reported full-pipeline result was approximately MLE 3.482 m, Median 3.070 m, P90 6.620 m and LSR@3 49.208% on merged Routes B/C.

## Paired comparison

`compare_g_vs_k.py` uses one trained full V36 checkpoint and one forward pass per frame, then records:

- `G`: `output.measurement_se`, i.e. SoftMS anchor plus GRU correction, before Kalman fusion.
- `K_raw`: result immediately after `kf.update()`, before the historical final GT-progress cap.
- `K_original`: the historical reported V36 final after `cap_kalman_to_current_gt()`.

`K_raw` is included because comparing G directly with the historical capped final would otherwise mix the effect of Kalman fusion with the effect of the historical final cap.

## Outputs

All generated artifacts stay under `v36-GvsK/output/`:

- `GvsK_summary.json`
- `GvsK_comparison_table.csv`
- `route_B_GvsK_summary.json`
- `route_C_GvsK_summary.json`
- `route_B_GvsK_frames.csv`
- `route_C_GvsK_frames.csv`
- `run_gvsk.log`
- `checkpoints/`

The compact table reports MLE, Median, P90 and LSR metrics for G, K_raw and K_original, plus absolute/relative MLE improvement.

## Run

`run_gvsk.sh` defaults to a from-scratch reproduction using the historical settings: visual 30 epochs, temporal 60 epochs, patience 10, jitter parameter 8 m. It uses GPU 0 by default and can be changed with `GPU=<id>`.
