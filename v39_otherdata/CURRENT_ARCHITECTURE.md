# Current Bearing V39 architecture

Current requested inference architecture:

`Forward 3x6 SAT candidates -> Soft MeanShift -> 3-frame GRU -> Constant Velocity -> Fixed-R Kalman -> Final 6x6 Soft MeanShift (BW=7m) -> Final Position`

The existing Weighted-Centroid four-city results are preserved under `v39_output_bearing_adapted/` for comparison.

The current no-retraining four-city evaluation entry point is:

```bash
GPU=0 DATASET_ROOT=/yh/study/cvpr_data/Bearing_UAV_90K bash v39_otherdata/run_bearing_v39_softms_eval_all4.sh
```

New outputs are written to `v39_output_bearing_softms_eval/` and `paper_all4_softms_eval_summary.json/csv`, so the old Weighted-Centroid results are not overwritten.
