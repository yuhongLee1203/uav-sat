# V39 SoftMS Ablation Suite

Current main pipeline:

`Forward 3x6 SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final SoftMS`

Run everything with:

```bash
GPU=0 bash v39_DirectFinalMS/run_softms_ablation_all.sh
```

Experiments:

1. Component removal: Full, w/o Forward 3x6, w/o GRU, w/o Kalman, w/o Final MS.
2. Temporal input: 1-frame, 2-frame, 3-frame. Each variant is freshly trained on Route A using the same training settings.
3. Final MeanShift window: 4x4, 5x5, 6x6, 7x7, 8x8. Front search remains Forward 3x6 SoftMS.

Every run is saved under a timestamped `softms_ablation_YYYYMMDD_HHMMSS/` directory. Per-run raw frame CSVs, logs, summaries, and newly trained temporal checkpoints are retained. The suite also writes `experiment_summary.csv`, `experiment_summary.json`, `paper_tables.md`, `audit_report.json`, and `run_manifest.json`.
