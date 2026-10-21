# V39 SoftMS Core Ablation Suite

Current main pipeline:

`Forward 3x6 SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final SoftMS`

Run everything with:

```bash
GPU=0 bash v39_DirectFinalMS/run_softms_ablation_all.sh
```

Core experiments:

1. Component removal: Full, w/o GRU, w/o Kalman, w/o Final MS.
2. Temporal input: 1-frame, 2-frame, 3-frame. Each variant is freshly trained on Route A using identical settings.
3. Final MeanShift window: 4x4, 5x5, 6x6, 7x7, 8x8. Front search remains Forward 3x6 SoftMS.

`w/o Forward 3x6 -> full 6x6` is intentionally excluded from the core ablation table. It changes several factors simultaneously: directional prior, candidate count (18 -> 36), satellite scoring cost, SoftMS input size, and backward/already-traversed search semantics. Therefore it is not a clean single-factor component ablation.

If directional-selection evidence is needed later, use a separate fixed-budget control (18 heading-guided candidates vs 18 heading-independent candidates) and report it as a candidate-selection study, not as the main component-removal table.

Every run is saved under a timestamped `softms_core_ablation_YYYYMMDD_HHMMSS/` directory. Per-run raw frame CSVs, logs, summaries, and freshly trained temporal checkpoints are retained. The suite also writes `experiment_summary.csv`, `experiment_summary.json`, `paper_tables.md`, `audit_report.json`, and `run_manifest.json`.
