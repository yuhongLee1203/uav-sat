# Current V39 architecture

The canonical V39 experiment is now pinned back to the 2026-09-13 pipeline and only the **front decoder** is changed from Weighted Centroid to Soft MeanShift.

Main chain:

`6x6 local SAT geometry -> heading-guided forward 3x6 = 18 candidates -> front Soft MeanShift -> 3-frame GRU -> Polynomial / Constant-Velocity motion -> Fixed-R Kalman -> post-Kalman MeanShift -> Final Position`

This reset intentionally does **not** use split-gate / dual-gate / GRU gate calibration patches. Historical experimental patch files are kept only for reproducibility; the canonical runner below does not load them.

The base model/training source remains the same `v39_DirectFinalMS/base_src` tree used by the 2026-09-13 V39 run. The only front-end change is:

`UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid` -> `UAVSAT_EXPERIMENT_ANCHOR=softms`

The forward-search geometry stays at the original 6x6 construction with heading-selected forward 3x6 scoring (18 candidates). No 5x5/15 front-search patch is applied by the canonical runner.

## One-command experiment entry point

```bash
cd /yh/study/uav-sat
bash v39_DirectFinalMS/run_0913_softms18_everything.sh
```

That command runs:

- the main Route-A-trained / Route-B+C-evaluated architecture;
- progressive module ablations and 1/2/3-frame temporal ablations;
- MeanShift window/runtime experiments from the 09/13 V39 runner;
- Bearing-UAV other-data runs for `citya`, `cityb`, `cityc`, and `cityd`;
- GPU 0/5/6 parallel scheduling;
- a final integrity audit that reports whether the complete model is actually best on the measured metrics.

The audit never edits, suppresses, or degrades ablation baselines to force a preferred ranking. If an ablation beats Full, it is reported as such and the run is marked `FULL_BEST_CHECK=FAIL` rather than altering the experiment.
