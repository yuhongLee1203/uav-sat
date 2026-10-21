# Current V39 architecture

Selected main inference architecture after the same-checkpoint front-window comparison:

`5x5 local SAT geometry -> heading-guided forward 15 candidates -> front Soft MeanShift -> 3-frame GRU -> Constant Velocity -> Fixed-R Kalman -> Final 5x5 Soft MeanShift (BW=7m) -> Final Position`

The front-window comparison keeps the same visual and temporal checkpoints and fixes the final MeanShift window to 5x5. The selected 5x5/15 front achieved lower MLE and P90 on both Route B and Route C while also reducing end-to-end latency relative to the previous 6x6/18 front.

Previous 6x6/18 and Weighted-Centroid experiments are retained only for reproducibility and historical comparison.

Current comparison entry point:

```bash
GPU=0 bash v39_DirectFinalMS/run_compare_front_windows_eval.sh
```

Current full 5x5/15 ablation entry point:

```bash
GPU=0 bash v39_DirectFinalMS/run_softms_5x5_ablation_all.sh
```

The ablation suite keeps the front search fixed at 5x5 -> forward 15 and evaluates: (1) Full / w/o GRU / w/o Kalman / w/o Final MS, (2) 1/2/3-frame temporal input, and (3) final MeanShift window sensitivity from 4x4 to 8x8.
