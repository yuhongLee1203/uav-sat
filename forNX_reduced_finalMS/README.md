# forNX reduced-input + final MeanShift

This folder is an isolated copy of the compact reduced-input temporal architecture recovered from the historical `rewrite-autonomous-ms1-kf-gru-ms2` branch.

## GRU inputs

The recurrent model consumes only:

1. MS1 visual localization XY
2. temporal visual mean
3. first visual difference
4. previous motion `[speed, acceleration, sin(heading), cos(heading)]`

It does **not** use satellite context, prior/previous position, previous MS position, or previous delta position as GRU input branches. The previous hidden state remains the GRUCell recurrent state.

## Localization flow

`reference-point local-window center -> MS1 forward 3x6 -> Kalman -> centered 6x6 visual posterior -> MeanShift -> final XY`

The second MeanShift (MS2) is centered on the Kalman posterior. Its candidate score is the visual logit plus a smooth Gaussian spatial prior centered at the Kalman output. The default prior sigma is 12 m.

All outputs and checkpoints are local to this new folder, so the existing `v36_byTeacher`, `v37`, and other experiment folders are not overwritten.

## Run

```bash
cd /yh/study/uav-sat/forNX_reduced_finalMS
CUDA_VISIBLE_DEVICES=0 bash run_train_eval.sh all full
python3 compare_kf_vs_final_ms.py
```

Evaluation only after a checkpoint exists:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_train_eval.sh eval full
python3 compare_kf_vs_final_ms.py
```

The evaluator writes per-frame CSV files containing both `kalman_error_m` and `error_final_m`. The comparison helper reports whether the post-Kalman MeanShift improves or worsens MLE/P90/LSR@15 and writes `kf_vs_final_meanshift_comparison.csv` under the run output directory.
