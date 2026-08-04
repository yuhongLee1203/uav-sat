# UAV--Satellite Temporal Tracking

This directory contains the current temporal diagnostic system.  It reuses the
archived P320/S32 6x6 HardMS visual localizer as a frozen measurement model and
applies a constant-velocity, robust Kalman-style temporal update.

## Layout

- `visual_model.py`: frozen MobileCLIP plus the archived UAV/SAT projection heads.
- `visual_localizer.py`: local 6x6 gallery lookup and Fixed HardMS measurement.
- `robust_tracker.py`: constant-velocity prediction and innovation-gated update.
- `render_results_video.py`: aspect-ratio-preserving UAV/orthomosaic videos.
- `outputs/temporal_prior_hardms/`: generated metrics, frame predictions, and videos.

## Run

The current tracker has no trainable module.  It evaluates the frozen visual
checkpoint and writes trajectories:

```bash
cd /yh/study/uav-sat
CUDA_VISIBLE_DEVICES=0 bash run_robust_tracker.sh
```

Render the resulting videos with:

```bash
cd /yh/study/uav-sat
bash render_results_video.sh
```

External dataset, orthomosaic, and archived visual-checkpoint paths are defined
in `config.py` and deliberately remain absolute because they are shared source
assets outside this project directory.
