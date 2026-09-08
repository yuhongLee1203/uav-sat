# v38_FinalMS

This folder contains the finalized MobileNetV3 localization variant whose reported final coordinate is produced by the second MeanShift stage.

## Final inference chain

MS1 (causal forward 3x6 Soft MeanShift) -> Previous-State GRU -> polynomial motion -> KF Predict/Update #1 -> temporary KF Update #2 -> full 6x6 prior-regularized MS2 -> Final.

KF Update #2 is current-frame-only. The persistent closed-loop navigation state for the next frame remains the KF#1 posterior. This prevents the second reference-guided correction from accumulating into route-progress drift.

MS2 searches all 36 candidates in a centered 6x6 SAT window. Its MeanShift logits combine (1) UAV-SAT visual likelihood, (2) a spatial prior around the temporary KF#2 posterior, and (3) a spatial prior around the predefined frame reference point. MS2 therefore remains a real visual MeanShift decoder while being prevented from jumping to a distant repetitive-field mode.

## Verified uploaded result

- Route B MLE: 1.5268 m
- Route C MLE: 1.3183 m
- Route B KF1 MAE: 4.7521 m -> temporary KF2: 2.1166 m -> MS2 final: 1.5268 m
- Route C KF1 MAE: 4.0711 m -> temporary KF2: 1.8054 m -> MS2 final: 1.3183 m
- Final output decoder: full 6x6 prior-regularized Soft MeanShift
- KF2 persistent feedback: disabled

The uploaded run is preserved under `reference_result/`.

## Run

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=5 UAVSAT_DEVICE=cuda:0 JITTER_M=8 bash v38_FinalMS/run.sh
```

`run.sh` uses the packaged MobileNetV3 visual checkpoint. If the previously trained temporal checkpoint is available, it reuses it; otherwise it trains the temporal model on Route A before evaluating Routes B and C.

## Protocol note

This experiment is a controlled predefined-reference-point protocol. KF#2 and the MS2 spatial prior use the current predefined frame reference coordinate. It should therefore be described transparently as reference-guided local refinement rather than fully autonomous unknown-position localization.
