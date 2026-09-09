# v41 Kalman-Anchored Adaptive Final MeanShift

## Why v40 was not enough

The v40 final stage already removed the predefined-reference prior and used only UAV-SAT visual likelihood plus a Kalman-centered Gaussian prior. However, the final MeanShift still slightly degraded the Kalman result:

- Route B: Kalman 4.7521 m -> v40 final 5.0655 m
- Route C: Kalman 4.0711 m -> v40 final 4.0870 m

The average final MeanShift displacement was only about 0.90 m / 0.82 m, so the main problem was not excessive displacement. The visual correction direction itself was not reliable enough in repetitive agricultural texture.

## v41 idea

Keep the external architecture simple:

```text
MS1 -> GRU -> Kalman Filter -> MS2 -> Final Position
```

MS2 uses no predefined frame-reference prior and there is no second Kalman update.

Internally, MS2 opens the full 6x6 satellite window around the Kalman posterior and builds a KDE over:

- 36 UAV-SAT visual candidate locations, and
- 1 explicit point located exactly at the Kalman posterior.

The mass of the Kalman anchor is adapted from the current visual posterior entropy and top-1/top-2 margin. Ambiguous visual responses keep a large mass at the Kalman mode, while a sharp and separated visual response is allowed to move MeanShift farther toward the visual mode.

The final coordinate is always the literal output of `soft_mean_shift`. There is no post-MS Kalman update, coordinate overwrite, or output clipping.

## Default parameters

```text
MS2_KF_SIGMA_M=5.0
MS2_KF_PRIOR_WEIGHT=1.0
MS2_ANCHOR_MIN_MASS=0.35
MS2_ANCHOR_MAX_MASS=0.90
MS2_ANCHOR_CONF_GAMMA=0.75
MS2_BANDWIDTH_M=4.0
```

These values are fixed before Route-B/Route-C evaluation. Do not tune them on the B/C test errors.

## Run

```bash
cd /yh/study/uav-sat && \
git fetch origin && \
git checkout v36-gvsk-original-comparison && \
git pull --ff-only origin v36-gvsk-original-comparison && \
CUDA_VISIBLE_DEVICES=5 \
UAVSAT_DEVICE=cuda:0 \
JITTER_M=8 \
bash v41_KalmanAnchoredMS/run.sh
```

Result:

```text
/yh/study/uav-sat/v41_KalmanAnchoredMS/output/robust_tracker_summary.json
```

Useful diagnostics are `Kalman_MAE_m`, `MLE_m`, `MS2_MeanShiftFromKalman_m`, `MS2_MeanAnchorMass`, and `MS2_MeanVisualConfidence`.
