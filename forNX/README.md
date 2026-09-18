# forNX — V39 DirectFinalMS deployment / latency package

This folder is the deployment copy of the selected `v39_DirectFinalMS` runtime. The old V36 four-backbone benchmark is no longer the main `forNX` workflow.

## Exact V39 pipeline

```text
MobileNetV3-Small UAV encoder
-> forward local SAT scoring
-> posterior Weighted Centroid visual observation
-> 3-frame GRU
-> Constant-Velocity motion
-> fixed-R external Kalman
-> exactly one final local Soft MeanShift (6x6, bandwidth 7 m)
-> final XY
```

`src/` is copied from `v39_DirectFinalMS/base_src/`; `patch_direct_finalms.py` is the same patch used by the formal V39 run. The benchmark creates a fresh runtime directory and applies that patch before evaluation, so it measures the same selected method instead of an older V36 implementation.

## 1. Prepare the portable folder on the workstation

From the repository root:

```bash
bash forNX/scripts/prepare_v39_package.sh
```

The preparation script does not train anything and does not delete prior results. It copies the existing visual/temporal checkpoints into:

```text
forNX/weights/v39_directfinalms/checkpoints/
  visual_retrieval_A_only.pt
  controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt
```

It also overlays the existing prepared `v36_GvsK/v36_training_data` into the Git-ignored `forNX/data/` directory so the entire `forNX/` folder is portable.

`forNX/data/`, `forNX/weights/`, and `forNX/pretrained_cache/` are intentionally Git-ignored because they are deployment assets, not source code.

## 2. Copy the complete `forNX/` directory to Jetson Xavier NX

Use your preferred SCP/rsync/USB method. Do not copy only `src/`; the NX needs the local weights, data, and pretrained cache too.

## 3. One command on the NX

```bash
cd ~/forNX && bash run_v39_nx_latency.sh
```

That one command validates CUDA/dependencies, reconstructs the V39 runtime, applies DirectFinalMS, runs Route B + Route C evaluation, warms up 30 frames per route, and prints the final latency as:

```text
V39_FULL_PIPELINE_MEAN_MS = ... ms
V39_FULL_PIPELINE_FPS     = ... FPS
```

It also prints Route-B/Route-C mean, median and P90 latency and saves a JSON result under `forNX/runs/v39_nx_latency_*/v39_nx_latency_result.json`.

## Timing definition

The measured per-frame interval is the V39 online inference path after the UAV image has already been transformed into a tensor. It includes the backbone/local visual retrieval, Weighted Centroid observation, 3-frame GRU, fixed-R Kalman and the final 6x6/BW7 MeanShift through final XY.

The reported number excludes image disk I/O, image preprocessing, model/checkpoint loading, and one-time satellite gallery / feature-cache construction. This makes the number suitable for device-to-device inference comparison.

Run the same `bash run_v39_nx_latency.sh` command on the 3090 package if you want a directly matched 3090 reference; do not mix it with older 5x5 timing logs.

## Jetson environment

Use the NVIDIA/JetPack-compatible PyTorch + torchvision build for your Jetson. Do not replace a working Jetson PyTorch installation with a generic desktop PyPI CUDA wheel. The Python runtime also needs `numpy`, `Pillow`, `open_clip_torch` (`import open_clip`) and its normal dependencies.

The launcher fails before inference if CUDA, source files, weights or deployment data are missing.
