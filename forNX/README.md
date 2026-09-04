# V36 Orin NX benchmark package

This is the standalone package for measuring the existing V36 backbone comparison on Orin NX.  It contains the real inference code, trained checkpoints, pretrained backbone files, and the actual Route B+C evaluation data.  It does not train anything and does not change any checkpoint.

## Included models

| Backbone | Trained V36 checkpoint | Included |
|---|---:|---:|
| MobileCLIP2-S2 | yes | yes |
| ResNet-18 | yes | yes |
| MobileNetV3-Small | yes | yes |
| VGG16 | yes | yes |
| ResNet-50 | no completed V36 checkpoint | no |

The package is about 3.5 GB.  The supplied V36 checkpoint protocol is SoftMS + 3-frame GRU + quadratic motion + learned external Kalman + forward 3x6 local search with the controlled smooth-jitter prior.  This is intentionally the same protocol as the completed backbone-comparison table; it is not a new 4x6 retraining.

## On Orin NX

Install a JetPack-compatible CUDA PyTorch and torchvision build, then install the Python packages:

```bash
cd forNX
python3 -m pip install -r requirements.txt
```

Run all four models:

```bash
bash scripts/run_benchmark.sh
```

Run one model only:

```bash
BACKBONES=resnet18 bash scripts/run_benchmark.sh
```

The output table is `runs/v36_backbone_comparison.md`.  Per-model timing logs and raw predictions are under `runs/v36_<backbone>/`.  Timing is from a prepared UAV tensor through the real backbone, visual retrieval, GRU, external Kalman, and final XY; feature-cache creation is excluded.

The package sets `TORCH_HOME` and `HF_HOME` to `pretrained_cache/`, so the included pretrained weights are used offline.  Do not omit `pretrained_cache/` when transferring this folder.
