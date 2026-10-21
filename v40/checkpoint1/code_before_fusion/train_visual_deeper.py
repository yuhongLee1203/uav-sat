#!/usr/bin/env python3
"""GT-loss-only visual fine-tuning with four MobileNet blocks on one GPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/yh/study/uav-sat")
V40 = ROOT / "v40"
OUT = Path(sys.argv[1])
if OUT.exists():
    raise SystemExit(f"Refusing existing output: {OUT}")
OUT.mkdir(parents=True)
source = V40 / "output/official96_fixed_citya_joint8_continue_20260925/checkpoints/citya/visual_retrieval_A_only.pt"
settings = json.loads((V40 / "output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json").read_text())
env = os.environ.copy()
env.update({key: str(value) for key, value in settings["environment"].items()})
env.update({"CUDA_VISIBLE_DEVICES": "5", "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
command = [sys.executable, "-u", "v40/train_visual_backbone_gtloss.py",
           "--city", "citya", "--source-checkpoint", str(source),
           "--samples", "18000", "--epochs", "4", "--batch-size", "48",
           "--backbone-lr", "0.000003", "--head-lr", "0.00003",
           "--unfreeze-last-blocks", "4", "--target-sigma-m", "4",
           "--distance-loss-weight", "5", "--live-sat-topk", "16",
           "--live-sat-random", "16", "--output", str(OUT)]
(OUT / "command.json").write_text(json.dumps({"gpu": 5, "command": command}, indent=2)+"\n")
with (OUT / "console.log").open("w") as log:
    status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
print(f"[VISUAL DEEPER] exit={status} output={OUT}", flush=True)
raise SystemExit(status)
