#!/usr/bin/env python3
"""Train one GT-loss-only City A temporal branch using corrected next-step labels."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/yh/study/uav-sat")
V40 = ROOT / "v40"
name, gpu, extra_dir, output = sys.argv[1:]
output = Path(output)
if output.exists():
    raise SystemExit(f"Refusing existing output: {output}")
output.mkdir(parents=True)
settings = json.loads((V40 / "output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json").read_text())
env = os.environ.copy()
env.update({k: str(v) for k, v in settings["environment"].items()})
env.update({
    "CUDA_VISIBLE_DEVICES": gpu,
    "V40_VISUAL_CHECKPOINT_ROOT": os.environ.get(
        "V40_CORRECTED_VISUAL_ROOT",
        str(V40 / "output/official96_fixed_citya_joint8_continue_20260925/checkpoints")),
    "V40_FEATURE_CACHE_ROOT": str(output / "cache"),
    "V40_GT_FRONT18_ORACLE": "0",
    "V40_GT_FRONT18_PATCH_ORACLE": "0",
    "V40_GT_FINAL_MS_ORACLE_CENTER": "0",
    "V40_GT_FINAL_MS_PATCH_ORACLE": "0",
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
})
command = [sys.executable, "-u", "v40/train_multicity_routes.py",
           "--train-cities", "citya", "--eval-cities", "citya",
           "--supervision", "gt_train", "--init", "random_temporal",
           "--epochs", "6", "--max-train-frames", "136", "--lr", "0.00001",
           "--teacher-acquisition-ratio", "0", "--teacher-center-ratio", "0",
           "--teacher-state-ratio", "0", "--dense-offbank-loss",
           "--acquisition-target-sigma-m", "4",
           "--measurement-loss-weight", os.environ.get("V40_CORRECTED_MEASUREMENT_WEIGHT", "2"),
           "--motion-loss-weight", os.environ.get("V40_CORRECTED_MOTION_WEIGHT", "2"),
           "--extra-train-routes-dir", str(Path(extra_dir).resolve()),
           "--validate-every-epochs", "1", "--early-stop-patience", "3",
           "--skip-test", "--output", str(output)]
(output / "command.json").write_text(json.dumps(
    {"name": name, "gpu": gpu, "command": command,
     "label_alignment": "next_step[t] = GT[t+1]-GT[t]",
     "validation_only": True}, indent=2) + "\n")
with (output / "console.log").open("w") as log:
    code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                          stderr=subprocess.STDOUT).returncode
print(f"[{name}] exit={code} output={output}", flush=True)
raise SystemExit(code)
