#!/usr/bin/env python3
"""City A no-GT causal validation search on GPUs 0, 5, and 6."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/yh/study/uav-sat")
V40 = ROOT / "v40"
BASE = V40 / "output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json"
VISUAL_SOURCE = V40 / "output/official96_fixed_citya_joint8_continue_20260925/checkpoints/citya/visual_retrieval_A_only.pt"
PREVIOUS_VISUAL_ROOT = VISUAL_SOURCE.parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else V40 / "output" / ("causal_4m_search_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
if OUT.exists():
    raise SystemExit(f"Refusing existing output: {OUT}")
OUT.mkdir(parents=True)
baseline = json.loads(BASE.read_text())
common = os.environ.copy()
common.update({key: str(value) for key, value in baseline["environment"].items()})
common.update({
    "UAVSAT_REFERENCE_PROTOCOL": "route_reference",
    "V40_GT_FRONT18_ORACLE": "0",
    "V40_GT_FRONT18_PATCH_ORACLE": "0",
    "V40_GT_FINAL_MS_ORACLE_CENTER": "0",
    "V40_GT_FINAL_MS_PATCH_ORACLE": "0",
    "V40_GT_JITTER_EVAL_M": "0",
    "V40_GT_JITTER_PATCH_SELECT": "0",
    "V40_VISUAL_CHECKPOINT_ROOT": str(PREVIOUS_VISUAL_ROOT),
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
})
jobs = [
    ("visual_continue", "0", [
        sys.executable, "-u", "v40/train_visual_backbone_gtloss.py",
        "--city", "citya",
        "--source-checkpoint", str(VISUAL_SOURCE),
        "--samples", "18000", "--epochs", "4", "--batch-size", "64",
        "--backbone-lr", "0.000005", "--head-lr", "0.00003",
        "--unfreeze-last-blocks", "2", "--target-sigma-m", "4",
        "--distance-loss-weight", "5", "--live-sat-topk", "16",
        "--live-sat-random", "16", "--output", str(OUT / "visual_continue"),
    ]),
    ("temporal_shifted", "5", [
        sys.executable, "-u", "v40/train_multicity_routes.py",
        "--train-cities", "citya", "--eval-cities", "citya",
        "--supervision", "gt_train", "--init", "random_temporal",
        "--epochs", "4", "--max-train-frames", "136", "--lr", "0.00001",
        "--teacher-acquisition-ratio", "0", "--teacher-center-ratio", "0",
        "--teacher-state-ratio", "0", "--dense-offbank-loss",
        "--acquisition-target-sigma-m", "4",
        "--extra-train-routes-dir", str(V40 / "data/official96_fixed_train_shifted_v3"),
        "--validate-every-epochs", "1", "--early-stop-patience", "3",
        "--skip-test", "--output", str(OUT / "temporal_shifted"),
    ]),
    ("temporal_grid", "6", [
        sys.executable, "-u", "v40/train_multicity_routes.py",
        "--train-cities", "citya", "--eval-cities", "citya",
        "--supervision", "gt_train", "--init", "random_temporal",
        "--epochs", "4", "--max-train-frames", "136", "--lr", "0.00001",
        "--teacher-acquisition-ratio", "0", "--teacher-center-ratio", "0",
        "--teacher-state-ratio", "0", "--dense-offbank-loss",
        "--acquisition-target-sigma-m", "4",
        "--extra-train-routes-dir", str(V40 / "data/official96_fixed_train_grid"),
        "--validate-every-epochs", "1", "--early-stop-patience", "3",
        "--skip-test", "--output", str(OUT / "temporal_grid"),
    ]),
]
(OUT / "plan.json").write_text(json.dumps({
    "baseline": str(BASE), "visual_source": str(VISUAL_SOURCE),
    "model_chain": "front18 -> GRU -> Kalman -> final MeanShift",
    "selection_split": "citya/val_01 only", "jobs": [
        {"name": name, "gpu": gpu, "command": command} for name, gpu, command in jobs],
}, indent=2) + "\n")
processes = []
for name, gpu, command in jobs:
    env = dict(common)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["V40_FEATURE_CACHE_ROOT"] = str(OUT / "cache" / name)
    folder = OUT / name
    folder.mkdir()
    handle = (folder / "console.log").open("w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    processes.append((name, gpu, process, handle))
    print(f"[START] {name} GPU {gpu} PID {process.pid}", flush=True)
statuses = {}
for name, gpu, process, handle in processes:
    statuses[name] = process.wait()
    handle.close()
    print(f"[EXIT] {name} GPU {gpu} status {statuses[name]}", flush=True)
(OUT / "status.json").write_text(json.dumps(statuses, indent=2) + "\n")
print(f"[RESULT ROOT] {OUT}", flush=True)
if any(statuses.values()):
    raise SystemExit(1)
