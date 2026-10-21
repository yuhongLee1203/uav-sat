#!/usr/bin/env python3
"""Validation-only sweep of the actual final Mean Shift runtime settings."""
import json
import os
import subprocess
import sys
from pathlib import Path

V40 = Path(__file__).resolve().parent
ROOT = V40.parent
BASE = V40 / "output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json"
VISUAL = V40 / "output/causal_4m_search_20260926_a/visual_epoch2_snapshot/checkpoints"
TEMPORAL = V40 / "output/causal_4m_search_20260926_a/temporal_grid/best_temporal_interim.pt"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
base = json.loads(BASE.read_text())
env = os.environ.copy()
env.update({key: str(value) for key, value in base["environment"].items()})
env.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "V40_VISUAL_CHECKPOINT_ROOT": str(VISUAL),
    "V40_GT_FRONT18_ORACLE": "0",
    "V40_GT_FRONT18_PATCH_ORACLE": "0",
    "V40_GT_FINAL_MS_ORACLE_CENTER": "0",
    "V40_GT_FINAL_MS_PATCH_ORACLE": "0",
    "V40_FINAL_MS_LATTICE_CENTER": "visual_anchor",
    "V40_MEANSHIFT_SCORE_TAU": "0.5",
    "MS_KF_PRIOR_WEIGHT": "0",
    "MS_REFERENCE_PRIOR_WEIGHT": "0",
    "V40_FEATURE_CACHE_ROOT": str(OUT / "cache"),
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
})
variants = [(10, x) for x in (8, 15, 20, 25, 30, 35, 40)]
rows = []
for grid, bandwidth in variants:
    name = f"init{bandwidth}"
    trial = OUT / name
    trial.mkdir()
    trial_env = dict(env)
    trial_env["MS_GRID_SIZE"] = str(grid)
    trial_env["MS_BANDWIDTH_M"] = "10"
    trial_env["V40_INIT_FORWARD_SPEED_M_PER_FRAME"] = str(bandwidth)
    command = [sys.executable, "-u", "v40/evaluate_official96_validation.py",
               "--city", "citya", "--checkpoint", str(TEMPORAL),
               "--output", str(trial), "--tta-rotations", "0,1,2,3"]
    with (trial / "console.log").open("w") as log:
        status = subprocess.run(command, cwd=ROOT, env=trial_env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    if status:
        raise RuntimeError(f"{name} failed: {trial / 'console.log'}")
    report = json.loads((trial / "report.json").read_text())
    row = {"grid": grid, "initial_forward_speed_m_per_frame": bandwidth,
           "MLE_m": report["MLE_m"], "P90_m": report["P90_m"],
           "LSR@15_pct": report["LSR@15_pct"]}
    rows.append(row)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("[SWEEP]", json.dumps(row), flush=True)
print("[BEST]", json.dumps(min(rows, key=lambda row: row["MLE_m"])), flush=True)
