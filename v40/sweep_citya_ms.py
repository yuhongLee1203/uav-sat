#!/usr/bin/env python3
"""Validation-only final MeanShift sweep with frozen 18patch-GRU-Kalman model."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/yh/study/uav-sat")
V40 = ROOT / "v40"
BASE = V40 / "output/official96_fixed_citya_val_joint8_blend0_20260925/run_settings.json"
CKPT = V40 / "output/causal_4m_search_20260926_a/temporal_grid/best_temporal_interim.pt"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else V40 / "output/causal_4m_ms_sweep_20260926_a"
if OUT.exists():
    raise SystemExit(f"Refusing existing output: {OUT}")
OUT.mkdir(parents=True)
base = json.loads(BASE.read_text())
env = os.environ.copy()
env.update({k: str(v) for k, v in base["environment"].items()})
env["CUDA_VISIBLE_DEVICES"] = "5"
env["V40_VISUAL_CHECKPOINT_ROOT"] = str(V40 / "output/official96_fixed_citya_joint8_continue_20260925/checkpoints")
env["V40_GT_FRONT18_ORACLE"] = "0"
env["V40_GT_FRONT18_PATCH_ORACLE"] = "0"
env["V40_GT_FINAL_MS_ORACLE_CENTER"] = "0"
env["V40_GT_FINAL_MS_PATCH_ORACLE"] = "0"
env["OMP_NUM_THREADS"] = "2"
env["OPENBLAS_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
variants = [
    ("baseline", "blend", 6, 4, .1),
    ("grid8", "blend", 8, 4, .1),
    ("grid10", "blend", 10, 4, .1),
    ("anchor6", "visual_anchor", 6, 4, .1),
    ("anchor8", "visual_anchor", 8, 4, .1),
    ("bw2", "blend", 6, 2, .1),
    ("bw6", "blend", 6, 6, .1),
    ("grid8_bw2", "blend", 8, 2, .1),
    ("grid8_bw6", "blend", 8, 6, .1),
    ("tau05", "blend", 6, 4, .5),
    ("tau1", "blend", 6, 4, 1),
    ("anchor_tau05", "visual_anchor", 6, 4, .5),
]
results = []
for name, center, grid, bandwidth, tau in variants:
    trial = OUT / name
    trial.mkdir()
    setting = dict(env)
    setting.update({
        "V40_FINAL_MS_LATTICE_CENTER": center,
        "MS_GRID_SIZE": str(grid),
        "V40_MEANSHIFT_BANDWIDTH_M": str(bandwidth),
        "V40_MEANSHIFT_SCORE_TAU": str(tau),
        "V40_FEATURE_CACHE_ROOT": str(OUT / "cache"),
    })
    command = [sys.executable, "-u", "v40/evaluate_official96_validation.py",
               "--city", "citya", "--checkpoint", str(CKPT),
               "--output", str(trial)]
    with (trial / "console.log").open("w") as handle:
        status = subprocess.run(command, cwd=ROOT, env=setting, stdout=handle,
                                stderr=subprocess.STDOUT).returncode
    if status:
        print(f"[FAIL] {name}: {trial / 'console.log'}", flush=True)
        results.append({"name": name, "status": status})
        continue
    report = json.loads((trial / "report.json").read_text())
    row = {"name": name, "MLE_m": report["MLE_m"], "P90_m": report["P90_m"],
           "LSR@15_pct": report["LSR@15_pct"], "grid": grid,
           "center": center, "bandwidth_m": bandwidth, "tau": tau}
    results.append(row)
    print("[SWEEP]", json.dumps(row), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("[BEST]", min((r for r in results if "MLE_m" in r),
                    key=lambda r: r["MLE_m"]), flush=True)
