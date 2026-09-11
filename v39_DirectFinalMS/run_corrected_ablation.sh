#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${ROOT}/base_src/visual_model.py"
RUNNER="${ROOT}/run.sh"

[[ -f "${BASE_MODEL}" ]] || { echo "ERROR: missing ${BASE_MODEL}" >&2; exit 2; }
[[ -f "${RUNNER}" ]] || { echo "ERROR: missing ${RUNNER}" >&2; exit 2; }

TMP_MODEL="$(mktemp)"
TMP_RUNNER="$(mktemp)"
cp "${BASE_MODEL}" "${TMP_MODEL}"
cp "${RUNNER}" "${TMP_RUNNER}"
restore() {
  cp "${TMP_MODEL}" "${BASE_MODEL}" || true
  cp "${TMP_RUNNER}" "${RUNNER}" || true
  rm -f "${TMP_MODEL}" "${TMP_RUNNER}"
}
trap restore EXIT INT TERM

# -----------------------------------------------------------------------------
# Critical correctness fix:
# In Constant-Velocity mode, training and inference must use the SAME motion
# equation. Previously the GRU's next_step loss used v + 0.5*a while the Kalman
# velocity-mode predictor discarded acceleration. The network could therefore
# learn v ~= 1.3 m/frame and let acceleration supply the rest of a ~3 m step;
# evaluation then threw that acceleration away and accumulated kilometre-scale
# progress error. Velocity mode now defines next_step from velocity only.
# -----------------------------------------------------------------------------
python3 - "${BASE_MODEL}" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = '''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
'''
new = '''        # Train/inference consistency for the selected motion model.
        # Constant Velocity MUST supervise and execute the same velocity-only
        # displacement. Only the quadratic ablation may use acceleration.
        motion_mode = str(getattr(config, "EXPERIMENT_MOTION", "quadratic"))
        if motion_mode == "velocity":
            base_forward = v_parallel.clamp(
                min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
            )
            base_cross = v_cross
        elif motion_mode == "none":
            # The no-motion mode is handled downstream by the Kalman ablation;
            # keep the GRU displacement interpretable and acceleration-free.
            base_forward = v_parallel.clamp(
                min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
            )
            base_cross = v_cross
        else:
            base_forward = (v_parallel + 0.5 * a_parallel).clamp(
                min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
            )
            base_cross = v_cross + 0.5 * a_cross
'''
if s.count(old) != 1:
    raise SystemExit(f"ERROR: velocity-consistency patch count={s.count(old)}")
s = s.replace(old, new, 1)
compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[FIX] Constant-Velocity GRU next_step is now velocity-only in training and inference")
PY

# -----------------------------------------------------------------------------
# Fail-fast guard: run.sh trains the canonical full model first. Before any
# parallel ablations are launched, inspect the fresh Route-A validation result.
# If it is still catastrophically wrong, stop immediately instead of consuming
# all GPUs on invalid experiments.
# -----------------------------------------------------------------------------
python3 - "${RUNNER}" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
needle = '''  [[ -f "${CANON_CKPT}" && ! -L "${CANON_CKPT}" ]] || { echo "ERROR: fresh canonical checkpoint missing" >&2; exit 20; }

  ( run_cfg 0 temporal_1frame'''
replacement = '''  [[ -f "${CANON_CKPT}" && ! -L "${CANON_CKPT}" ]] || { echo "ERROR: fresh canonical checkpoint missing" >&2; exit 20; }

  python3 - "${CANON_CKPT}" <<'PYCHECK'
import math, sys, torch
p = torch.load(sys.argv[1], map_location="cpu")
v = p.get("validation", {})
mle = float(v.get("mle", float("inf")))
speed = float(v.get("speed_mae", float("inf")))
progress = float(v.get("progress_mae", float("inf")))
print(f"[SANITY][full_model] Route-A val_mle={mle:.3f}m speed_mae={speed:.3f}m/frame progress_mae={progress:.3f}m")
if not math.isfinite(mle) or mle > 50.0 or not math.isfinite(progress) or progress > 100.0:
    raise SystemExit("SANITY FAILED: canonical full model diverged; aborting before all ablations")
PYCHECK

  ( run_cfg 0 temporal_1frame'''
if s.count(needle) != 1:
    raise SystemExit(f"ERROR: sanity-guard insertion count={s.count(needle)}")
s = s.replace(needle, replacement, 1)
p.write_text(s, encoding="utf-8")
print("[FIX] Added canonical validation fail-fast guard before parallel ablations")
PY

# Static verification before starting expensive jobs.
python3 - "${BASE_MODEL}" "${RUNNER}" <<'PY'
from pathlib import Path
import sys
model = Path(sys.argv[1]).read_text(encoding="utf-8")
runner = Path(sys.argv[2]).read_text(encoding="utf-8")
checks = {
    "velocity-only branch": 'if motion_mode == "velocity":' in model,
    "quadratic acceleration branch": 'v_parallel + 0.5 * a_parallel' in model,
    "weighted centroid": 'UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid' in runner,
    "scheduled reference": 'UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference' in runner,
    "3-frame canonical": 'run_cfg 0 full_model 3 fixed 0 1' in runner,
    "fail-fast guard": 'SANITY FAILED: canonical full model diverged' in runner,
}
bad = [k for k,v in checks.items() if not v]
for k,v in checks.items(): print(f"[STATIC] {k}: {'PASS' if v else 'FAIL'}")
if bad: raise SystemExit("STATIC AUDIT FAILED: " + ", ".join(bad))
PY

RUN_ALL_EXPERIMENTS=1 bash "${RUNNER}"
