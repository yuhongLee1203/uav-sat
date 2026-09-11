#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${ROOT}/base_src/visual_model.py"
BASE_TRACKER="${ROOT}/base_src/robust_tracker.py"
BASE_CONFIG="${ROOT}/base_src/config.py"
RUNNER="${ROOT}/run.sh"

for f in "${BASE_MODEL}" "${BASE_TRACKER}" "${BASE_CONFIG}" "${RUNNER}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

TMP_MODEL="$(mktemp)"
TMP_TRACKER="$(mktemp)"
TMP_CONFIG="$(mktemp)"
TMP_RUNNER="$(mktemp)"
cp "${BASE_MODEL}" "${TMP_MODEL}"
cp "${BASE_TRACKER}" "${TMP_TRACKER}"
cp "${BASE_CONFIG}" "${TMP_CONFIG}"
cp "${RUNNER}" "${TMP_RUNNER}"
restore() {
  cp "${TMP_MODEL}" "${BASE_MODEL}" || true
  cp "${TMP_TRACKER}" "${BASE_TRACKER}" || true
  cp "${TMP_CONFIG}" "${BASE_CONFIG}" || true
  cp "${TMP_RUNNER}" "${RUNNER}" || true
  rm -f "${TMP_MODEL}" "${TMP_TRACKER}" "${TMP_CONFIG}" "${TMP_RUNNER}"
}
trap restore EXIT INT TERM

# =============================================================================================
# 1) Constant-Velocity correctness: the GRU training target and Kalman inference must use the
#    exact same motion equation. Velocity mode therefore uses v only; acceleration is retained
#    only for the explicit quadratic ablation and never contributes to the selected model.
# =============================================================================================
python3 - "${BASE_MODEL}" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
old='''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
'''
new='''        motion_mode = str(getattr(config, "EXPERIMENT_MOTION", "quadratic"))
        if motion_mode in {"velocity", "none"}:
            # Selected Constant-Velocity model: training and inference both use
            # exactly the velocity output. Acceleration cannot silently supply
            # displacement during training and then disappear at inference.
            base_forward = v_parallel.clamp(
                min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
            )
            base_cross = v_cross
        else:
            # Explicit Velocity+Acceleration ablation only.
            base_forward = (v_parallel + 0.5 * a_parallel).clamp(
                min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
            )
            base_cross = v_cross + 0.5 * a_cross
'''
if s.count(old)!=1: raise SystemExit(f'ERROR: velocity-consistency patch count={s.count(old)}')
s=s.replace(old,new,1)
compile(s,str(p),'exec'); p.write_text(s,encoding='utf-8')
print('[FIX] velocity-mode next_step is velocity-only')
PY

# =============================================================================================
# 2) Stable temporal optimisation. A 0.1-0.2 m/frame speed bias looks small to a per-frame loss
#    but accumulates to tens/hundreds of metres in closed loop. Use a smaller learning rate,
#    stronger direct velocity/step supervision, and an explicit TBPTT cumulative-progress loss.
# =============================================================================================
python3 - "${BASE_CONFIG}" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
repls=[
('TEMPORAL_LR = 2e-4','TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "1e-4"))'),
('''if FRAME_REFERENCE_SUPERVISION:
    LOSS_MEASUREMENT = 3.0
    LOSS_NEXT_STEP = 3.0
    LOSS_VELOCITY = 0.10
''','''if FRAME_REFERENCE_SUPERVISION:
    LOSS_MEASUREMENT = 3.0
    LOSS_NEXT_STEP = float(os.environ.get("UAVSAT_LOSS_NEXT_STEP", "4.0"))
    LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "4.0"))
'''),
('''LOSS_PROGRESS = 0.0
''','''LOSS_PROGRESS = 0.0
# Sequence-level pace consistency inside each TBPTT chunk. This directly
# penalises systematic per-frame speed bias before it can accumulate into
# large closed-loop progress drift.
LOSS_CUMULATIVE_PROGRESS = float(
    os.environ.get("UAVSAT_LOSS_CUMULATIVE_PROGRESS", "0.5")
)
''')]
for old,new in repls:
    if s.count(old)!=1: raise SystemExit(f'ERROR: config patch count={s.count(old)} for {old[:40]!r}')
    s=s.replace(old,new,1)
compile(s,str(p),'exec'); p.write_text(s,encoding='utf-8')
print('[FIX] temporal LR/loss configuration patched')
PY

python3 - "${BASE_TRACKER}" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')

# Initialise absolute forward speed from Route-A TRAINING labels only. The old
# hard-coded 0.75 m/frame bias guaranteed a huge initial closed-loop lag even
# though Route A moves about 3 m/frame. This is an initialization, not an
# inference input and it never reads Route B/C.
old='''    gt_state = build_gt_route_state(cache, route)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
'''
new='''    gt_state = build_gt_route_state(cache, route)
    if start_epoch == 1 and str(getattr(config, "EXPERIMENT_MOTION", "quadratic")) == "velocity":
        train_speed = float(np.mean(gt_state["velocity"][train_start:train_end, 0]))
        train_speed = float(np.clip(train_speed, 0.25, float(config.MAX_FORWARD_SPEED_M_PER_FRAME) - 1e-3))
        with torch.no_grad():
            last = model.motion_head[-1]
            last.weight.zero_()
            last.bias.zero_()
            last.bias[0] = math.log(math.expm1(train_speed))
        print(
            "Constant-Velocity initialization from Route-A train mean: %.3f m/frame" % train_speed,
            flush=True,
        )
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
'''
if s.count(old)!=1: raise SystemExit(f'ERROR: speed-init patch count={s.count(old)}')
s=s.replace(old,new,1)

# Track differentiable cumulative displacement inside each TBPTT chunk.
old='''        losses = []
        component_rows = []

        optimizer.zero_grad(set_to_none=True)
'''
new='''        losses = []
        component_rows = []
        chunk_predicted_progress = None
        chunk_target_progress = 0.0

        optimizer.zero_grad(set_to_none=True)
'''
if s.count(old)!=1: raise SystemExit(f'ERROR: chunk-init patch count={s.count(old)}')
s=s.replace(old,new,1)

old='''            step_loss, components = temporal_loss(
                output=output,
                observation=obs,
                target_se=gt_state["se"][index],
                target_velocity=target_velocity,
                target_acceleration=target_acceleration,
                target_step=target_step,
                target_heading_residual=target_heading,
                target_turn_rate=target_turn,
                current_reference_se=current_reference_se,
                next_reference_se=next_reference_se,
            )
            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss
'''
new='''            step_loss, components = temporal_loss(
                output=output,
                observation=obs,
                target_se=gt_state["se"][index],
                target_velocity=target_velocity,
                target_acceleration=target_acceleration,
                target_step=target_step,
                target_heading_residual=target_heading,
                target_turn_rate=target_turn,
                current_reference_se=current_reference_se,
                next_reference_se=next_reference_se,
            )
            # TBPTT cumulative-progress consistency. A constant +0.2 m/frame
            # bias becomes +6.4 m over 32 frames and is therefore strongly
            # visible to this loss instead of looking harmless frame-by-frame.
            current_predicted_progress = output.next_step_se[:, 0]
            if chunk_predicted_progress is None:
                chunk_predicted_progress = current_predicted_progress
                chunk_target_progress = float(target_step[0])
            else:
                chunk_predicted_progress = chunk_predicted_progress + current_predicted_progress
                chunk_target_progress += float(target_step[0])
            cumulative_target = torch.tensor(
                [chunk_target_progress],
                dtype=chunk_predicted_progress.dtype,
                device=chunk_predicted_progress.device,
            )
            cumulative_progress_loss = F.smooth_l1_loss(
                chunk_predicted_progress, cumulative_target
            )
            step_loss = step_loss + float(config.LOSS_CUMULATIVE_PROGRESS) * cumulative_progress_loss
            components["cumulative_progress"] = float(cumulative_progress_loss.detach().cpu())
            chunk_loss = step_loss if chunk_loss is None else chunk_loss + step_loss
'''
if s.count(old)!=1: raise SystemExit(f'ERROR: cumulative-loss patch count={s.count(old)}')
s=s.replace(old,new,1)

old='''                chunk_loss = None
                chunk_count = 0
'''
new='''                chunk_loss = None
                chunk_count = 0
                chunk_predicted_progress = None
                chunk_target_progress = 0.0
'''
if s.count(old)!=1: raise SystemExit(f'ERROR: chunk-reset patch count={s.count(old)}')
s=s.replace(old,new,1)

# Make the validation log explicit: this is the persistent Kalman closed-loop
# state used to judge temporal stability before the final MS refinement.
s=s.replace('val_mle=%.3fm val_p90=%.3fm val_speed_mae=%.3f ',
            'val_preMS_mle=%.3fm val_preMS_p90=%.3fm val_speed_mae=%.3f ',1)

compile(s,str(p),'exec'); p.write_text(s,encoding='utf-8')
print('[FIX] Route-A speed initialization + cumulative-progress training loss patched')
PY

# =============================================================================================
# 3) Fail fast on the canonical model before spending GPUs on all ablations.
# =============================================================================================
python3 - "${RUNNER}" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
needle='''  [[ -f "${CANON_CKPT}" && ! -L "${CANON_CKPT}" ]] || { echo "ERROR: fresh canonical checkpoint missing" >&2; exit 20; }

  ( run_cfg 0 temporal_1frame'''
replacement='''  [[ -f "${CANON_CKPT}" && ! -L "${CANON_CKPT}" ]] || { echo "ERROR: fresh canonical checkpoint missing" >&2; exit 20; }

  python3 - "${CANON_CKPT}" <<'PYCHECK'
import math, sys, torch
p=torch.load(sys.argv[1],map_location="cpu"); v=p.get("validation",{})
mle=float(v.get("mle",float("inf"))); speed=float(v.get("speed_mae",float("inf"))); progress=float(v.get("progress_mae",float("inf")))
print(f"[SANITY][full_model] Route-A pre-MS val_mle={mle:.3f}m speed_mae={speed:.3f}m/frame progress_mae={progress:.3f}m")
if (not math.isfinite(mle) or mle>25.0 or not math.isfinite(progress) or progress>35.0 or not math.isfinite(speed) or speed>1.0):
    raise SystemExit("SANITY FAILED: canonical temporal model is not stable enough; aborting before all ablations")
PYCHECK

  ( run_cfg 0 temporal_1frame'''
if s.count(needle)!=1: raise SystemExit(f'ERROR: sanity-guard insertion count={s.count(needle)}')
s=s.replace(needle,replacement,1)
p.write_text(s,encoding='utf-8')
print('[FIX] strict canonical fail-fast guard added')
PY

# Stable training values are explicit and shared by every freshly-trained ablation.
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-1e-4}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-4.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-4.0}"
export UAVSAT_LOSS_CUMULATIVE_PROGRESS="${UAVSAT_LOSS_CUMULATIVE_PROGRESS:-0.5}"

# Static audit before starting any expensive job.
python3 - "${BASE_MODEL}" "${BASE_TRACKER}" "${BASE_CONFIG}" "${RUNNER}" <<'PY'
from pathlib import Path
import sys
model=Path(sys.argv[1]).read_text(encoding='utf-8')
tracker=Path(sys.argv[2]).read_text(encoding='utf-8')
config=Path(sys.argv[3]).read_text(encoding='utf-8')
runner=Path(sys.argv[4]).read_text(encoding='utf-8')
checks={
 'velocity train/inference consistency':'motion_mode in {"velocity", "none"}' in model,
 'Route-A train speed initialization':'Constant-Velocity initialization from Route-A train mean' in tracker,
 'cumulative progress loss':'LOSS_CUMULATIVE_PROGRESS' in tracker and 'chunk_predicted_progress' in tracker,
 'lower temporal LR':'UAVSAT_TEMPORAL_LR' in config,
 'weighted centroid':'UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid' in runner,
 'scheduled reference':'UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference' in runner,
 '3-frame canonical':'run_cfg 0 full_model 3 fixed 0 1' in runner,
 'strict fail-fast':'mle>25.0' in runner,
}
bad=[k for k,v in checks.items() if not v]
for k,v in checks.items(): print(f"[STATIC] {k}: {'PASS' if v else 'FAIL'}")
if bad: raise SystemExit('STATIC AUDIT FAILED: '+', '.join(bad))
PY

RUN_ALL_EXPERIMENTS=1 bash "${RUNNER}"
