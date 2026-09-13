#!/usr/bin/env bash
# True full retraining launcher for OTHER DATA using the selected final V39 method.
# No FieldAnchor task-specific visual/temporal checkpoint is reused.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
V39_ROOT="${REPO_ROOT}/v39_DirectFinalMS"
BASE_SRC="${V39_ROOT}/base_src"
PATCH_FINALMS="${V39_ROOT}/patch_direct_finalms.py"

DATA_ROOT="${UAVSAT_DATA_ROOT:?ERROR: set UAVSAT_DATA_ROOT to the prepared otherdata root}"
WAYPOINT_DIR="${UAVSAT_WAYPOINT_DIR:-${DATA_ROOT}/route_waypoints}"
SAT_IMAGE="${UAVSAT_SAT_IMAGE:-${DATA_ROOT}/satellite/sim_map_competition_roi_crop.png}"
SAT_JSON="${UAVSAT_SAT_JSON:-${DATA_ROOT}/satellite/sim_map_competition_roi_crop_worldfile_epsg3826.json}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output/otherdata_v39_full_train_${TS}}"
RUNTIME="${OUT}/runtime"
CACHE_DIR="${OUT}/feature_cache"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# Selected formal settings.
BACKBONE="mobilenet_v3_small"
ARCH="V39_Forward3x6_ContextGRU_FixedKalman_FinalMS5x5"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
JITTER_M="${JITTER_M:-8}"
FINAL_MS_GRID=5
FINAL_MS_BANDWIDTH=7.0
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${PATCH_FINALMS}" ]] || { echo "ERROR: missing ${PATCH_FINALMS}" >&2; exit 2; }

for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2
    exit 2
  }
  [[ -f "${WAYPOINT_DIR}/${route}_waypoints.json" ]] || {
    echo "ERROR: missing ${WAYPOINT_DIR}/${route}_waypoints.json" >&2
    exit 2
  }
done
[[ -f "${SAT_IMAGE}" ]] || { echo "ERROR: missing satellite image ${SAT_IMAGE}" >&2; exit 2; }
[[ -f "${SAT_JSON}" ]] || { echo "ERROR: missing satellite geo JSON ${SAT_JSON}" >&2; exit 2; }

# A scratch run must never share an old output/checkpoint/cache directory.
if [[ -e "${OUT}" ]]; then
  echo "ERROR: output already exists: ${OUT}" >&2
  echo "Use a new UAVSAT_OUTPUT_DIR or omit it for the timestamped default." >&2
  exit 3
fi
mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${CACHE_DIR}"
cp -a "${BASE_SRC}/." "${RUNTIME}/"

# Final architecture patch: front visual coordinate readout is posterior weighted
# localization, followed by GRU -> external Kalman -> exactly one final MS.
python3 "${PATCH_FINALMS}" "${RUNTIME}/robust_tracker.py"

# Apply the exact Context-GRU/runtime patch used by the latest formal paper run.
# Main GRU inputs become:
#   temporal mean + first difference + second difference
#   + posterior-weighted satellite context + previous state.
python3 - "${RUNTIME}" <<'PY'
from pathlib import Path
import sys

runtime = Path(sys.argv[1])

p = runtime / "visual_model.py"
s = p.read_text(encoding="utf-8")
old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new = "        self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)\n"
if s.count(old) != 1:
    raise SystemExit(f"ERROR: GRU input-size patch count={s.count(old)}")
s = s.replace(old, new, 1)

old_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
new_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: GRU recurrent-input patch count={s.count(old_block)}")
s = s.replace(old_block, new_block, 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")

p = runtime / "robust_tracker.py"
s = p.read_text(encoding="utf-8")
old_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            step = velocity.copy()
'''
new_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            if bool(getattr(config, "EXPERIMENT_DISABLE_GRU", False)):
                velocity = self.x[2:4].copy()
                step = velocity.copy()
            else:
                step = velocity.copy()
'''
if s.count(old_motion) != 1:
    raise SystemExit(f"ERROR: velocity-fusion patch count={s.count(old_motion)}")
s = s.replace(old_motion, new_motion, 1)

marker = "ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)\n\n\n"
if s.count(marker) != 1:
    raise SystemExit(f"ERROR: fast-grid insertion marker count={s.count(marker)}")
helper = r'''ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)


def fast_regular_grid_indices(
    gallery_xy, gallery_pixel, pixel_index, prior_xy, grid_size, stride, device,
):
    grid_size = int(grid_size)
    start = -(grid_size // 2)
    offsets = range(start, start + grid_size)
    stride = int(stride)
    rows = []
    prior_xy = prior_xy.to(gallery_xy.device, dtype=gallery_xy.dtype)
    for prior in prior_xy:
        distance_squared = (gallery_xy - prior[None, :]).square().sum(dim=1)
        center_index = int(distance_squared.argmin().item())
        center_pixel = gallery_pixel[center_index].detach().cpu().tolist()
        center_x, center_y = (int(round(float(v))) for v in center_pixel)
        row, complete = [], True
        for offset_y in offsets:
            for offset_x in offsets:
                index = pixel_index.get(
                    (center_x + offset_x * stride, center_y + offset_y * stride)
                )
                if index is None:
                    complete = False
                    break
                row.append(index)
            if not complete:
                break
        if not complete:
            row = torch.topk(
                distance_squared, k=grid_size * grid_size, largest=False,
            ).indices.detach().cpu().tolist()
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


'''
s = s.replace(marker, helper, 1)
call_count = s.count("regular_grid_indices(")
if call_count < 2:
    raise SystemExit(f"ERROR: expected >=2 regular_grid_indices calls, got {call_count}")
s = s.replace("regular_grid_indices(", "fast_regular_grid_indices(")
s = s.replace("def fast_fast_regular_grid_indices(", "def fast_regular_grid_indices(", 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")
print("[PATCH] formal Context-GRU runtime PASS")
PY

# Use the otherdata route geometry, never the FieldAnchor waypoint files copied
# with base_src.
rm -rf "${RUNTIME}/route_waypoints"
ln -sfn "${WAYPOINT_DIR}" "${RUNTIME}/route_waypoints"

# Hard guarantee: the scratch output starts with no task-specific checkpoint.
rm -f \
  "${OUT}/checkpoints/visual_retrieval_A_only.pt" \
  "${OUT}/checkpoints/${CKPT_NAME}" \
  "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"

# Static audit of the selected training method before any expensive training.
python3 - "${RUNTIME}/config.py" "${RUNTIME}/visual_model.py" <<'PY'
from pathlib import Path
import sys
cfg = Path(sys.argv[1]).read_text(encoding="utf-8")
mdl = Path(sys.argv[2]).read_text(encoding="utf-8")
checks = {
    "temporal LR 2e-4": "TEMPORAL_LR = 2e-4" in cfg,
    "next-step loss 3.00": "LOSS_NEXT_STEP = 3.00" in cfg,
    "velocity auxiliary loss 0.25": "LOSS_VELOCITY = 0.25" in cfg,
    "3-frame Context GRU": "nn.GRUCell(feature_dim * 5, hidden_dim)" in mdl,
    "SAT context enters GRU": "self.sat_projection(sat_context)" in mdl,
}
for name, ok in checks.items():
    print(f"[STATIC] {name}: {'PASS' if ok else 'FAIL'}")
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("STATIC TRAINING AUDIT FAILED: " + ", ".join(failed))
PY

echo "============================================================================================================"
echo "OTHERDATA - FINAL V39 TRUE FULL TRAINING"
echo "data root        : ${DATA_ROOT}"
echo "waypoints        : ${WAYPOINT_DIR}"
echo "satellite image  : ${SAT_IMAGE}"
echo "satellite geo    : ${SAT_JSON}"
echo "output           : ${OUT}"
echo "GPU              : ${GPU}"
echo "backbone         : ${BACKBONE}"
echo "visual train     : FRESH Route-A task heads, max ${VISUAL_EPOCHS} epochs"
echo "temporal train   : FRESH 3-frame Context GRU on Route A, max ${TEMPORAL_EPOCHS} epochs"
echo "patience         : ${PATIENCE}"
echo "jitter           : ${JITTER_M} m"
echo "visual search    : 6x6 geometry -> causal forward 3x6 = 18 scored patches"
echo "motion           : velocity"
echo "Kalman           : fixed-R, variance=25 m^2"
echo "Final MeanShift  : 5x5, bandwidth=${FINAL_MS_BANDWIDTH} m"
echo "eval             : Route B + Route C"
echo "old task weights : FORBIDDEN"
echo "============================================================================================================"

(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_SAT_IMAGE="${SAT_IMAGE}" \
  UAVSAT_SAT_JSON="${SAT_JSON}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=velocity \
  UAVSAT_EXPERIMENT_KALMAN=fixed \
  UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2=25.0 \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  MS_ENABLED=1 \
  MS_GRID_SIZE="${FINAL_MS_GRID}" \
  MS_BANDWIDTH_M="${FINAL_MS_BANDWIDTH}" \
  MS_MEASURE_LATENCY=0 \
  UAVSAT_MEASURE_LATENCY=0 \
  python3 -u robust_tracker.py \
    --mode train_eval \
    --visual-epochs "${VISUAL_EPOCHS}" \
    --temporal-epochs "${TEMPORAL_EPOCHS}" \
    --patience "${PATIENCE}" \
    --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/train_eval.log"

VISUAL_CKPT="${OUT}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${OUT}/checkpoints/${CKPT_NAME}"
SUMMARY="${OUT}/robust_tracker_summary.json"

[[ -f "${VISUAL_CKPT}" && ! -L "${VISUAL_CKPT}" ]] || {
  echo "AUDIT FAILED: visual checkpoint is missing or is a symlink" >&2; exit 40; }
[[ -f "${TEMPORAL_CKPT}" && ! -L "${TEMPORAL_CKPT}" ]] || {
  echo "AUDIT FAILED: temporal checkpoint is missing or is a symlink" >&2; exit 41; }
[[ -f "${SUMMARY}" ]] || { echo "AUDIT FAILED: missing ${SUMMARY}" >&2; exit 42; }

python3 - "${VISUAL_CKPT}" "${TEMPORAL_CKPT}" "${SUMMARY}" "${ARCH}" <<'PY'
import json
import sys
from pathlib import Path
import torch

visual_path = Path(sys.argv[1])
temporal_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
expected_arch = sys.argv[4]

visual = torch.load(visual_path, map_location="cpu")
temporal = torch.load(temporal_path, map_location="cpu")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

checks = {
    "visual trained only on Route A": visual.get("visual_train_routes") == ["route_A"],
    "visual validation only on Route A": visual.get("visual_validation_routes") == ["route_A"],
    "no previous visual task checkpoint loaded": visual.get("previous_task_checkpoint_loaded") is False,
    "visual checkpoint is task-specific-only": visual.get("model_format") == "task_specific_only",
    "temporal architecture matches final V39": temporal.get("architecture") == expected_arch,
    "temporal trained only on Route A": temporal.get("train_routes") == ["route_A"],
    "temporal validation only on Route A": temporal.get("validation_routes") == ["route_A"],
    "temporal eval routes are B/C": temporal.get("eval_routes") == ["route_B", "route_C"],
    "summary uses 3 frames": int(summary.get("experiment_frame_count", -1)) == 3,
    "summary front-end is weighted centroid": summary.get("experiment_anchor") == "weighted_centroid",
    "summary motion is velocity": summary.get("experiment_motion") == "velocity",
    "summary Kalman is fixed": summary.get("experiment_kalman") == "fixed",
}
for route in ("route_B", "route_C"):
    row = summary.get(route, {})
    checks[f"{route}: one final MS"] = int(row.get("OnlineMeanShiftCount", -1)) == 1
    checks[f"{route}: final MS is 5x5"] = int(row.get("MS_GridSize", -1)) == 5

for name, ok in checks.items():
    print(f"[AUDIT] {name}: {'PASS' if ok else 'FAIL'}")
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("OTHERDATA TRAINING AUDIT FAILED: " + ", ".join(failed))

report = {
    "status": "PASS",
    "visual_training": "fresh task-specific heads on otherdata Route A; public pretrained backbone only",
    "temporal_training": "fresh formal 5-input Context GRU on otherdata Route A",
    "evaluation": ["route_B", "route_C"],
    "candidate_geometry": "6x6 -> causal forward 3x6=18",
    "motion": "velocity",
    "kalman": "fixed-R, variance 25 m^2",
    "final_ms": "5x5, bandwidth 7 m",
    "prior_task_checkpoint_reused": False,
}
(summary_path.parent / "training_audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)

b = summary["route_B"]
c = summary["route_C"]
print("=" * 90)
print("OTHERDATA TRUE FULL TRAINING AUDIT PASS")
print(f"Route B: MLE={b['MLE_m']:.3f} m, P90={b['P90_m']:.3f} m, LSR@5={b['LSR@5_pct']:.2f}%")
print(f"Route C: MLE={c['MLE_m']:.3f} m, P90={c['P90_m']:.3f} m, LSR@5={c['LSR@5_pct']:.2f}%")
print("=" * 90)
PY

echo "[DONE] output: ${OUT}"
echo "[DONE] audit : ${OUT}/training_audit.json"
