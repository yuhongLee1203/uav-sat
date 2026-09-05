#!/usr/bin/env bash
# V36 ablation: SoftMS measurement + Kalman-only constant-velocity prediction.
# No GRU correction, no GRU motion/heading, no learned polynomial.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/original_forNX"
SRC="${ROOT}/ms_kalman_only"
FORNX="${ROOT}/../forNX"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${ROOT}/v36_training_data}"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output/ms_kalman_only}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
BACKBONE="${UAVSAT_BACKBONE:-mobileclip2_s2}"
JITTER_M="${JITTER_M:-8}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${ROOT}/output/meanshift_gru/feature_cache}"
ARCH="V36_SoftMS_KalmanOnly_Forward3x6"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE}/${f}" ]] || { echo "ERROR: missing ${BASE}/${f}" >&2; exit 2; }
done
for route in route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2;
  }
done

ORIG_VISUAL="${FORNX}/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
[[ -s "${ORIG_VISUAL}" ]] || {
  echo "ERROR: missing original forNX visual checkpoint: ${ORIG_VISUAL}" >&2; exit 2;
}

# Rebuild the experiment folder from the CURRENT original forNX source.
rm -rf "${SRC}"
mkdir -p "${SRC}"
cp -a "${BASE}/." "${SRC}/"

# Make the copied source itself default to the intended ablation:
#   GRU disabled -> measurement = raw SoftMS anchor; variance = MS response variance.
#   motion none  -> RouteKalman.predict() uses its own posterior velocity x[2:4].
python3 - "${SRC}/config.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old_arch = '"V34ProtocolCompactGRUSoftMSModeVarianceForward3x6PolynomialKalman_v36",'
new_arch = '"V36_SoftMS_KalmanOnly_Forward3x6",'
if s.count(old_arch) != 1:
    raise SystemExit(f"ERROR: architecture default pattern count={s.count(old_arch)}")
s = s.replace(old_arch, new_arch, 1)

old_motion = 'EXPERIMENT_MOTION = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "quadratic")'
new_motion = 'EXPERIMENT_MOTION = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "none")'
if s.count(old_motion) != 1:
    raise SystemExit(f"ERROR: motion default pattern count={s.count(old_motion)}")
s = s.replace(old_motion, new_motion, 1)

old_gru = 'EXPERIMENT_DISABLE_GRU = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "0") == "1"'
new_gru = 'EXPERIMENT_DISABLE_GRU = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "1") == "1"'
if s.count(old_gru) != 1:
    raise SystemExit(f"ERROR: disable-GRU default pattern count={s.count(old_gru)}")
s = s.replace(old_gru, new_gru, 1)

p.write_text(s, encoding="utf-8")
PY

cat > "${SRC}/EXPERIMENT_NOTE.txt" <<'EOF'
SoftMS + Kalman-only ablation
==============================
Main estimator flow:
  SoftMS anchor + SoftMS response variance -> RouteKalman.update()
  RouteKalman posterior [s,e,vs,ve] -> constant-velocity RouteKalman.predict()

Disabled:
  GRU recurrent correction
  GRU velocity / acceleration
  GRU heading / turn-rate
  learned second-order polynomial

Preserved:
  original visual checkpoint
  forward 3x6 local candidate search
  Soft Mean-Shift decoder
  visual confidence / response variance used by Kalman update
  original constrained route-coordinate Kalman
  same controlled 8 m smooth-jitter protocol
EOF

# Everything except config.py + the note must remain byte-identical to original forNX.
for f in data.py robust_tracker.py visual_localizer.py visual_model.py; do
  cmp -s "${BASE}/${f}" "${SRC}/${f}" || {
    echo "ERROR: unintended code difference in ${f}" >&2; exit 3;
  }
done

rm -rf "${OUT}"
mkdir -p "${OUT}/checkpoints" "${CACHE_DIR}"
ln -sfn "${ORIG_VISUAL}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

export TORCH_HOME="${ROOT}/pretrained_cache/torch"
export HF_HOME="${ROOT}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== V36 SOFTMS + KALMAN-ONLY ABLATION ==="
echo "base source       : ${BASE}"
echo "experiment source : ${SRC}"
echo "output            : ${OUT}"
echo "device            : ${DEVICE}"
echo "backbone          : ${BACKBONE}"
echo "visual checkpoint : ${ORIG_VISUAL} (REUSED)"
echo "feature cache     : ${CACHE_DIR}"
echo "GRU               : DISABLED"
echo "Kalman prediction : own posterior velocity x[2:4] (constant velocity)"
echo "measurement       : raw SoftMS anchor + MS response variance"
echo "preserved         : forward 3x6, controlled 8m smooth jitter, constrained Kalman"

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=none \
  UAVSAT_EXPERIMENT_KALMAN=learned \
  UAVSAT_EXPERIMENT_DISABLE_GRU=1 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  python3 -u robust_tracker.py \
    --mode eval \
    --reuse-visual \
    --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/eval.log"

echo "[DONE] SoftMS + Kalman-only result: ${OUT}/robust_tracker_summary.json"
