#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/run_softms_forward18_final5x5_ablation_all.sh"
PATCH="${ROOT}/patch_temporal_fusion_v2.py"
DERIVED="${ROOT}/.generated_forward18_temporalfix_${$}.sh"

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ -s "${BASE}" ]] || fail "missing ${BASE}"
[[ -s "${PATCH}" ]] || fail "missing ${PATCH}"
trap 'rm -f "${DERIVED}"' EXIT

python3 - "${BASE}" "${DERIVED}" <<'PY'
from pathlib import Path
import sys

src=Path(sys.argv[1])
out=Path(sys.argv[2])
s=src.read_text(encoding='utf-8')

repls=[
    ('TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"',
     'TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"'),
    ('PATIENCE="${PATIENCE:-10}"',
     'PATIENCE="${PATIENCE:-15}"'),
    ('softms_forward18_final5x5_ablation_${TS}',
     'softms_forward18_temporalfix_ablation_${TS}'),
    ('V39_MobileNetV3_Forward18of6x6_SoftMS_GRU_CV_Kalman_Final5x5MS',
     'V39_MobileNetV3_Forward18_SoftMS_GRU_TemporalFusionV2_Kalman_Final5x5MS'),
    ('"method": "6x6 local geometry -> heading-forward 18 -> front SoftMS -> temporal GRU -> constant velocity -> fixed-R Kalman -> final 5x5 SoftMS",',
     '"method": "6x6 -> forward18 -> front SoftMS -> conservative GRU temporal motion -> fixed-R Kalman -> final 5x5 SoftMS with visual+Kalman prior only",'),
    ('for p in patch_direct_finalms.py patch_front_softms.py; do',
     'for p in patch_direct_finalms.py patch_front_softms.py patch_temporal_fusion_v2.py; do'),
    ('  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"\n  python3 -m py_compile',
     '  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"\n  python3 "${ROOT}/patch_temporal_fusion_v2.py" "${runtime}/robust_tracker.py" "${runtime}/config.py"\n  python3 -m py_compile'),
    ('    UAVSAT_EXPERIMENT_MOTION=velocity \\\n',
     '    UAVSAT_EXPERIMENT_MOTION="$([[ "${disable_gru}" == "1" ]] && echo none || echo velocity)" \\\n'
     '    UAVSAT_LOSS_VELOCITY=1.0 \\\n'
     '    UAVSAT_MOTION_VELOCITY_EMA_ALPHA=0.30 \\\n'
     '    UAVSAT_MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME=1.0 \\\n'
     '    UAVSAT_MAX_MEASUREMENT_CORRECTION_PARALLEL_M=2.0 \\\n'
     '    UAVSAT_MAX_MEASUREMENT_CORRECTION_CROSS_M=2.0 \\\n'
     '    UAVSAT_EARLY_STOP_MIN_DELTA=0.02 \\\n'),
]
for old,new in repls:
    n=s.count(old)
    if n != 1:
        raise SystemExit(f'ERROR: expected exactly one occurrence, got {n}: {old[:90]!r}')
    s=s.replace(old,new,1)

# Make the new scientific distinction explicit in the generated log.
needle='echo "[RUN] ${name} | frames=${frames} | front=6x6->forward18 | GRU=$((1-disable_gru)) | Kalman=${kalman} | FinalMS=${ms_enabled} grid=${grid}"'
replacement='echo "[RUN] ${name} | frames=${frames} | front=6x6->forward18 | GRU=$((1-disable_gru)) | motion=$([[ "${disable_gru}" == "1" ]] && echo kalman-CV || echo GRU-velocity) | Kalman=${kalman} | FinalMS=${ms_enabled} grid=${grid} | final-prior=visual+Kalman-only"'
if s.count(needle) != 1:
    raise SystemExit('ERROR: run banner audit failed')
s=s.replace(needle,replacement,1)

out.write_text(s,encoding='utf-8')
print('[OK] generated temporal-fusion v2 ablation runner:',out)
PY

chmod +x "${DERIVED}"
bash -n "${DERIVED}"

echo "============================================================================================================"
echo "TEMPORAL-FUSION V2"
echo "Front       : 6x6 -> heading-forward 18 -> Front SoftMS"
echo "Full motion : conservative GRU velocity"
echo "w/o GRU     : external Kalman constant-velocity baseline (fair non-recurrent baseline)"
echo "Final MS    : 5x5 / BW=7m / visual likelihood + Kalman prior ONLY"
echo "GT final MS : DISABLED (no direct current-frame GT/reference prior in final decoder)"
echo "Training    : Route A, up to ${TEMPORAL_EPOCHS:-80} epochs, patience ${PATIENCE:-15}"
echo "============================================================================================================"

exec bash "${DERIVED}"
