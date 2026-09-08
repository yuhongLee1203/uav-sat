#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${ROOT}/run_mobilenetv3_kf2_regularized_ms2_eval.sh"
TMP_RUNNER="$(mktemp /tmp/uavsat_kf2tmp_ms2_XXXXXX.sh)"
trap 'rm -f "${TMP_RUNNER}"' EXIT

[[ -f "${BASE_RUNNER}" ]] || { echo "ERROR: missing ${BASE_RUNNER}" >&2; exit 2; }

# Build a dedicated isolated runner from the already validated KF2+MS2 experiment.
# The ONLY estimator-state change is that KF#2 becomes current-frame temporary:
# KF#1 remains the persistent closed-loop state for the next frame, while KF#2
# still exists in the required MS1 -> GRU -> KF1 -> KF2 -> MS2 -> Final chain.
python3 - "${BASE_RUNNER}" "${TMP_RUNNER}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
s = src.read_text(encoding="utf-8")

replacements = [
    (
        'mobilenetv3_kf2_regularized_ms2/src',
        'mobilenetv3_kf2_temporary_ms2/src',
    ),
    (
        'output/mobilenetv3_kf2_regularized_ms2',
        'output/mobilenetv3_kf2_temporary_ms2',
    ),
    (
        'V36_PreviousStateOnly_MobileNetV3_MS1_GRU_KF1_KF2_RegularizedMS2',
        'V36_PreviousStateOnly_MobileNetV3_MS1_GRU_KF1_TemporaryKF2_RegularizedMS2',
    ),
    (
        '        kf.P = IKH2 @ P_before_kf2 @ IKH2.T + K2 @ R2 @ K2.T\\n'
        '        kf.x = state_after_kf2\\n'
        '        kf2_se = kf.se()\\n',
        '        # KF#2 is a CURRENT-FRAME refinement only. Preserve the\\n'
        '        # persistent KF#1 state/covariance for the next frame so KF#2\\n'
        '        # cannot corrupt closed-loop route progress.\\n'
        '        kf2_P = IKH2 @ P_before_kf2 @ IKH2.T + K2 @ R2 @ K2.T\\n'
        '        kf2_se = state_after_kf2[:2].copy()\\n',
    ),
    (
        'second constrained position Kalman update using current predefined frame reference measurement',
        'temporary current-frame second constrained Kalman update using predefined frame reference measurement; KF1 remains persistent state',
    ),
    (
        'current predefined frame reference position with adaptive covariance and bounded correction',
        'temporary current-frame predefined reference update; persistent KF1 state is unchanged',
    ),
]

for old, new in replacements:
    if old not in s:
        raise SystemExit(f"ERROR: expected template pattern not found: {old[:100]!r}")
    s = s.replace(old, new)

# Add an explicit summary flag so uploaded results prove that the new run did not
# feed KF#2 into the next frame.
needle = 'd["second_kalman_update"] = "temporary current-frame predefined reference update; persistent KF1 state is unchanged"\n'
if needle not in s:
    raise SystemExit("ERROR: summary insertion anchor missing")
s = s.replace(
    needle,
    needle + 'd["kf2_persistent_feedback"] = False\n'
             'd["persistent_navigation_state"] = "KF1 posterior only"\n'
             'd["required_final_chain"] = "MS1 -> GRU -> KF Predict/Update #1 -> temporary KF Update #2 -> MS2 -> Final"\n',
    1,
)

dst.write_text(s, encoding="utf-8")
PY

chmod +x "${TMP_RUNNER}"

echo "============================================================================================================"
echo "MobileNetV3 temporary-KF2 + regularized MS2"
echo "persistent next-frame state: KF#1 only"
echo "current-frame final chain: MS1 -> GRU -> KF1 -> temporary KF2 -> MS2 -> FINAL"
echo "============================================================================================================"

exec bash "${TMP_RUNNER}"
