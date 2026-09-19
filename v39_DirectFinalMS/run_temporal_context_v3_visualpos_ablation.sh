#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/run_temporal_context_v3_ablation.sh"
[[ -s "${BASE}" ]] || { echo "ERROR: missing ${BASE}" >&2; exit 2; }

TMP="${ROOT}/.temporal_context_v3_visualpos_$$.sh"
trap 'rm -f "${TMP}"' EXIT

python3 - "${BASE}" "${TMP}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')

replacements=[
    ("ARCH=\"V39_Forward18_TemporalContextV3_NoInnovation_GRU_Kalman_Final6\"",
     "ARCH=\"V39_Forward18_TemporalContextV3_VisualPosition_NoInnovation_GRU_Kalman_Final6\""),
    ("# Architecture audit: exactly 5 branches and no position-innovation feature.",
     "# Architecture audit: exactly 6 branches; direct visual position is present; no position innovation."),
    ("grep -q 'GRUCell(feature_dim \\\\* 5' \"${RUNTIME}/visual_model.py\" || fail \"GRU 5-branch audit failed\"",
     "grep -q 'GRUCell(feature_dim \\\\* 6' \"${RUNTIME}/visual_model.py\" || fail \"GRU 6-branch audit failed\""),
    ("grep -q 'self.sat_projection(sat_context)' \"${RUNTIME}/visual_model.py\" || fail \"SAT context missing from main GRU\"",
     "grep -q 'self.sat_projection(sat_context)' \"${RUNTIME}/visual_model.py\" || fail \"SAT context missing from main GRU\"\ngrep -q 'self.visual_position_projection(visual_position)' \"${RUNTIME}/visual_model.py\" || fail \"Forward18 SoftMS visual position missing from main GRU\""),
    ("echo \"[AUDIT OK] GRU = mean + delta + delta2 + SAT context + previous_state; NO position innovation\"",
     "echo \"[AUDIT OK] GRU = mean + delta + delta2 + SAT context + Forward18 SoftMS visual position + previous_state; NO position innovation\""),
    ('\"gru_inputs\": [\"temporal_mean\", \"delta\", \"delta2\", \"satellite_context\", \"previous_state\"]',
     '\"gru_inputs\": [\"temporal_mean\", \"delta\", \"delta2\", \"satellite_context\", \"forward18_softms_visual_position\", \"previous_state\"]'),
    ("echo \"GRU input : temporal mean + delta + delta2 + satellite context + previous state\"",
     "echo \"GRU input : temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state\""),
    ("'GRU input: temporal mean + delta + delta2 + satellite context + previous state only.',",
     "'GRU input: temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state.',"),
]

for old,new in replacements:
    count=src.count(old)
    if count != 1:
        raise SystemExit(f'ERROR: expected exactly one runner token, got {count}: {old[:80]}')
    src=src.replace(old,new,1)

# Strong safety audit remains in the generated runner: no subtraction-based innovation.
Path(sys.argv[2]).write_text(src,encoding='utf-8')
PY

chmod +x "${TMP}"
bash -n "${TMP}"

echo "============================================================================================================"
echo "VISUAL-POSITION / NO-INNOVATION CONTROL"
echo "GRU inputs: mean + delta + delta2 + SAT context + Forward18 SoftMS visual position + previous state"
echo "Forbidden : visual_anchor_se - predicted_se (position innovation)"
echo "Feedback  : previous recurrent state only"
echo "============================================================================================================"

exec bash "${TMP}"
