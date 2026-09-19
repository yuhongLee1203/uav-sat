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

base = Path(sys.argv[1])
out = Path(sys.argv[2])
src = base.read_text(encoding="utf-8")

old_arch = 'ARCH="V39_Forward18_TemporalContextV3_NoInnovation_GRU_Kalman_Final6"'
new_arch = 'ARCH="V39_Forward18_TemporalContextV3_VisualPosition_NoInnovation_GRU_Kalman_Final6"'
if old_arch not in src:
    raise SystemExit("ERROR: base runner architecture token not found")
src = src.replace(old_arch, new_arch, 1)

# Replace the complete stale architecture-audit block rather than trying to
# match an escaped grep expression. This makes the wrapper robust to shell
# escaping differences in the base runner.
start = src.find("# Architecture audit:")
end = src.find("export TORCH_HOME=", start)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("ERROR: cannot locate architecture audit block in base runner")

audit = r'''# Architecture audit: six direct inputs and NO position innovation.
grep -Fq 'GRUCell(feature_dim * 6' "${RUNTIME}/visual_model.py" || fail "GRU 6-branch audit failed"
grep -Fq 'self.sat_projection(sat_context)' "${RUNTIME}/visual_model.py" || fail "SAT context missing from main GRU"
grep -Fq 'self.visual_position_projection(visual_position)' "${RUNTIME}/visual_model.py" || fail "Forward18 SoftMS visual position missing from main GRU"
grep -Fq 'visual_anchor_se[:, 0:1]' "${RUNTIME}/visual_model.py" || fail "direct visual position is not sourced from visual_anchor_se"
if grep -Fq 'visual_anchor_se - predicted_se' "${RUNTIME}/visual_model.py"; then
  fail "position innovation unexpectedly present in main GRU"
fi
if grep -Fq 'innovation_projection' "${RUNTIME}/visual_model.py"; then
  fail "innovation projection unexpectedly present in main GRU"
fi

echo "[AUDIT OK] GRU = mean + delta + delta2 + SAT context + Forward18 SoftMS visual position + previous_state; NO position innovation"

'''
src = src[:start] + audit + src[end:]

old_inputs = '"gru_inputs": ["temporal_mean", "delta", "delta2", "satellite_context", "previous_state"]'
new_inputs = '"gru_inputs": ["temporal_mean", "delta", "delta2", "satellite_context", "forward18_softms_visual_position", "previous_state"]'
if old_inputs in src:
    src = src.replace(old_inputs, new_inputs, 1)

src = src.replace(
    'echo "GRU input : temporal mean + delta + delta2 + satellite context + previous state"',
    'echo "GRU input : temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state"',
    1,
)
src = src.replace(
    "'GRU input: temporal mean + delta + delta2 + satellite context + previous state only.',",
    "'GRU input: temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state.',",
    1,
)

# Static sanity checks on the generated runner itself.
if "GRUCell(feature_dim * 5" in src:
    raise SystemExit("ERROR: stale 5-branch audit remains in generated runner")
if "Forward18 SoftMS visual position" not in src:
    raise SystemExit("ERROR: generated runner lost visual-position declaration")

out.write_text(src, encoding="utf-8")
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
