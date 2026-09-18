#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/run_softms_ablation_all.sh"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
[[ -s "$BASE" ]] || { echo "missing $BASE"; exit 2; }

if [[ -z "${SOURCE_SUITE:-}" ]]; then
  SOURCE_SUITE="$({ ls -dt "$ROOT"/softms_core_ablation_* 2>/dev/null || true; ls -dt "$ROOT"/softms_ablation_* 2>/dev/null || true; } | while read -r d; do
    [[ -f "$d/temporal_1frame/checkpoints/$CKPT_NAME" ]] || continue
    [[ -f "$d/temporal_2frame/checkpoints/$CKPT_NAME" ]] || continue
    [[ -f "$d/temporal_3frame/checkpoints/$CKPT_NAME" ]] || continue
    echo "$d"; break
  done)"
fi
[[ -n "${SOURCE_SUITE:-}" ]] || { echo "No previous original Forward18 suite found. Set SOURCE_SUITE=/path/to/softms_core_ablation_..."; exit 2; }

export SOURCE_CKPT1="$SOURCE_SUITE/temporal_1frame/checkpoints/$CKPT_NAME"
export SOURCE_CKPT2="$SOURCE_SUITE/temporal_2frame/checkpoints/$CKPT_NAME"
export SOURCE_CKPT3="$SOURCE_SUITE/temporal_3frame/checkpoints/$CKPT_NAME"
for x in "$SOURCE_CKPT1" "$SOURCE_CKPT2" "$SOURCE_CKPT3"; do [[ -s "$x" ]] || { echo "missing $x"; exit 2; }; done

TMP="${ROOT}/.strict_original18_final5_$$.sh"
trap 'rm -f "$TMP"' EXIT
python3 - "$BASE" "$TMP" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
s=s.replace('DEFAULT_MS_GRID="${MS_GRID_SIZE:-6}"','DEFAULT_MS_GRID="${MS_GRID_SIZE:-5}"',1)
s=s.replace('softms_core_ablation_${TS}','original18_final5_strict_eval_${TS}',1)
old='''run_cfg temporal_1frame temporal_frames train_eval 1 0 fixed 1 "${DEFAULT_MS_GRID}" "" 0 0
run_cfg temporal_2frame temporal_frames train_eval 2 0 fixed 1 "${DEFAULT_MS_GRID}" "" 0 0
run_cfg temporal_3frame temporal_frames train_eval 3 0 fixed 1 "${DEFAULT_MS_GRID}" "" 0 0

CKPT1="${SUITE_ROOT}/temporal_1frame/checkpoints/${CKPT_NAME}"
CKPT2="${SUITE_ROOT}/temporal_2frame/checkpoints/${CKPT_NAME}"
CKPT3="${SUITE_ROOT}/temporal_3frame/checkpoints/${CKPT_NAME}"
for ck in "${CKPT1}" "${CKPT2}" "${CKPT3}"; do
  [[ -f "${ck}" && ! -L "${ck}" ]] || fail "expected freshly trained checkpoint missing: ${ck}"
done
'''
new='''run_cfg temporal_1frame temporal_frames eval 1 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT1}" 0 0
run_cfg temporal_2frame temporal_frames eval 2 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT2}" 0 0
run_cfg temporal_3frame temporal_frames eval 3 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT3}" 0 0

CKPT1="${SOURCE_CKPT1}"
CKPT2="${SOURCE_CKPT2}"
CKPT3="${SOURCE_CKPT3}"
'''
if old not in s:
    raise SystemExit('ERROR: preserved original runner layout changed; cannot make strict eval safely')
s=s.replace(old,new,1)
s=s.replace("'checkpoint_retrained':category=='temporal_frames'","'checkpoint_retrained':False",1)
Path(sys.argv[2]).write_text(s)
PY
chmod +x "$TMP"
bash -n "$TMP"
echo "STRICT CONTROL: reuse old 1/2/3-frame checkpoints; original Forward18/GRU/Kalman/reference protocol; ONLY default Final MS 6x6 -> 5x5"
echo "SOURCE_SUITE=$SOURCE_SUITE"
exec bash "$TMP"
