#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${ROOT}/run_softms_ablation_all.sh"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
[[ -s "$BASE" ]] || { echo "missing $BASE"; exit 2; }

# Prefer the complete Forward-18 suite because it contains independently trained
# 1/2/3-frame checkpoints with the same front geometry and temporal model.
if [[ -z "${SOURCE_SUITE:-}" ]]; then
  SOURCE_SUITE="$({
    ls -dt "$ROOT"/softms_forward18_final5x5_ablation_* 2>/dev/null || true
    ls -dt "$ROOT"/softms_core_ablation_* 2>/dev/null || true
    ls -dt "$ROOT"/softms_ablation_* 2>/dev/null || true
  } | while read -r d; do
    [[ -f "$d/temporal_1frame/checkpoints/$CKPT_NAME" ]] || continue
    [[ -f "$d/temporal_2frame/checkpoints/$CKPT_NAME" ]] || continue
    [[ -f "$d/temporal_3frame/checkpoints/$CKPT_NAME" ]] || continue
    echo "$d"; break
  done)"
fi
[[ -n "${SOURCE_SUITE:-}" ]] || { echo "No complete Forward18 1/2/3-frame suite found. Set SOURCE_SUITE=/path/to/suite"; exit 2; }

export SOURCE_CKPT1="$SOURCE_SUITE/temporal_1frame/checkpoints/$CKPT_NAME"
export SOURCE_CKPT2="$SOURCE_SUITE/temporal_2frame/checkpoints/$CKPT_NAME"
export SOURCE_CKPT3="$SOURCE_SUITE/temporal_3frame/checkpoints/$CKPT_NAME"
for x in "$SOURCE_CKPT1" "$SOURCE_CKPT2" "$SOURCE_CKPT3"; do
  [[ -s "$x" ]] || { echo "missing $x"; exit 2; }
done

SOURCE_SUMMARY="$SOURCE_SUITE/temporal_3frame/robust_tracker_summary.json"
[[ -s "$SOURCE_SUMMARY" ]] || { echo "missing source summary: $SOURCE_SUMMARY"; exit 2; }
SOURCE_ARCH="$(python3 - "$SOURCE_SUMMARY" <<'PY_ARCH'
import json, sys
p=sys.argv[1]
d=json.load(open(p,encoding='utf-8'))
print(d.get('architecture',''))
PY_ARCH
)"
[[ -n "$SOURCE_ARCH" ]] || { echo "source checkpoint architecture metadata is empty"; exit 2; }

TMP="${ROOT}/.strict_original18_final5_$$.sh"
trap 'rm -f "$TMP"' EXIT
python3 - "$BASE" "$TMP" "$SOURCE_ARCH" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(encoding='utf-8')
out=Path(sys.argv[2])
source_arch=sys.argv[3]

old_grid='DEFAULT_MS_GRID="${MS_GRID_SIZE:-6}"'
if s.count(old_grid) != 1:
    raise SystemExit(f'ERROR: default grid token count={s.count(old_grid)}')
s=s.replace(old_grid,'DEFAULT_MS_GRID="${MS_GRID_SIZE:-5}"',1)

old_suite='softms_core_ablation_${TS}'
if s.count(old_suite) != 1:
    raise SystemExit(f'ERROR: suite-root token count={s.count(old_suite)}')
s=s.replace(old_suite,'original18_final5_strict_eval_${TS}',1)

old_arch='ARCH="V39_Forward3x6_SoftMS_GRU_Kalman_FinalMS"'
if s.count(old_arch) != 1:
    raise SystemExit(f'ERROR: architecture token count={s.count(old_arch)}')
s=s.replace(old_arch, f'ARCH="{source_arch}"', 1)

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
out.write_text(s,encoding='utf-8')
PY
chmod +x "$TMP"
bash -n "$TMP"

echo "============================================================================================================"
echo "STRICT EVAL-ONLY CONTROL"
echo "Source suite : $SOURCE_SUITE"
echo "Source arch  : $SOURCE_ARCH"
echo "Front        : original 6x6 geometry -> heading-forward 18 -> Front SoftMS"
echo "Temporal     : reuse existing 1/2/3-frame checkpoints; NO training"
echo "GRU/Kalman   : original source behavior"
echo "Final MS     : only default grid changed 6x6 -> 5x5; all other priors/weights unchanged"
echo "============================================================================================================"
exec bash "$TMP"
