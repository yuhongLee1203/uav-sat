#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/v39_otherdata/run_frozen_v5_all_paper.sh"
[[ -s "${SRC}" ]] || { echo "ERROR: missing ${SRC}" >&2; exit 2; }

TMP="$(mktemp /tmp/run_frozen_v5_all_paper_stable.XXXXXX.sh)"
cleanup(){ rm -f "${TMP}"; }
trap cleanup EXIT

python3 - "${SRC}" "${TMP}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
out=Path(sys.argv[2])

old_root='ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
new_root='ROOT="${FROZEN_WRAPPER_ROOT:?FROZEN_WRAPPER_ROOT is required}"\n'
if src.count(old_root) != 1:
    raise SystemExit(f'expected one ROOT declaration, got {src.count(old_root)}')
src=src.replace(old_root,new_root,1)

needle='''  git -C "${WT}" reset --hard "${FROZEN_SHA}" >/dev/null
  git -C "${WT}" clean -fdx >/dev/null
  (
'''
insert='''  git -C "${WT}" reset --hard "${FROZEN_SHA}" >/dev/null
  git -C "${WT}" clean -fdx >/dev/null

  # Stability-only launcher patch. Algorithm/model files remain frozen.
  # The original runner launches frame-2/frame-3 and ablation evals in parallel
  # on GPU0/5/6. For reproducibility, run them sequentially on GPU0 so a failed
  # background worker cannot collapse into the generic "temporal training failed".
  python3 - "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh" <<'PYSEQ'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text(encoding='utf-8')
old_train='''  ( run_train 2 5 ) & p2=$!
  ( run_train 3 6 ) & p3=$!
  status=0
  wait "${p2}" || status=1
  wait "${p3}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: temporal training failed" >&2; exit 20; }
'''
new_train='''  echo "[STABLE MODE] frame-2 and frame-3 training run sequentially on GPU0"
  run_train 2 0
  run_train 3 0
'''
if s.count(old_train) != 1:
    raise SystemExit(f'sequential training patch expected 1 match, got {s.count(old_train)}')
s=s.replace(old_train,new_train,1)
old_eval='''# Full first warms shared held-out caches.
run_eval_group 0 full

( run_eval_group 0 no_gru grid4 grid7 ) & p0=$!
( run_eval_group 5 no_kalman frames1 grid5 ) & p5=$!
( run_eval_group 6 no_ms frames2 grid8 ) & p6=$!
status=0
wait "${p0}" || status=1
wait "${p5}" || status=1
wait "${p6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: evaluation failed; inspect logs" >&2; exit 21; }
'''
new_eval='''# Full first warms shared held-out caches.
run_eval_group 0 full

echo "[STABLE MODE] ablation evaluation runs sequentially on GPU0"
run_eval_group 0 no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8
'''
if s.count(old_eval) != 1:
    raise SystemExit(f'sequential evaluation patch expected 1 match, got {s.count(old_eval)}')
s=s.replace(old_eval,new_eval,1)
p.write_text(s,encoding='utf-8')
print('[STABLE PATCH] temporal train parallelism removed: PASS')
print('[STABLE PATCH] ablation eval parallelism removed: PASS')
PYSEQ
  bash -n "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh"
  grep -Fq '[STABLE MODE] frame-2 and frame-3 training run sequentially on GPU0' "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh"
  grep -Fq '[STABLE MODE] ablation evaluation runs sequentially on GPU0' "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh"
  echo "[STABLE PRECHECK] frozen runner sequential patch: PASS"
  (
'''
if src.count(needle) != 1:
    raise SystemExit(f'expected one per-city reset block, got {src.count(needle)}')
src=src.replace(needle,insert,1)

# Make the banner explicit so logs identify the stable launcher.
src=src.replace('echo "FROZEN V5 ALL-PAPER SUITE"','echo "FROZEN V5 ALL-PAPER SUITE (STABLE SEQUENTIAL)"',1)
out.write_text(src,encoding='utf-8')
PY

chmod +x "${TMP}"
bash -n "${TMP}"
export FROZEN_WRAPPER_ROOT="${ROOT}"
exec bash "${TMP}"
