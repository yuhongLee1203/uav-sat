#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SOURCE_RUN="${ROOT}/run.sh"
BEARING_RUNNER="${REPO_ROOT}/v39_otherdata/bearing_runner.py"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/softms18_0913_all_${TS}}"
LOG_ROOT="${SUITE_ROOT}/launcher_logs"
mkdir -p "${SUITE_ROOT}" "${LOG_ROOT}"

# This launcher intentionally keeps the 2026-09-13 V39 source/training pipeline.
# It changes only the front decoder from weighted_centroid -> softms.  The
# original 6x6 geometry and heading-forward 3x6=18 scored candidates stay fixed.
# Historical gate/split-gate patch files are NOT loaded.

[[ -f "${SOURCE_RUN}" ]] || { echo "ERROR: missing ${SOURCE_RUN}" >&2; exit 2; }
[[ -f "${BEARING_RUNNER}" ]] || { echo "ERROR: missing ${BEARING_RUNNER}" >&2; exit 2; }

# The canonical base source tree on the branch is the same tree used by the
# Sep-13 commit (tree SHA 9a3e020dabf776a17b16cc040d265ed51dbd9dd9).
# Do not apply later 5x5/15, split-gate, dual-gate, temporal-v3, or residual-gate patches.
for forbidden in \
  patch_forward5x5_15.py \
  patch_gru_gate_eval.py \
  patch_gru_dualgate_eval.py \
  patch_temporal_context_v3.py \
  patch_temporal_fusion_v2.py \
  patch_split_gru_gate_calibrated.py; do
  if grep -Fq "${forbidden}" "${SOURCE_RUN}"; then
    echo "ERROR: canonical run.sh unexpectedly loads ${forbidden}" >&2
    exit 3
  fi
done

PATCHED_RUN="${ROOT}/.run_0913_softms18_${TS}_$$.sh"
PATCHED_BEARING="${REPO_ROOT}/v39_otherdata/.bearing_runner_softms18_${TS}_$$.py"
cleanup() {
  rm -f "${PATCHED_RUN}" "${PATCHED_BEARING}"
}
trap cleanup EXIT

python3 - "${SOURCE_RUN}" "${PATCHED_RUN}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
# Functional change: front 18-candidate decoder only.
s=src.replace('weighted_centroid','softms')
# Make generated labels explicit without altering computation.
s=s.replace('V39_WeightedCentroid_GRU_Kalman_MS','V39_Sep13_SoftMS18_GRU_Kalman_FinalMS')
s=s.replace('Weighted Centroid','Front SoftMS(18)')
s=s.replace('wc_final_ablation_','softms18_0913_ablation_')
s=s.replace('abl_wc_','abl_ms_')
s=s.replace('WC only','Front SoftMS only')
s=s.replace('WC + GRU','Front SoftMS + GRU')
s=s.replace('WC + GRU + Kalman','Front SoftMS + GRU + Kalman')
if s == src:
    raise SystemExit('ERROR: run.sh did not contain weighted_centroid; refusing ambiguous patch')
# Guard the requested geometry and old training settings.
required=[
    'UAVSAT_EXPERIMENT_FORWARD_ONLY=1',
    'DEFAULT_MS_GRID="6"',
    'TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"',
    'DEFAULT_MOTION="velocity"',
    'DEFAULT_KALMAN="fixed"',
]
missing=[x for x in required if x not in s]
if missing:
    raise SystemExit('ERROR: Sep13 invariant missing after patch: '+repr(missing))
if 'weighted_centroid' in s:
    raise SystemExit('ERROR: weighted_centroid remains in patched canonical runner')
Path(sys.argv[2]).write_text(s, encoding='utf-8')
PY
chmod +x "${PATCHED_RUN}"
bash -n "${PATCHED_RUN}"

python3 - "${BEARING_RUNNER}" "${PATCHED_BEARING}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
s=src.replace('weighted_centroid','softms')
s=s.replace('V39_Forward3x6_ContextGRU_FixedKalman_FinalMS5x5_BearingUAV',
            'V39_Sep13_Forward18SoftMS_ContextGRU_FixedKalman_FinalMS5x5_BearingUAV')
# Keep every other Sep-13 Bearing setting unchanged, including its existing
# post-Kalman MS window, dataset adapter, losses, route sampling, and optimizer.
if s == src:
    raise SystemExit('ERROR: Bearing runner did not contain weighted_centroid')
if 'weighted_centroid' in s:
    raise SystemExit('ERROR: weighted_centroid remains in patched Bearing runner')
for bad in ('split_gate','dualgate','dual_gate','patch_gru_gate_eval','patch_gru_dualgate_eval'):
    if bad in s.lower():
        raise SystemExit('ERROR: forbidden gate logic found in Bearing runtime: '+bad)
Path(sys.argv[2]).write_text(s, encoding='utf-8')
PY

cat > "${SUITE_ROOT}/experiment_definition.txt" <<EOF
Source baseline : V39 pipeline as recorded on 2026-09-13
Only model change: front decoder weighted_centroid -> Soft MeanShift
Front geometry   : 6x6 constructed, heading-forward 3x6 = 18 scored candidates
Temporal         : 3-frame GRU for Full; fair 1/2/3-frame retraining in temporal ablation
Motion           : Constant Velocity (Sep-13 selected setting)
Kalman           : Fixed-R (Sep-13 selected setting)
Post-Kalman MS   : unchanged from each Sep-13 runner
Forbidden extras : split gate / dual gate / residual gate / 5x5-forward15 patch
GPU pool         : 0, 5, 6
EOF

printf '\n===== PHASE 1: canonical main + ablations (GPU 0/5/6) =====\n'
RUN_ALL_EXPERIMENTS=1 \
EXPERIMENT_SUITE_DIR="${SUITE_ROOT}/main_ablation" \
bash "${PATCHED_RUN}" 2>&1 | tee "${LOG_ROOT}/main_ablation.log"

printf '\n===== PHASE 2: Bearing-UAV city A/B/C/D (GPU 0/5/6) =====\n'
BEARING_DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BEARING_VISUAL_EPOCHS="${BEARING_VISUAL_EPOCHS:-30}"
BEARING_EPOCHS_PER_ROUTE="${BEARING_EPOCHS_PER_ROUTE:-20}"
BEARING_PATIENCE="${BEARING_PATIENCE:-5}"
BEARING_JITTER_M="${BEARING_JITTER_M:-8}"

run_city() {
  local city="$1" gpu="$2"
  echo "[BEARING START] ${city} GPU${gpu}"
  python3 -u "${PATCHED_BEARING}" \
    --dataset-root "${BEARING_DATASET_ROOT}" \
    --city "${city}" \
    --gpu "${gpu}" \
    --visual-epochs "${BEARING_VISUAL_EPOCHS}" \
    --epochs-per-route "${BEARING_EPOCHS_PER_ROUTE}" \
    --patience "${BEARING_PATIENCE}" \
    --jitter-m "${BEARING_JITTER_M}" \
    2>&1 | tee "${LOG_ROOT}/bearing_${city}.log"
  echo "[BEARING DONE] ${city} GPU${gpu}"
}

# Three GPU queues. GPU0 handles two cities sequentially; GPU5/GPU6 each handle one.
( run_city citya 0; run_city cityd 0 ) & q0=$!
( run_city cityb 5 ) & q5=$!
( run_city cityc 6 ) & q6=$!
status=0
wait "${q0}" || status=1
wait "${q5}" || status=1
wait "${q6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: one or more Bearing city runs failed" >&2; exit 30; }

printf '\n===== PHASE 3: integrity + Full-best audit =====\n'
python3 - "${SUITE_ROOT}" <<'PY'
import json, sys
from pathlib import Path
suite=Path(sys.argv[1])
main=suite/'main_ablation'

def load(name):
    p=main/name/'robust_tracker_summary.json'
    if not p.exists():
        raise SystemExit(f'AUDIT ERROR: missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def combined_mle(d):
    # Use actual route frame counts when available; fall back to the historical
    # B/C counts used by the Sep-13 table builder.
    b=d['route_B']; c=d['route_C']
    nb=int(b.get('frames',2276)); nc=int(c.get('frames',1258))
    return (float(b['MLE_m'])*nb + float(c['MLE_m'])*nc) / max(nb+nc,1)

# Patched Sep-13 runner renames abl_wc_* -> abl_ms_*.
names=['temporal_1frame','temporal_2frame','temporal_3frame',
       'abl_ms_only','abl_ms_gru','abl_ms_gru_kalman']
rows={n:load(n) for n in names}
for n,d in rows.items():
    if d.get('experiment_anchor')!='softms':
        raise SystemExit(f'AUDIT ERROR [{n}]: front decoder is {d.get("experiment_anchor")!r}, expected softms')
    if not bool(d.get('forward_only_local_search', d.get('forward_only', True))):
        raise SystemExit(f'AUDIT ERROR [{n}]: forward-only search is not enabled')

scores={n:combined_mle(d) for n,d in rows.items()}
full=scores['temporal_3frame']
module_comp={k:v for k,v in scores.items() if k.startswith('abl_ms_')}
temporal_comp={k:v for k,v in scores.items() if k.startswith('temporal_')}
module_best = all(full <= v + 1e-12 for v in module_comp.values())
temporal_best = all(full <= v + 1e-12 for k,v in temporal_comp.items() if k!='temporal_3frame')
full_best=bool(module_best and temporal_best)
report={
    'front_decoder':'softms',
    'front_candidates':18,
    'split_gate_used':False,
    'combined_mle_m':scores,
    'full_definition':'temporal_3frame = front SoftMS18 + 3-frame GRU + fixed-R Kalman + final MS',
    'full_best_module_ablation':module_best,
    'full_best_temporal_ablation':temporal_best,
    'FULL_BEST_CHECK':'PASS' if full_best else 'FAIL',
    'research_integrity_note':(
        'PASS/FAIL is reported from measured results only. No ablation is weakened and no metric is edited to force Full to win.'
    ),
}
out=suite/'full_best_audit.json'
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
print(f'[AUDIT] saved {out}')
if not full_best:
    print('[AUDIT] FULL_BEST_CHECK=FAIL: results are preserved; architecture/training needs scientific redesign if Full must genuinely outperform.')
PY

printf '\n====================================================================================================\n'
echo "ALL REQUESTED RUNS FINISHED"
echo "Suite root : ${SUITE_ROOT}"
echo "Main table : ${SUITE_ROOT}/main_ablation/paper_tables.md"
echo "Full audit : ${SUITE_ROOT}/full_best_audit.json"
echo "Logs       : ${LOG_ROOT}"
echo "===================================================================================================="
