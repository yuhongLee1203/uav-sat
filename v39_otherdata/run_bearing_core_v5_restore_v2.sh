#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SUITE="${1:-}"
[[ -n "${SUITE}" && -d "${SUITE}" ]] || { echo "ERROR: pass completed Smooth-V1 suite" >&2; exit 2; }
SUITE="$(readlink -f "${SUITE}")"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SEED="${SEED:-2033}"
THREADS="${CPU_THREADS_PER_CITY:-2}"
FORCE="${FORCE_CORE_V5:-1}"
UPLOAD="${UPLOAD_CORE_V5:-1}"
BRANCH="bearing-v5-formal-smooth-v1"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
export MALLOC_ARENA_MAX=2

# Refresh only Core-V5 helpers. Never overwrite the local generated Formal-V5 runner.
git fetch origin "${BRANCH}" >/dev/null
git show "origin/${BRANCH}:v39_otherdata/bearing_core_v5_restore.py" \
  > v39_otherdata/bearing_core_v5_restore.py
git show "origin/${BRANCH}:v39_otherdata/bearing_core_v5_restore_compat.py" \
  > v39_otherdata/bearing_core_v5_restore_compat.py
git show "origin/${BRANCH}:v39_otherdata/build_bearing_core_v5_restore_tables.py" \
  > v39_otherdata/build_bearing_core_v5_restore_tables.py
git show "origin/${BRANCH}:v39_otherdata/bearing_paper_ablation.py" \
  > v39_otherdata/bearing_paper_ablation.py

python3 -m py_compile \
  v39_otherdata/bearing_core_v5_restore.py \
  v39_otherdata/bearing_core_v5_restore_compat.py \
  v39_otherdata/build_bearing_core_v5_restore_tables.py \
  v39_otherdata/bearing_paper_ablation.py

echo "[CORE-V5 REFRESH] helper files restored from origin: PASS"
echo "[CORE-V5 REFRESH] local bearing_iclr_ablation.py preserved: PASS"

# Compatibility preflight against the actual local generated runner, not the GitHub template.
python3 - <<'PY'
from pathlib import Path
import importlib.util
import inspect
import re
import sys

sys.path.insert(0, str(Path('v39_otherdata').resolve()))
import bearing_iclr_ablation as ab
import bearing_core_v5_restore as core
import bearing_core_v5_restore_compat as compat

src = Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
checks = {
    'formal_v5_calibration': 'formal_v5_direct_delta2_residual_kalman' in src,
    'generated_train_frames_ref': 'args.train_frames' in src,
    'compat_entry_exists': Path('v39_otherdata/bearing_core_v5_restore_compat.py').is_file(),
    'compat_forces_3f': 'args.train_frames = 3' in Path('v39_otherdata/bearing_core_v5_restore_compat.py').read_text(encoding='utf-8'),
    'core_parser': callable(core.parser),
    'train_full': callable(ab.train_full),
    'evaluate': callable(ab.evaluate),
}
for k, v in checks.items():
    print(f'[CORE-V5 PRECHECK] {k}: {"PASS" if v else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('ERROR: Core V5 compatibility precheck failed. Do not start GPU jobs.')

# Build a real Core-V5 Namespace and verify the compatibility layer adds train_frames.
p = core.parser()
probe = p.parse_args([
    'train', '--suite-root', '/tmp/corev5_probe', '--city', 'citya', '--gpu', '0'
])
compat._ensure_compat(probe)
assert probe.train_frames == 3
assert probe.epochs_per_route == 100
print('[CORE-V5 PRECHECK] Namespace train_frames=3 injection: PASS')

# Print direct args.* references from the actual local train/eval entrypoints so any future
# generated-runner mismatch is visible before training starts.
for label, fn in [('train_full', ab.train_full), ('evaluate', ab.evaluate)]:
    try:
        text = inspect.getsource(fn)
        refs = sorted(set(re.findall(r'args\.([A-Za-z_][A-Za-z0-9_]*)', text)))
        print(f'[CORE-V5 PRECHECK] {label} direct args refs: {refs}')
    except Exception as exc:
        print(f'[CORE-V5 PRECHECK] {label} source inspection skipped: {exc}')
PY

mkdir -p "${SUITE}/logs"
COMMON=(
  --suite-root "${SUITE}" --dataset-root "${DATASET_ROOT}"
  --backbone mobilenet_v3_small --visual-epochs 30 --temporal-epochs 100
  --epochs-per-route 100 --patience 4 --jitter-m 8 --max-sample-distance-m 15
  --heading-weight-px-per-deg 0 --ms-bandwidth-m 7 --seed "${SEED}"
)
ENTRY="v39_otherdata/bearing_core_v5_restore_compat.py"

train_city(){
  local city="$1" gpu="$2" train_root="${SUITE}/${city}/train_core_v5_restore"
  if [[ "${FORCE}" == "1" ]]; then rm -rf "${train_root}"; fi
  echo "================================================================================"
  echo "[CORE-V5 TRAIN RESTORED 3F] ${city} GPU${gpu}"
  echo "================================================================================"
  python3 -u "${ENTRY}" train \
    "${COMMON[@]}" --city "${city}" --gpu "${gpu}" \
    2>&1 | tee "${SUITE}/logs/core_v5_${city}_train.log"
  [[ -s "${train_root}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" ]] || {
    echo "ERROR: restored 3f checkpoint missing for ${city}: ${train_root}" >&2
    return 21
  }
}

train_city citya 0 & PA=$!
train_city cityb 5 & PB=$!
train_city cityc 6 & PC=$!
failed=0
wait "${PA}" || failed=1
wait "${PB}" || failed=1
wait "${PC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c restored training failed" >&2; exit 30; }
train_city cityd 0

VARIANTS=(corev5_full corev5_no_gru corev5_no_kalman corev5_no_ms corev5_ctx1 corev5_ctx2)
run_city(){
  local city="$1" gpu="$2" variant out
  for variant in "${VARIANTS[@]}"; do
    out="${SUITE}/${city}/variants_core_v5_restore/${variant}"
    if [[ "${FORCE}" == "1" ]]; then rm -rf "${out}"; fi
    echo "================================================================================"
    echo "[CORE-V5 EVAL] ${city} ${variant} GPU${gpu}"
    echo "================================================================================"
    python3 -u "${ENTRY}" eval \
      "${COMMON[@]}" --city "${city}" --gpu "${gpu}" --variant "${variant}" \
      2>&1 | tee "${SUITE}/logs/core_v5_${city}_${variant}.log"
    [[ -s "${out}/bearing_v39_summary.json" ]] || {
      echo "ERROR: missing ${out}/bearing_v39_summary.json" >&2
      return 40
    }
  done
}

run_city citya 0 & PA=$!
run_city cityb 5 & PB=$!
run_city cityc 6 & PC=$!
failed=0
wait "${PA}" || failed=1
wait "${PB}" || failed=1
wait "${PC}" || failed=1
[[ "${failed}" -eq 0 ]] || { echo "ERROR: citya/b/c Core V5 eval failed" >&2; exit 50; }
run_city cityd 0

rm -rf "${SUITE}/paper_core_v5_restore"
python3 -u v39_otherdata/build_bearing_core_v5_restore_tables.py \
  --suite-root "${SUITE}" \
  --output-dir "${SUITE}/paper_core_v5_restore"
cat "${SUITE}/paper_core_v5_restore/PAPER_CORE_V5_RESTORE_TABLES.md"

echo "================================================================================"
echo "CORE V5-RESTORE V2 LOCAL COMPLETE"
echo "Tables: ${SUITE}/paper_core_v5_restore/PAPER_CORE_V5_RESTORE_TABLES.md"
echo "Formal-V5 generated runner compatibility: train_frames=3 injected before calls."
echo "================================================================================"

if [[ "${UPLOAD}" != "1" ]]; then exit 0; fi
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_REL="paper_results/formal_bearing_v5_core_v5_restore_v2_${STAMP}"
TMP="$(mktemp -d /tmp/uavsat-corev5v2-upload.XXXXXX)"
cleanup(){ git worktree remove --force "${TMP}" >/dev/null 2>&1 || true; rm -rf "${TMP}"; }
trap cleanup EXIT

git fetch origin "${BRANCH}" >/dev/null
git worktree add --detach "${TMP}" "origin/${BRANCH}" >/dev/null
mkdir -p "${TMP}/${DEST_REL}/paper_core_v5_restore"
cp -a "${SUITE}/paper_core_v5_restore/." "${TMP}/${DEST_REL}/paper_core_v5_restore/"
cat > "${TMP}/${DEST_REL}/PROTOCOL.txt" <<'EOF'
Core V5-Restore V2:
- Same architecture: Forward-18 SoftMS -> 3-frame recurrent GRU -> constrained Kalman -> final MeanShift -> XY.
- Restores pre-Smooth-V1 lateral/heading estimator dynamics and retrains the 3-frame temporal checkpoint.
- Longitudinal motion/Kalman limits remain train_01-cadence-derived.
- Component table: w/o GRU, w/o Kalman, w/o Final MeanShift, Full.
- Temporal table: same restored 3-frame checkpoint with 1f/2f context truncation.
- Local generated Formal-V5 runner compatibility explicitly injects train_frames=3.
- Held-out nav50/nav51 are not used for automatic numeric editing.
- Localization protocol remains controlled_gt_jitter.
EOF
for city in citya cityb cityc cityd; do
  cp -a "${SUITE}/${city}/train_core_v5_restore/core_v5_restore_manifest.json" \
    "${TMP}/${DEST_REL}/${city}_train_manifest.json"
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE}/${city}/variants_core_v5_restore/${variant}"
    dst="${TMP}/${DEST_REL}/${city}/${variant}"
    mkdir -p "${dst}"
    cp "${src}/bearing_v39_summary.json" "${dst}/"
    cp "${src}/core_v5_restore_manifest.json" "${dst}/"
  done
done
cd "${TMP}"
git add "${DEST_REL}"
git -c user.name="OpenAI" -c user.email="noreply@openai.com" \
  commit -m "Upload Bearing V5 Core V5 restore V2 results ${STAMP}" >/dev/null
git push origin "HEAD:${BRANCH}"
echo "[CORE V5 RESTORE V2 UPLOAD DONE] ${DEST_REL}"
