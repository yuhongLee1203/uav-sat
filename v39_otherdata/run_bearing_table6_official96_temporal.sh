#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BASE_SUITE="${ABCD_BASE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SUITE_ROOT="${ABCD_OFFICIAL96_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_table6_official96_temporal_20260923}"
CITIES=(citya cityb cityc cityd)
REPREPARE="${REPREPARE_OFFICIAL96:-0}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"

METADATA="${DATASET_ROOT}/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv"
[[ -s "${METADATA}" ]] || { echo "ERROR: official rsi_id=96 metadata missing: ${METADATA}" >&2; exit 2; }
[[ -d "${BASE_SUITE}" ]] || { echo "ERROR: base shared suite missing: ${BASE_SUITE}" >&2; exit 2; }
mkdir -p "${SUITE_ROOT}/logs"

python3 -m py_compile \
  v39_otherdata/bearing_prepare_official96_temporal.py \
  v39_otherdata/bearing_iclr_ablation_route_reference.py \
  v39_otherdata/calibrate_final_output_kalman_city.py \
  v39_otherdata/calibrate_final_output_kalman_global_city.py \
  v39_otherdata/eval_with_global_route_reference_profile.py

# Preparation is intentionally sequential and limited to two BLAS threads so
# the 90K metadata adaptation does not consume the whole CPU host.
for city in "${CITIES[@]}"; do
  prepared="${SUITE_ROOT}/${city}/prepared"
  audit="${prepared}/official96_split_audit.json"
  if [[ -s "${audit}" && "${REPREPARE}" != "1" ]]; then
    echo "[OFFICIAL96 PREPARED REUSE] ${city}: ${audit}"
    continue
  fi
  if [[ -e "${prepared}" ]]; then
    archive="${SUITE_ROOT}/audit_prepared_$(date +%Y%m%d_%H%M%S)/${city}"
    mkdir -p "$(dirname "${archive}")"
    mv "${prepared}" "${archive}"
  fi
  python3 -u v39_otherdata/bearing_prepare_official96_temporal.py \
    --dataset-root "${DATASET_ROOT}" --city "${city}" \
    --output-root "${prepared}" \
    --step-m "${OFFICIAL96_EPISODE_STEP_M:-8.0}" \
    --max-target-distance-m "${OFFICIAL96_MAX_TARGET_DISTANCE_M:-30.0}" \
    --max-frame-step-m "${OFFICIAL96_MAX_FRAME_STEP_M:-42.0}" \
    2>&1 | tee "${SUITE_ROOT}/logs/prepare_${city}.log"
done

# This is the exact official 90K split audit expected before model evaluation.
python3 - "${SUITE_ROOT}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
expected={"train":76500,"val":4500,"test":9000}
seen={}
for city in ("citya","cityb","cityc","cityd"):
    p=root/city/"prepared"/"official96_split_audit.json"
    d=json.loads(p.read_text(encoding="utf-8"))
    assert d["official_dataset_id"] == 96
    assert d["split_seed"] == 42
    assert d["split_counts_global"] == expected
    assert d["no_cross_split_samples"] is True
    hashes=d["split_sha256_global"]
    if seen and hashes != seen:
        raise RuntimeError("official split hashes differ across cities")
    seen=hashes
print("[OFFICIAL96 SPLIT AUDIT PASS]", expected, seen)
PY

USE_EXISTING_PREPARED=1 \
BEARING_DATASET_ROOT="${DATASET_ROOT}" \
ABCD_BASE_SUITE_ROOT="${BASE_SUITE}" \
ABCD_ROUTE_REF_SUITE_ROOT="${SUITE_ROOT}" \
ROUTE_REF_ADAPT_ROUNDS="${ROUTE_REF_ADAPT_ROUNDS:-0}" \
CALIBRATION_GPUS="${CALIBRATION_GPUS:-0 5 6}" \
TRAIN_GPU="${TRAIN_GPU:-0}" \
REQUIRE_MLE_VALIDATION="${REQUIRE_MLE_VALIDATION:-1}" \
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-0}" \
bash v39_otherdata/run_bearing_table6_route_reference.sh

echo "[OFFICIAL96 TABLE6 DONE] ${SUITE_ROOT}/table6_global_component.md"
