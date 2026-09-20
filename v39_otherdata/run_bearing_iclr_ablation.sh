#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${FORMAL_SUITE_ROOT:-${REPO_ROOT}/v39_otherdata/formal_bearing_v5_allcities_${TS}}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
CITIES=(citya cityb cityc cityd)
GPUS=(0 5 6)

mkdir -p "${SUITE_ROOT}/logs"

# Exact supervision/profile used by the frozen V5 CityA PASS checkpoint.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-2.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-3.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.40}"
export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.50}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-8e-5}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.05}"
export UAVSAT_EARLY_MIN_EPOCH="${UAVSAT_EARLY_MIN_EPOCH:-10}"
export UAVSAT_MOTION_VEL_ALPHA="${UAVSAT_MOTION_VEL_ALPHA:-0.65}"
export UAVSAT_MOTION_STEP_ALPHA="${UAVSAT_MOTION_STEP_ALPHA:-0.70}"
export UAVSAT_MOTION_RESIDUAL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_FORWARD_M:-2.5}"
export UAVSAT_MOTION_RESIDUAL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_CROSS_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE:-0.45}"
export UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DELTA2_SCALE="${UAVSAT_TEMPORAL_DELTA2_SCALE:-1.00}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M="${UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M:-2.00}"
export UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M="${UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M:-1.00}"

export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-6.0}"
export UAVSAT_KALMAN_Q_PROGRESS="${UAVSAT_KALMAN_Q_PROGRESS:-1.50}"
export UAVSAT_KALMAN_Q_CROSS="${UAVSAT_KALMAN_Q_CROSS:-0.40}"
export UAVSAT_KALMAN_Q_VELOCITY="${UAVSAT_KALMAN_Q_VELOCITY:-1.00}"
export UAVSAT_KALMAN_CONFIDENCE_POWER="${UAVSAT_KALMAN_CONFIDENCE_POWER:-0.50}"
export UAVSAT_KALMAN_PRIOR_BLEND_BASE="${UAVSAT_KALMAN_PRIOR_BLEND_BASE:-0.00}"
export UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN="${UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN:-0.18}"
export UAVSAT_KALMAN_PRIOR_BLEND_MAX="${UAVSAT_KALMAN_PRIOR_BLEND_MAX:-0.30}"
export UAVSAT_KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF="${UAVSAT_KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF:-0.60}"
export UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE="${UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE:-0.55}"
export UAVSAT_KALMAN_STEP_RELAX_WIDTH="${UAVSAT_KALMAN_STEP_RELAX_WIDTH:-0.08}"
export UAVSAT_KALMAN_STEP_VISUAL_SLACK_M="${UAVSAT_KALMAN_STEP_VISUAL_SLACK_M:-3.0}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

common_args(){
  local city="$1" gpu="$2"
  COMMON=(
    --suite-root "${SUITE_ROOT}"
    --dataset-root "${DATASET_ROOT}"
    --city "${city}"
    --gpu "${gpu}"
    --backbone mobilenet_v3_small
    --visual-epochs "${VISUAL_EPOCHS}"
    --temporal-epochs "${TEMPORAL_EPOCHS}"
    --epochs-per-route "${TEMPORAL_EPOCHS}"
    --patience "${PATIENCE}"
    --jitter-m 8
    --max-sample-distance-m 15
    --heading-weight-px-per-deg 0
    --ms-bandwidth-m 7
    --seed "${SEED}"
  )
}

echo "================================================================================"
echo "Bearing-UAV FORMAL V5 ALL-CITY RUN"
echo "Cities           : citya cityb cityc cityd"
echo "GPUs             : 0 5 6"
echo "Model            : Full 3-frame only"
echo "Patience         : ${PATIENCE}"
echo "Suite            : ${SUITE_ROOT}"
echo "Ablations        : DISABLED"
echo "Figures          : nav50 + nav51 for every city"
echo "================================================================================"

# Fresh preparation for all four cities. No old generated package is reused.
for city in "${CITIES[@]}"; do
  prepared="${SUITE_ROOT}/${city}/prepared"
  rm -rf "${prepared}" "${prepared}__building"
  mkdir -p "${SUITE_ROOT}/${city}"
  echo "[PREP START] ${city}"
  python3 -u v39_otherdata/bearing_prepare_multicity.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --output-root "${prepared}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_prepare.log"

  common_args "${city}" 0
  python3 -u v39_otherdata/bearing_iclr_ablation.py check \
    "${COMMON[@]}" --train-frames 3 \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_preflight.log"
  echo "[PREP DONE] ${city}"
done

run_city(){
  local city="$1" gpu="$2"
  local prepared="${SUITE_ROOT}/${city}/prepared"
  local full_dir="${SUITE_ROOT}/${city}/variants/full"
  local fig_dir="${full_dir}/formal_figures"
  common_args "${city}" "${gpu}"

  echo "================================================================================"
  echo "[FORMAL TRAIN START] ${city} GPU${gpu}"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_iclr_ablation.py train \
    "${COMMON[@]}" \
    --train-frames 3 \
    --force-train \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_train_full_f3_gpu${gpu}.log"

  ck="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  [[ -s "${ck}" ]] || {
    echo "ERROR: ${city} training checkpoint missing: ${ck}" >&2
    return 20
  }

  echo "================================================================================"
  echo "[FORMAL TEST START] ${city} nav50/nav51 GPU${gpu}"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
    "${COMMON[@]}" \
    --variant full \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_test_full_gpu${gpu}.log"

  summary="${full_dir}/bearing_v39_summary.json"
  [[ -s "${summary}" ]] || {
    echo "ERROR: ${city} formal summary missing: ${summary}" >&2
    return 21
  }

  echo "================================================================================"
  echo "[FORMAL PLOT START] ${city} nav50/nav51"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${prepared}" \
    --output-dir "${full_dir}" \
    --routes test_01 test_02 \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_plot_full.log"

  mkdir -p "${fig_dir}"
  cp "${full_dir}/paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg" \
     "${fig_dir}/nav50_result.jpg"
  cp "${full_dir}/paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg" \
     "${fig_dir}/nav51_result.jpg"
  cp "${full_dir}/paper_figures_waypoint_gt/plot_source_audit.json" \
     "${fig_dir}/plot_source_audit.json"

  for required in \
    "${fig_dir}/nav50_result.jpg" \
    "${fig_dir}/nav51_result.jpg" \
    "${fig_dir}/plot_source_audit.json"; do
    [[ -s "${required}" ]] || {
      echo "ERROR: ${city} required formal figure output missing: ${required}" >&2
      return 22
    }
  done

  echo "[FORMAL FIGURES DONE] ${city}"
  echo "  nav50 -> ${fig_dir}/nav50_result.jpg"
  echo "  nav51 -> ${fig_dir}/nav51_result.jpg"
  echo "[FORMAL CITY DONE] ${city} GPU${gpu}"
}

# Dynamic 3-GPU scheduler: A/B/C fill GPUs 0/5/6. Whichever GPU becomes free
# first immediately starts D.
declare -A PID_GPU=()
declare -A PID_CITY=()
launch_city(){
  local city="$1" gpu="$2"
  ( run_city "${city}" "${gpu}" ) &
  local pid=$!
  PID_GPU["${pid}"]="${gpu}"
  PID_CITY["${pid}"]="${city}"
  echo "[SCHEDULER] launched ${city} on GPU${gpu} pid=${pid}"
}

launch_city citya 0
launch_city cityb 5
launch_city cityc 6
remaining_city="cityd"
failed=0

while ((${#PID_GPU[@]} > 0)); do
  done_pid=""
  set +e
  wait -n -p done_pid
  rc=$?
  set -e

  if [[ -z "${done_pid}" ]]; then
    echo "ERROR: scheduler could not identify completed PID" >&2
    failed=1
    break
  fi

  gpu="${PID_GPU[${done_pid}]}"
  city="${PID_CITY[${done_pid}]}"
  unset 'PID_GPU['"${done_pid}"']'
  unset 'PID_CITY['"${done_pid}"']'

  if [[ "${rc}" -ne 0 ]]; then
    echo "ERROR: ${city} failed on GPU${gpu} rc=${rc}; see ${SUITE_ROOT}/logs/${city}_*.log" >&2
    failed=1
  else
    echo "[SCHEDULER] ${city} completed on GPU${gpu}"
  fi

  if [[ -n "${remaining_city}" ]]; then
    next_city="${remaining_city}"
    remaining_city=""
    launch_city "${next_city}" "${gpu}"
  fi
done

[[ "${failed}" -eq 0 ]] || {
  echo "ERROR: one or more formal city runs failed" >&2
  exit 30
}

# Aggregate the measured Full outputs and the eight final figure paths.
python3 - "${SUITE_ROOT}" <<'PY'
from pathlib import Path
import json, sys

root = Path(sys.argv[1])
cities = ["citya", "cityb", "cityc", "cityd"]
out = {
    "run_type": "formal_full_only",
    "method": "Bearing V5 frozen checkpoint architecture",
    "cities": {},
    "figures": {},
    "macro_average_over_8_held_out_routes": {},
}
rows = []

for city in cities:
    full = root / city / "variants" / "full"
    p = full / "bearing_v39_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out["cities"][city] = data

    nav50 = full / "formal_figures" / "nav50_result.jpg"
    nav51 = full / "formal_figures" / "nav51_result.jpg"
    audit = full / "formal_figures" / "plot_source_audit.json"
    for path in (nav50, nav51, audit):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing formal output: {path}")

    out["figures"][city] = {
        "nav50": str(nav50),
        "nav51": str(nav51),
        "plot_source_audit": str(audit),
    }

    for route_name, metrics in data.items():
        rows.append((city, route_name, metrics))

for key in ("MLE_m", "P90_m", "LSR@3_pct", "LSR@5_pct", "LSR@10_pct"):
    vals = [float(m[key]) for _, _, m in rows if key in m]
    if vals:
        out["macro_average_over_8_held_out_routes"][key] = sum(vals) / len(vals)

out["held_out_route_count"] = len(rows)
out["figure_count"] = 8
out["note"] = (
    "P90_m is the macro-average of per-route P90 values, not a pooled-error P90. "
    "Each city has nav50/nav51 paper figures generated from raw final_x/final_y."
)
(root / "formal_allcities_results.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)
print(json.dumps(out["macro_average_over_8_held_out_routes"], indent=2))
print("[FORMAL OUTPUT] held-out routes =", len(rows))
print("[FORMAL OUTPUT] figures =", out["figure_count"])
PY

printf '%s\n' "${SUITE_ROOT}" > v39_otherdata/LATEST_FORMAL_BEARING_V5_ALLCITIES.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-formal-upload-XXXXXX")"
  upload_branch="formal-v5-upload-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  git fetch origin bearing-v5-formal-allcities
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/bearing-v5-formal-allcities
  dest="paper_results/formal_bearing_v5_allcities_${TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}/formal_allcities_results.json" "${upload_wt}/${dest}/"

  for city in "${CITIES[@]}"; do
    mkdir -p \
      "${upload_wt}/${dest}/${city}/train_frames3" \
      "${upload_wt}/${dest}/${city}/full/formal_figures"

    cp "${SUITE_ROOT}/${city}/prepared/experiment.json" \
      "${upload_wt}/${dest}/${city}/prepared_experiment.json"

    cp "${SUITE_ROOT}/${city}/train_frames3/experiment_manifest.json" \
      "${upload_wt}/${dest}/${city}/train_frames3/" 2>/dev/null || true
    cp "${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json" \
      "${upload_wt}/${dest}/${city}/train_frames3/" 2>/dev/null || true

    cp "${SUITE_ROOT}/${city}/variants/full/bearing_v39_summary.json" \
      "${upload_wt}/${dest}/${city}/full/"
    cp "${SUITE_ROOT}/${city}/variants/full/experiment_manifest.json" \
      "${upload_wt}/${dest}/${city}/full/"
    cp "${SUITE_ROOT}/${city}/variants/full"/*_frames.csv \
      "${upload_wt}/${dest}/${city}/full/" 2>/dev/null || true

    cp "${SUITE_ROOT}/${city}/variants/full/formal_figures/nav50_result.jpg" \
      "${upload_wt}/${dest}/${city}/full/formal_figures/"
    cp "${SUITE_ROOT}/${city}/variants/full/formal_figures/nav51_result.jpg" \
      "${upload_wt}/${dest}/${city}/full/formal_figures/"
    cp "${SUITE_ROOT}/${city}/variants/full/formal_figures/plot_source_audit.json" \
      "${upload_wt}/${dest}/${city}/full/formal_figures/"
  done

  (
    cd "${upload_wt}"
    git add "${dest}"
    git commit -m "Add formal Bearing V5 all-city data and figures ${TS}"
    git fetch origin bearing-v5-formal-allcities
    git rebase origin/bearing-v5-formal-allcities
    git push origin HEAD:bearing-v5-formal-allcities
  )
  echo "GitHub results: ${dest}"
fi

echo "================================================================================"
echo "DONE: FORMAL Bearing V5 ALL CITIES + FIGURES"
echo "Suite   : ${SUITE_ROOT}"
echo "Summary : ${SUITE_ROOT}/formal_allcities_results.json"
for city in "${CITIES[@]}"; do
  echo "${city} nav50: ${SUITE_ROOT}/${city}/variants/full/formal_figures/nav50_result.jpg"
  echo "${city} nav51: ${SUITE_ROOT}/${city}/variants/full/formal_figures/nav51_result.jpg"
done
echo "Branch  : bearing-v5-formal-allcities"
echo "================================================================================"
