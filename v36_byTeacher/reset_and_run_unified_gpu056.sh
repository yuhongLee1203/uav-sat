#!/usr/bin/env bash
set -euo pipefail

BRANCH="six-mgk-autonomous-reference-bank-v2"
REPO_ROOT="$(git rev-parse --show-toplevel)"
HERE="${REPO_ROOT}/v36_byTeacher"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${CPU_THREADS:-2}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PUSH_CLEANUP="${PUSH_CLEANUP:-1}"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "Expected branch $BRANCH, got $current_branch" >&2
  exit 2
fi
if ! git diff --cached --quiet; then
  echo "Refusing cleanup because the repository already has staged changes." >&2
  echo "Commit or unstage them first; this protects unrelated work." >&2
  exit 3
fi

echo "============================================================"
echo "UNIFIED v36 reset + full rerun"
echo "formal prior: fixed radial 8.0 m"
echo "formal search: full centered 6x6 = 36"
echo "SoftMS: bandwidth 8 m, tau 0.30"
echo "Route A train; Routes B/C eval"
echo "no 0m oracle in formal tables"
echo "============================================================"

KEEP_TOP=(
  "config.py"
  "config_base.py"
  "data.py"
  "visual_model.py"
  "visual_localizer.py"
  "six_architecture_model.py"
  "robust_tracker_base.py"
  "unified_protocol.py"
  "unified_gpu_runner.py"
  "unified_main_architectures.py"
  "unified_mkg_ablation.py"
  "unified_decoder_timing.py"
  "collect_unified_results.py"
  "reset_and_run_unified_gpu056.sh"
  "UNIFIED_PROTOCOL.md"
)

is_kept () {
  local name="$1"
  local x
  for x in "${KEEP_TOP[@]}"; do
    [[ "$name" == "$x" ]] && return 0
  done
  return 1
}

echo "[cleanup] removing legacy tracked files under v36_byTeacher..."
mapfile -t tracked < <(git ls-files "v36_byTeacher")
remove_tracked=()
for path in "${tracked[@]}"; do
  rel="${path#v36_byTeacher/}"
  top="${rel%%/*}"
  if ! is_kept "$top"; then
    remove_tracked+=("$path")
  fi
done
if (( ${#remove_tracked[@]} > 0 )); then
  git rm -r -f --ignore-unmatch -- "${remove_tracked[@]}"
fi

echo "[cleanup] removing legacy local/untracked files under v36_byTeacher..."
while IFS= read -r -d '' path; do
  name="$(basename "$path")"
  if ! is_kept "$name"; then
    rm -rf -- "$path"
  fi
done < <(find "$HERE" -mindepth 1 -maxdepth 1 -print0)

if ! git diff --cached --quiet -- v36_byTeacher; then
  git commit -m "Reset v36_byTeacher to unified fixed-8m experiment protocol"
  if [[ "$PUSH_CLEANUP" == "1" ]]; then
    git push origin "$BRANCH"
  fi
fi

cd "$HERE"

"$PYTHON_BIN" -m py_compile \
  unified_protocol.py \
  unified_gpu_runner.py \
  unified_main_architectures.py \
  unified_mkg_ablation.py \
  unified_decoder_timing.py \
  collect_unified_results.py \
  six_architecture_model.py \
  visual_localizer.py \
  visual_model.py \
  data.py \
  robust_tracker_base.py

export UAVSAT_CPU_THREADS="$CPU_THREADS"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export UAVSAT_MAIN_JITTER_M="8.0"
export UAVSAT_MAIN_GRID_SIZE="6"
export UAVSAT_MAIN_BANDWIDTH_M="8.0"
export UAVSAT_MAIN_TAU="0.30"
export UAVSAT_MAIN_SEED="2026"
export UAVSAT_CAPTURE_MIN_RATE="0.95"

mkdir -p unified_run_logs

run_gpu () {
  local physical_gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  "$PYTHON_BIN" unified_gpu_runner.py "$@"
}

echo "[stage1] retraining Route-A visual model from scratch on GPU0..."
run_gpu 0 unified_main_architectures.py \
  --mode prepare-visual \
  --visual-epochs "$VISUAL_EPOCHS" \
  --device cuda:0 \
  2>&1 | tee unified_run_logs/stage1_visual_gpu0.log

echo "[preflight] building caches and validating 8m/6x6 candidate capture..."
run_gpu 0 unified_main_architectures.py \
  --mode preflight \
  --device cuda:0 \
  2>&1 | tee unified_run_logs/formal_capture_preflight_gpu0.log

run_arch () {
  local gpu="$1"; local arch="$2"
  echo "[GPU${gpu}] START main ${arch}"
  run_gpu "$gpu" unified_main_architectures.py \
    --mode train-eval \
    --arch "$arch" \
    --epochs "$TEMPORAL_EPOCHS" \
    --device cuda:0 \
    2>&1 | tee "unified_run_logs/main_${arch}_gpu${gpu}.log"
  echo "[GPU${gpu}] DONE main ${arch}"
}

main_worker0 () {
  run_arch 0 MKG
  run_arch 0 GKM
  run_arch 0 delayKG
}
main_worker5 () {
  run_arch 5 MGK
  run_arch 5 KGM
  run_arch 5 delayGK
}
main_worker6 () {
  run_arch 6 GMK
  run_arch 6 KMG
}

main_worker0 & p0=$!
main_worker5 & p5=$!
main_worker6 & p6=$!
status=0
wait "$p0" || status=1
wait "$p5" || status=1
wait "$p6" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "A main-architecture worker failed. See unified_run_logs/." >&2
  exit 1
fi

run_ablation () {
  local gpu="$1"; shift
  local id="$1"; shift
  echo "[GPU${gpu}] START ablation ${id}"
  run_gpu "$gpu" unified_mkg_ablation.py \
    --run-id "$id" \
    --epochs "$TEMPORAL_EPOCHS" \
    --device cuda:0 \
    "$@" \
    2>&1 | tee "unified_run_logs/ablation_${id}_gpu${gpu}.log"
  echo "[GPU${gpu}] DONE ablation ${id}"
}

abl_worker0 () {
  run_ablation 0 window_4x4 --grid-size 4 --allow-low-capture
  run_ablation 0 window_7x7 --grid-size 7 --allow-low-capture
  run_ablation 0 gru_no_xy --gru-ablation no_xy
  run_ablation 0 bandwidth_4m --bandwidth 4
  run_ablation 0 tau_0p20 --tau 0.20
  run_ablation 0 seed_123 --model-seed 123
}

abl_worker5 () {
  run_ablation 5 window_5x5 --grid-size 5 --allow-low-capture
  run_ablation 5 window_8x8 --grid-size 8 --allow-low-capture
  run_ablation 5 gru_no_variance --gru-ablation no_variance
  run_ablation 5 bandwidth_12m --bandwidth 12
  run_ablation 5 tau_0p50 --tau 0.50
  run_ablation 5 decoder_weighted --decoder weighted
}

abl_worker6 () {
  run_ablation 6 component_m --pipeline M
  run_ablation 6 component_mk --pipeline MK
  run_ablation 6 component_mg --pipeline MG
  run_ablation 6 gru_no_temporal_mean --gru-ablation no_temporal_mean
  run_ablation 6 gru_no_first_difference --gru-ablation no_first_difference
  run_ablation 6 seed_456 --model-seed 456
  run_ablation 6 robustness_fixed8_model \
    --mode robustness \
    --robustness-levels "4,8,12,16,20"
}

abl_worker0 & a0=$!
abl_worker5 & a5=$!
abl_worker6 & a6=$!
status=0
wait "$a0" || status=1
wait "$a5" || status=1
wait "$a6" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "An ablation worker failed. See unified_run_logs/." >&2
  exit 1
fi

echo "[timing] Weighted 36-point aggregation vs converged SoftMS-mode aggregation..."
run_gpu 6 unified_decoder_timing.py \
  --device cuda:0 \
  --frames 512 \
  --repeats 200 \
  2>&1 | tee unified_run_logs/decoder_aggregation_timing_gpu6.log

"$PYTHON_BIN" collect_unified_results.py \
  2>&1 | tee unified_run_logs/collect_unified_tables.log

echo "============================================================"
echo "UNIFIED FULL RERUN COMPLETE"
echo "protocol: output/<backbone>/unified_fixed8m_v1/"
echo "tables:   output/<backbone>/unified_fixed8m_v1/tables/"
echo "logs:     v36_byTeacher/unified_run_logs/"
echo "main architecture table is table_10_*_LAST.csv"
echo "============================================================"
