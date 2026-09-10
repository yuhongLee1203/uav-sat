#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SCRIPT_PATH="${ROOT}/run.sh"
BASE_SRC="${ROOT}/base_src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output}"
SRC="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_src}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${OUT}/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
FINAL_ARCH="V39_DirectFinalMS_MobileNetV3_MS1_GRU_Kalman_RegularizedMS2"

# ============================================================================
# Complete paper experiment suite.
# Run once with:
#   RUN_ALL_EXPERIMENTS=1 bash v39_DirectFinalMS/run.sh
# GPUs 0, 5, 6 are used in parallel after the default run warms the feature cache.
# ============================================================================
if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/experiments_${TS}}"
  SHARED_CACHE="${UAVSAT_SHARED_FEATURE_CACHE_DIR:-${ROOT}/output/feature_cache}"
  mkdir -p "${SUITE_ROOT}" "${SHARED_CACHE}"

  run_one() {
    local gpu="$1"
    local name="$2"
    local jitter="$3"
    local kf_weight="$4"
    local ref_weight="$5"
    local bandwidth="$6"
    local out="${SUITE_ROOT}/${name}"
    local runtime="${SUITE_ROOT}/runtime_${name}"

    echo "[START][GPU ${gpu}] ${name}: jitter=${jitter}, kf_w=${kf_weight}, ref_w=${ref_weight}, bw=${bandwidth}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_RUNTIME_DIR="${runtime}" \
    UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${SHARED_CACHE}" \
    JITTER_M="${jitter}" \
    MS2_KF_SIGMA_M=4.0 \
    MS2_REFERENCE_SIGMA_M=4.0 \
    MS2_KF_PRIOR_WEIGHT="${kf_weight}" \
    MS2_REFERENCE_PRIOR_WEIGHT="${ref_weight}" \
    MS2_BANDWIDTH_M="${bandwidth}" \
    EXPERIMENT_TAG="${name}" \
    RUN_ALL_EXPERIMENTS=0 \
    bash "${SCRIPT_PATH}"
    echo "[DONE ][GPU ${gpu}] ${name}"
  }

  echo "============================================================================================================"
  echo "v39 complete paper experiment suite"
  echo "output root: ${SUITE_ROOT}"
  echo "shared feature cache: ${SHARED_CACHE}"
  echo "GPU plan: 0 / 5 / 6"
  echo "============================================================================================================"

  # Main result first: reproduces the selected method and safely warms reusable features.
  run_one 0 "main_full_j8" 8 1.50 2.50 5.0

  # Component ablation + compact one-factor-at-a-time sensitivity.
  (
    run_one 0 "abl_visual_only"       8 0.00 0.00 5.0
    run_one 0 "abl_visual_kf"         8 1.50 0.00 5.0
    run_one 0 "abl_visual_reference"  8 0.00 2.50 5.0
    run_one 0 "sens_bandwidth_4"      8 1.50 2.50 4.0
  ) & pid0=$!

  # Robustness to increasing predefined-reference perturbation.
  (
    run_one 5 "robust_jitter_0"       0 1.50 2.50 5.0
    run_one 5 "robust_jitter_4"       4 1.50 2.50 5.0
    run_one 5 "robust_jitter_12"     12 1.50 2.50 5.0
    run_one 5 "robust_jitter_16"     16 1.50 2.50 5.0
    run_one 5 "sens_bandwidth_6"      8 1.50 2.50 6.0
  ) & pid5=$!

  # Prior-weight sensitivity around the selected defaults.
  (
    run_one 6 "sens_reference_w1p5"   8 1.50 1.50 5.0
    run_one 6 "sens_reference_w3p5"   8 1.50 3.50 5.0
    run_one 6 "sens_kalman_w0p75"     8 0.75 2.50 5.0
    run_one 6 "sens_kalman_w2p25"     8 2.25 2.50 5.0
  ) & pid6=$!

  status=0
  wait "${pid0}" || status=1
  wait "${pid5}" || status=1
  wait "${pid6}" || status=1
  if [[ "${status}" != "0" ]]; then
    echo "ERROR: at least one v39 experiment failed. Check ${SUITE_ROOT}/*/*.log" >&2
    exit 3
  fi

  # Publication-friendly aggregate table. Also includes the established no-MS2 Kalman baseline.
  python3 - "${SUITE_ROOT}" "${REPO_ROOT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

suite = Path(sys.argv[1])
repo = Path(sys.argv[2])
rows = []


def metric_row(name, d, kind="v39"):
    b = d.get("route_B", {})
    c = d.get("route_C", {})
    hp = d.get("ms2_hyperparameters", {})

    def csv_count(route_dict, fallback):
        p = route_dict.get("CSV")
        if p and Path(p).exists():
            try:
                with open(p, newline="", encoding="utf-8") as f:
                    return max(sum(1 for _ in f) - 1, 0) or fallback
            except OSError:
                pass
        return fallback

    nb = csv_count(b, 2276)
    nc = csv_count(c, 1258)
    bm = float(b.get("MLE_m", math.nan))
    cm = float(c.get("MLE_m", math.nan))
    bc = (bm * nb + cm * nc) / max(nb + nc, 1)
    return {
        "Experiment": name,
        "Type": kind,
        "Jitter_m": d.get("experiment_jitter_m", "-"),
        "KF_weight": hp.get("kalman_prior_weight", "-"),
        "Reference_weight": hp.get("reference_prior_weight", "-"),
        "Bandwidth_m": hp.get("bandwidth_m", "-"),
        "B_MLE_m": bm,
        "C_MLE_m": cm,
        "BC_weighted_MLE_m": bc,
        "B_P90_m": b.get("P90_m", math.nan),
        "C_P90_m": c.get("P90_m", math.nan),
        "B_LSR5_pct": b.get("LSR@5_pct", math.nan),
        "C_LSR5_pct": c.get("LSR@5_pct", math.nan),
        "B_LSR15_pct": b.get("LSR@15_pct", math.nan),
        "C_LSR15_pct": c.get("LSR@15_pct", math.nan),
        "B_JumpRate_pct": b.get("JumpRate_pct", math.nan),
        "C_JumpRate_pct": c.get("JumpRate_pct", math.nan),
        "B_MS2Shift_m": b.get("MS2_MeanShiftFromKalman_m", 0.0 if kind == "baseline" else math.nan),
        "C_MS2Shift_m": c.get("MS2_MeanShiftFromKalman_m", 0.0 if kind == "baseline" else math.nan),
    }

baseline = repo / "PreviousState-exp/output/mobilenetv3_prevstate/robust_tracker_summary.json"
if baseline.exists():
    d = json.loads(baseline.read_text(encoding="utf-8"))
    d["experiment_jitter_m"] = 8
    rows.append(metric_row("baseline_kalman_no_ms2", d, "baseline"))

for p in sorted(suite.glob("*/robust_tracker_summary.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.append(metric_row(d.get("experiment_tag", p.parent.name), d))

main = next((r for r in rows if r["Experiment"] == "main_full_j8"), None)
if main:
    main_bc = float(main["BC_weighted_MLE_m"])
    for r in rows:
        bc = float(r["BC_weighted_MLE_m"])
        r["Delta_vs_Main_pct"] = (bc / main_bc - 1.0) * 100.0
else:
    for r in rows:
        r["Delta_vs_Main_pct"] = math.nan

columns = [
    "Experiment", "Type", "Jitter_m", "KF_weight", "Reference_weight", "Bandwidth_m",
    "B_MLE_m", "C_MLE_m", "BC_weighted_MLE_m", "Delta_vs_Main_pct",
    "B_P90_m", "C_P90_m", "B_LSR5_pct", "C_LSR5_pct",
    "B_LSR15_pct", "C_LSR15_pct", "B_JumpRate_pct", "C_JumpRate_pct",
    "B_MS2Shift_m", "C_MS2Shift_m",
]

csv_path = suite / "experiment_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)

md_path = suite / "experiment_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# v39 Experiment Summary\n\n")
    f.write("| Experiment | Jitter | KF w | Ref w | BW | B MLE | C MLE | B+C MLE | Delta vs Main | B LSR@5 | C LSR@5 | B Jump | C Jump |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        def fmt(v, n=3):
            if isinstance(v, str):
                return v
            try:
                if math.isnan(float(v)):
                    return "-"
            except (TypeError, ValueError):
                return str(v)
            return f"{float(v):.{n}f}"
        f.write(
            f"| {r['Experiment']} | {fmt(r['Jitter_m'],1)} | {fmt(r['KF_weight'],2)} | "
            f"{fmt(r['Reference_weight'],2)} | {fmt(r['Bandwidth_m'],1)} | "
            f"{fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | "
            f"{fmt(r['Delta_vs_Main_pct'],2)}% | {fmt(r['B_LSR5_pct'],2)}% | {fmt(r['C_LSR5_pct'],2)}% | "
            f"{fmt(r['B_JumpRate_pct'],3)}% | {fmt(r['C_JumpRate_pct'],3)}% |\n"
        )

print(f"[TABLE] {csv_path}")
print(f"[TABLE] {md_path}")
PY

  echo "============================================================================================================"
  echo "ALL v39 PAPER EXPERIMENTS COMPLETED"
  echo "Results: ${SUITE_ROOT}"
  echo "CSV table: ${SUITE_ROOT}/experiment_summary.csv"
  echo "Markdown table: ${SUITE_ROOT}/experiment_summary.md"
  echo "============================================================================================================"
  exit 0
fi

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
LEGACY_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LOCAL_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }

for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2
    exit 2
  }
done

rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
cp -a "${BASE_SRC}/." "${SRC}/"
python3 "${ROOT}/patch_direct_finalms.py" "${SRC}/robust_tracker.py"

ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

MODE="eval"
if [[ ! -s "${LOCAL_TEMPORAL_CKPT}" ]]; then
  if [[ -s "${LEGACY_TEMPORAL_CKPT}" ]]; then
    ln -sfn "${LEGACY_TEMPORAL_CKPT}" "${LOCAL_TEMPORAL_CKPT}"
    echo "[INFO] Reusing trained Previous-State temporal checkpoint."
  else
    MODE="train_eval"
    echo "[INFO] Temporal checkpoint not found; training Route A before evaluation."
  fi
fi

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-4.0}"
export MS2_REFERENCE_SIGMA_M="${MS2_REFERENCE_SIGMA_M:-4.0}"
export MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-1.50}"
export MS2_REFERENCE_PRIOR_WEIGHT="${MS2_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-5.0}"

echo "============================================================================================================"
echo "v39 Direct FinalMS"
echo "flow: MS1 -> GRU -> Kalman Predict/Update -> regularized MS2 -> FINAL"
echo "NO second Kalman update"
echo "MS2 score: visual similarity + Kalman spatial prior + predefined-reference spatial prior"
echo "experiment: ${EXPERIMENT_TAG:-single_default}"
echo "output: ${OUT}"
echo "============================================================================================================"

cd "${SRC}"
ARGS=(--mode "${MODE}" --reuse-visual --jitter-m "${JITTER_M}")
if [[ "${MODE}" == "train_eval" ]]; then
  ARGS+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
fi

UAVSAT_DEVICE="${DEVICE}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_BACKBONE="${BACKBONE}" \
UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
UAVSAT_EXPERIMENT_ANCHOR=softms \
UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
UAVSAT_EXPERIMENT_MOTION=quadratic \
UAVSAT_EXPERIMENT_KALMAN=learned \
UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${MODE}.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["experiment_tag"] = os.environ.get("EXPERIMENT_TAG", "single_default")
d["experiment_jitter_m"] = float(os.environ.get("JITTER_M", "8"))
d["final_chain"] = "MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final"
d["second_kalman_update"] = "none"
d["persistent_navigation_state"] = "single Kalman posterior"
d["final_decoder"] = "full 6x6 prior-regularized Soft MeanShift; MeanShift output is final"
d["ms2_search_center"] = "nearest permanent SAT lattice point to the single Kalman posterior"
d["ms2_score"] = "UAV-SAT visual likelihood + Kalman spatial prior + predefined-reference spatial prior"
d["ms2_hyperparameters"] = {
    "kalman_sigma_m": float(os.environ.get("MS2_KF_SIGMA_M", "4.0")),
    "reference_sigma_m": float(os.environ.get("MS2_REFERENCE_SIGMA_M", "4.0")),
    "kalman_prior_weight": float(os.environ.get("MS2_KF_PRIOR_WEIGHT", "1.50")),
    "reference_prior_weight": float(os.environ.get("MS2_REFERENCE_PRIOR_WEIGHT", "2.50")),
    "bandwidth_m": float(os.environ.get("MS2_BANDWIDTH_M", "5.0")),
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] result: ${OUT}/robust_tracker_summary.json"
