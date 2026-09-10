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
FINAL_ARCH="V39_DirectFinalMS_MobileNetV3_MS1_GRU_Kalman_MS2"

# ============================================================================
# Paper experiment suite: architecture/module ablations + genuine design sizes.
# No reference-noise robustness sweep and no reference-weight sweep.
# ============================================================================
if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/experiments_${TS}}"
  SHARED_CACHE="${UAVSAT_SHARED_FEATURE_CACHE_DIR:-${ROOT}/output/feature_cache}"
  mkdir -p "${SUITE_ROOT}" "${SHARED_CACHE}"

  run_one() {
    local gpu="$1"
    local name="$2"
    local anchor="$3"
    local motion="$4"
    local kalman="$5"
    local disable_gru="$6"
    local forward_only="$7"
    local ms2_enabled="$8"
    local ms2_grid="$9"
    local bandwidth="${10}"
    local category="${11}"
    local out="${SUITE_ROOT}/${name}"
    local runtime="${SUITE_ROOT}/runtime_${name}"

    echo "[START][GPU ${gpu}] ${name} | category=${category} anchor=${anchor} motion=${motion} kalman=${kalman} gru=$((1-disable_gru)) forward=${forward_only} ms2=${ms2_enabled} grid=${ms2_grid} bw=${bandwidth}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_RUNTIME_DIR="${runtime}" \
    UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${SHARED_CACHE}" \
    JITTER_M=8 \
    UAVSAT_EXPERIMENT_ANCHOR="${anchor}" \
    UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
    UAVSAT_EXPERIMENT_MOTION="${motion}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY="${forward_only}" \
    MS2_ENABLED="${ms2_enabled}" \
    MS2_GRID_SIZE="${ms2_grid}" \
    MS2_BANDWIDTH_M="${bandwidth}" \
    EXPERIMENT_TAG="${name}" \
    EXPERIMENT_CATEGORY="${category}" \
    RUN_ALL_EXPERIMENTS=0 \
    bash "${SCRIPT_PATH}"
    echo "[DONE ][GPU ${gpu}] ${name}"
  }

  echo "============================================================================================================"
  echo "v39 architecture-focused paper experiment suite"
  echo "Fixed protocol: JITTER_M=8, 3 temporal frames, same trained checkpoints"
  echo "Ablation axes: module on/off, MS1 decoder, forward search, motion model, Kalman variance, MS2 grid, MS bandwidth"
  echo "No reference robustness/weight experiment is included."
  echo "output root: ${SUITE_ROOT}"
  echo "GPU plan: 0 / 5 / 6"
  echo "============================================================================================================"

  # Selected full method first; this also warms the shared feature cache.
  run_one 0 "full_model" softms quadratic learned 0 1 1 6 5.0 "module_ablation"

  # GPU 0: progressive architecture ablation, exactly following the overview chain.
  (
    run_one 0 "abl_ms1_only"             softms none      none    1 1 0 6 5.0 "module_ablation"
    run_one 0 "abl_ms1_gru"              softms quadratic none    0 1 0 6 5.0 "module_ablation"
    run_one 0 "abl_ms1_gru_kalman"       softms quadratic learned 0 1 0 6 5.0 "module_ablation"
    run_one 0 "design_kalman_fixed_var"  softms quadratic fixed   0 1 1 6 5.0 "kalman_design"
  ) & pid0=$!

  # GPU 5: method-design choices that correspond to real blocks in the architecture.
  (
    run_one 5 "design_ms1_weighted"       weighted_centroid quadratic learned 0 1 1 6 5.0 "ms1_decoder"
    run_one 5 "design_search_full6x6"     softms            quadratic learned 0 0 1 6 5.0 "candidate_search"
    run_one 5 "design_motion_none"        softms            none      learned 0 1 1 6 5.0 "motion_model"
    run_one 5 "design_motion_velocity"    softms            velocity  learned 0 1 1 6 5.0 "motion_model"
  ) & pid5=$!

  # GPU 6: genuine size/hyperparameter sensitivity for MS2 itself.
  (
    run_one 6 "sens_ms2_grid4x4"          softms quadratic learned 0 1 1 4 5.0 "ms2_window"
    run_one 6 "sens_ms2_grid8x8"          softms quadratic learned 0 1 1 8 5.0 "ms2_window"
    run_one 6 "sens_ms_bandwidth3"        softms quadratic learned 0 1 1 6 3.0 "meanshift_bandwidth"
    run_one 6 "sens_ms_bandwidth7"        softms quadratic learned 0 1 1 6 7.0 "meanshift_bandwidth"
  ) & pid6=$!

  status=0
  wait "${pid0}" || status=1
  wait "${pid5}" || status=1
  wait "${pid6}" || status=1
  if [[ "${status}" != "0" ]]; then
    echo "ERROR: at least one experiment failed. Check ${SUITE_ROOT}/*/*.log" >&2
    exit 3
  fi

  python3 - "${SUITE_ROOT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

suite = Path(sys.argv[1])
rows = []

def get(d, path, default=float("nan")):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def metric_row(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    b = d.get("route_B", {})
    c = d.get("route_C", {})
    nb, nc = 2276, 1258
    bm = float(b.get("MLE_m", math.nan))
    cm = float(c.get("MLE_m", math.nan))
    bc = (bm * nb + cm * nc) / float(nb + nc)
    disable_gru = bool(d.get("experiment_disable_gru", False))
    kalman = str(d.get("experiment_kalman", "learned"))
    ms2 = bool(d.get("MS2_Enabled", d.get("ms2_enabled", True)))
    forward = bool(d.get("experiment_forward_only", True))
    return {
        "Experiment": d.get("experiment_tag", p.parent.name),
        "Category": d.get("experiment_category", "-"),
        "MS1": "yes",
        "GRU": "no" if disable_gru else "yes",
        "Kalman": "no" if kalman == "none" else kalman,
        "MS2": "yes" if ms2 else "no",
        "MS1_decoder": d.get("experiment_anchor", "softms"),
        "Search": "forward 3x6" if forward else "full 6x6",
        "Motion": d.get("experiment_motion", "quadratic"),
        "MS2_grid": d.get("MS2_GridSize", d.get("ms2_grid_size", "-")) if ms2 else "-",
        "MS_bandwidth_m": get(d, ["ms2_hyperparameters", "bandwidth_m"], "-") if ms2 else "-",
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
    }

for p in sorted(suite.glob("*/robust_tracker_summary.json")):
    rows.append(metric_row(p))

main = next((r for r in rows if r["Experiment"] == "full_model"), None)
for r in rows:
    if main and math.isfinite(float(r["BC_weighted_MLE_m"])):
        r["Delta_vs_Full_pct"] = (
            float(r["BC_weighted_MLE_m"]) / float(main["BC_weighted_MLE_m"]) - 1.0
        ) * 100.0
    else:
        r["Delta_vs_Full_pct"] = math.nan

order = [
    "abl_ms1_only", "abl_ms1_gru", "abl_ms1_gru_kalman", "full_model",
    "design_ms1_weighted", "design_search_full6x6", "design_motion_none",
    "design_motion_velocity", "design_kalman_fixed_var", "sens_ms2_grid4x4",
    "sens_ms2_grid8x8", "sens_ms_bandwidth3", "sens_ms_bandwidth7",
]
rank = {name: i for i, name in enumerate(order)}
rows.sort(key=lambda r: rank.get(r["Experiment"], 999))

columns = [
    "Experiment", "Category", "MS1", "GRU", "Kalman", "MS2",
    "MS1_decoder", "Search", "Motion", "MS2_grid", "MS_bandwidth_m",
    "B_MLE_m", "C_MLE_m", "BC_weighted_MLE_m", "Delta_vs_Full_pct",
    "B_P90_m", "C_P90_m", "B_LSR5_pct", "C_LSR5_pct",
    "B_LSR15_pct", "C_LSR15_pct", "B_JumpRate_pct", "C_JumpRate_pct",
]

csv_path = suite / "experiment_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)

md_path = suite / "experiment_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# v39 Architecture Ablation Summary\n\n")
    f.write("| Experiment | MS1 | GRU | Kalman | MS2 | Decoder | Search | Motion | MS2 Grid | BW | B MLE | C MLE | B+C MLE | Delta vs Full |\n")
    f.write("|---|:---:|:---:|:---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        def fmt(v, n=3):
            try:
                x = float(v)
                if math.isnan(x):
                    return "-"
                return f"{x:.{n}f}"
            except (TypeError, ValueError):
                return str(v)
        f.write(
            f"| {r['Experiment']} | {r['MS1']} | {r['GRU']} | {r['Kalman']} | {r['MS2']} | "
            f"{r['MS1_decoder']} | {r['Search']} | {r['Motion']} | {r['MS2_grid']} | {fmt(r['MS_bandwidth_m'],1)} | "
            f"{fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | "
            f"{fmt(r['Delta_vs_Full_pct'],2)}% |\n"
        )

print(f"[TABLE] {csv_path}")
print(f"[TABLE] {md_path}")
PY

  echo "============================================================================================================"
  echo "ALL ARCHITECTURE-FOCUSED EXPERIMENTS COMPLETED"
  echo "Results: ${SUITE_ROOT}"
  echo "CSV: ${SUITE_ROOT}/experiment_summary.csv"
  echo "Markdown: ${SUITE_ROOT}/experiment_summary.md"
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

# Selected v39 MS2 implementation. These stay FIXED in the paper ablation suite;
# they are not treated as separate experiment axes.
export MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-4.0}"
export MS2_REFERENCE_SIGMA_M="${MS2_REFERENCE_SIGMA_M:-4.0}"
export MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-1.50}"
export MS2_REFERENCE_PRIOR_WEIGHT="${MS2_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-5.0}"
export MS2_ENABLED="${MS2_ENABLED:-1}"
export MS2_GRID_SIZE="${MS2_GRID_SIZE:-6}"

echo "============================================================================================================"
echo "v39 Direct FinalMS"
echo "flow: MS1 -> GRU -> Kalman -> MS2 -> FINAL"
echo "experiment: ${EXPERIMENT_TAG:-single_default}"
echo "category: ${EXPERIMENT_CATEGORY:-single}"
echo "GRU disabled: ${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}"
echo "Kalman mode: ${UAVSAT_EXPERIMENT_KALMAN:-learned}"
echo "MS1 decoder: ${UAVSAT_EXPERIMENT_ANCHOR:-softms}"
echo "forward-only search: ${UAVSAT_EXPERIMENT_FORWARD_ONLY:-1}"
echo "motion: ${UAVSAT_EXPERIMENT_MOTION:-quadratic}"
echo "MS2 enabled/grid/bandwidth: ${MS2_ENABLED}/${MS2_GRID_SIZE}/${MS2_BANDWIDTH_M}"
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
UAVSAT_EXPERIMENT_ANCHOR="${UAVSAT_EXPERIMENT_ANCHOR:-softms}" \
UAVSAT_EXPERIMENT_FRAME_COUNT="${UAVSAT_EXPERIMENT_FRAME_COUNT:-3}" \
UAVSAT_EXPERIMENT_MOTION="${UAVSAT_EXPERIMENT_MOTION:-quadratic}" \
UAVSAT_EXPERIMENT_KALMAN="${UAVSAT_EXPERIMENT_KALMAN:-learned}" \
UAVSAT_EXPERIMENT_DISABLE_GRU="${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" \
UAVSAT_EXPERIMENT_FORWARD_ONLY="${UAVSAT_EXPERIMENT_FORWARD_ONLY:-1}" \
python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${MODE}.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["experiment_tag"] = os.environ.get("EXPERIMENT_TAG", "single_default")
d["experiment_category"] = os.environ.get("EXPERIMENT_CATEGORY", "single")
d["experiment_jitter_m"] = float(os.environ.get("JITTER_M", "8"))
d["experiment_anchor"] = os.environ.get("UAVSAT_EXPERIMENT_ANCHOR", "softms")
d["experiment_motion"] = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "quadratic")
d["experiment_kalman"] = os.environ.get("UAVSAT_EXPERIMENT_KALMAN", "learned")
d["experiment_disable_gru"] = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "0") == "1"
d["experiment_forward_only"] = os.environ.get("UAVSAT_EXPERIMENT_FORWARD_ONLY", "1") == "1"
d["ms2_enabled"] = os.environ.get("MS2_ENABLED", "1") not in {"0", "false", "False", "no", "off"}
d["ms2_grid_size"] = int(os.environ.get("MS2_GRID_SIZE", "6"))
d["final_chain"] = "MS1 -> GRU -> Kalman -> MS2 -> Final"
d["second_kalman_update"] = "none"
d["final_decoder"] = "Soft MeanShift when MS2 is enabled; MS2 output is final"
d["ms2_search_center"] = "nearest permanent SAT lattice point to the single Kalman posterior"
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
