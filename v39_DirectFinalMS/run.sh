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
FINAL_ARCH="V39_GRU_Kalman_MS"

# Selected method after the pilot study:
# GRU -> Kalman(fixed measurement variance) -> MS(6x6, bandwidth 7m)
DEFAULT_KALMAN="${DEFAULT_KALMAN:-fixed}"
DEFAULT_MS_GRID="${DEFAULT_MS_GRID:-6}"
DEFAULT_MS_BANDWIDTH="${DEFAULT_MS_BANDWIDTH:-7.0}"

if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/experiments_${TS}}"
  SHARED_CACHE="${UAVSAT_SHARED_FEATURE_CACHE_DIR:-${ROOT}/output/feature_cache}"
  mkdir -p "${SUITE_ROOT}" "${SHARED_CACHE}"

  run_one() {
    local gpu="$1"
    local name="$2"
    local motion="$3"
    local kalman="$4"
    local disable_gru="$5"
    local ms_enabled="$6"
    local ms_grid="$7"
    local bandwidth="$8"
    local category="$9"
    local out="${SUITE_ROOT}/${name}"
    local runtime="${SUITE_ROOT}/runtime_${name}"

    echo "[START][GPU ${gpu}] ${name} | category=${category} gru=$((1-disable_gru)) kalman=${kalman} ms=${ms_enabled} motion=${motion} grid=${ms_grid} bw=${bandwidth}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_RUNTIME_DIR="${runtime}" \
    UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${SHARED_CACHE}" \
    JITTER_M=8 \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
    UAVSAT_EXPERIMENT_MOTION="${motion}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${ms_grid}" \
    MS_BANDWIDTH_M="${bandwidth}" \
    MS_LATENCY_WARMUP=30 \
    EXPERIMENT_TAG="${name}" \
    EXPERIMENT_CATEGORY="${category}" \
    RUN_ALL_EXPERIMENTS=0 \
    bash "${SCRIPT_PATH}"
    echo "[DONE ][GPU ${gpu}] ${name}"
  }

  echo "============================================================================================================"
  echo "v39 paper suite: GRU -> Kalman -> MS"
  echo "Selected default: Kalman=${DEFAULT_KALMAN}, MS=${DEFAULT_MS_GRID}x${DEFAULT_MS_GRID}, bandwidth=${DEFAULT_MS_BANDWIDTH}m"
  echo "MS latency = Kalman output -> final MS coordinate; first 30 frames excluded."
  echo "GPU plan: 0 / 5 / 6"
  echo "output root: ${SUITE_ROOT}"
  echo "============================================================================================================"

  # Run selected full model first to populate shared feature cache safely.
  run_one 0 "full_model" quadratic "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" "module_ablation"

  # GPU 0: progressive architecture ablation + learned-variance comparison.
  (
    run_one 0 "baseline_visual"          none      none    1 0 6 "${DEFAULT_MS_BANDWIDTH}" "module_ablation"
    run_one 0 "abl_gru_only"             quadratic none    0 0 6 "${DEFAULT_MS_BANDWIDTH}" "module_ablation"
    run_one 0 "abl_gru_kalman"           quadratic fixed   0 0 6 "${DEFAULT_MS_BANDWIDTH}" "module_ablation"
    run_one 0 "design_kalman_learned"    quadratic learned 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "kalman_design"
  ) & pid0=$!

  # GPU 5: GRU motion design + no-Kalman comparison + 4x4 efficiency point.
  (
    run_one 5 "design_motion_none"        none      fixed 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "gru_motion"
    run_one 5 "design_motion_velocity"    velocity  fixed 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "gru_motion"
    run_one 5 "design_kalman_none"        quadratic none  0 1 6 "${DEFAULT_MS_BANDWIDTH}" "kalman_design"
    run_one 5 "sens_ms_grid4x4"           quadratic fixed 0 1 4 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
  ) & pid5=$!

  # GPU 6: 8x8 efficiency point + bandwidth alternatives.
  (
    run_one 6 "sens_ms_grid8x8"           quadratic fixed 0 1 8 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
    run_one 6 "sens_ms_bandwidth3"        quadratic fixed 0 1 6 3.0 "meanshift_bandwidth"
    run_one 6 "sens_ms_bandwidth5"        quadratic fixed 0 1 6 5.0 "meanshift_bandwidth"
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
NB, NC = 2276, 1258
NALL = NB + NC


def weighted(b, c):
    try:
        return (float(b) * NB + float(c) * NC) / NALL
    except Exception:
        return float("nan")


def metric_row(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    b = d.get("route_B", {})
    c = d.get("route_C", {})
    bm = float(b.get("MLE_m", math.nan))
    cm = float(c.get("MLE_m", math.nan))
    b_lat = float(b.get("MS_LatencyMean_ms", 0.0))
    c_lat = float(c.get("MS_LatencyMean_ms", 0.0))
    ms_enabled = bool(d.get("MS_Enabled", d.get("ms_enabled", True)))
    bc_lat = weighted(b_lat, c_lat) if ms_enabled else 0.0
    return {
        "Experiment": d.get("experiment_tag", p.parent.name),
        "Category": d.get("experiment_category", "-"),
        "GRU": "no" if bool(d.get("experiment_disable_gru", False)) else "yes",
        "Kalman": "no" if str(d.get("experiment_kalman", "fixed")) == "none" else "yes",
        "Kalman_mode": str(d.get("experiment_kalman", "fixed")),
        "MS": "yes" if ms_enabled else "no",
        "Motion": d.get("experiment_motion", "quadratic"),
        "MS_grid": d.get("MS_GridSize", d.get("ms_grid_size", "-")) if ms_enabled else "-",
        "MS_bandwidth_m": d.get("ms_hyperparameters", {}).get("bandwidth_m", "-") if ms_enabled else "-",
        "B_MLE_m": bm,
        "C_MLE_m": cm,
        "BC_weighted_MLE_m": weighted(bm, cm),
        "B_P90_m": b.get("P90_m", math.nan),
        "C_P90_m": c.get("P90_m", math.nan),
        "B_LSR5_pct": b.get("LSR@5_pct", math.nan),
        "C_LSR5_pct": c.get("LSR@5_pct", math.nan),
        "B_LSR15_pct": b.get("LSR@15_pct", math.nan),
        "C_LSR15_pct": c.get("LSR@15_pct", math.nan),
        "B_JumpRate_pct": b.get("JumpRate_pct", math.nan),
        "C_JumpRate_pct": c.get("JumpRate_pct", math.nan),
        "B_MS_Latency_ms": b_lat,
        "C_MS_Latency_ms": c_lat,
        "BC_MS_Latency_ms": bc_lat,
        "MS_FPS": (1000.0 / bc_lat) if bc_lat > 0 else 0.0,
        "B_SpeedError": b.get("MeanSpeedError_m_per_frame", math.nan),
        "C_SpeedError": c.get("MeanSpeedError_m_per_frame", math.nan),
    }

for p in sorted(suite.glob("*/robust_tracker_summary.json")):
    rows.append(metric_row(p))

main = next((r for r in rows if r["Experiment"] == "full_model"), None)
for r in rows:
    if main and math.isfinite(float(r["BC_weighted_MLE_m"])):
        r["Delta_vs_Full_pct"] = (float(r["BC_weighted_MLE_m"]) / float(main["BC_weighted_MLE_m"]) - 1.0) * 100.0
    else:
        r["Delta_vs_Full_pct"] = math.nan

order = [
    "baseline_visual", "abl_gru_only", "abl_gru_kalman", "full_model",
    "design_motion_none", "design_motion_velocity",
    "design_kalman_none", "design_kalman_learned",
    "sens_ms_grid4x4", "sens_ms_grid8x8",
    "sens_ms_bandwidth3", "sens_ms_bandwidth5",
]
rank = {name: i for i, name in enumerate(order)}
rows.sort(key=lambda r: rank.get(r["Experiment"], 999))

columns = [
    "Experiment", "Category", "GRU", "Kalman", "Kalman_mode", "MS", "Motion",
    "MS_grid", "MS_bandwidth_m", "B_MLE_m", "C_MLE_m", "BC_weighted_MLE_m",
    "Delta_vs_Full_pct", "B_P90_m", "C_P90_m", "B_LSR5_pct", "C_LSR5_pct",
    "B_LSR15_pct", "C_LSR15_pct", "B_JumpRate_pct", "C_JumpRate_pct",
    "B_MS_Latency_ms", "C_MS_Latency_ms", "BC_MS_Latency_ms", "MS_FPS",
    "B_SpeedError", "C_SpeedError",
]

csv_path = suite / "experiment_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)

by_name = {r["Experiment"]: r for r in rows}

def fmt(v, n=3):
    try:
        x = float(v)
        return "-" if math.isnan(x) else f"{x:.{n}f}"
    except Exception:
        return str(v)

def row(name):
    return by_name[name]

md_path = suite / "paper_tables.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# v39 Paper Tables\n\n")
    f.write("## Table 1. Progressive architecture ablation\n\n")
    f.write("| Setting | GRU | Kalman | MS | B MLE | C MLE | B+C MLE | B LSR@5 | C LSR@5 | B/C Jump |\n")
    f.write("|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|\n")
    for name, label in [("baseline_visual","Baseline"),("abl_gru_only","+ GRU"),("abl_gru_kalman","+ GRU + Kalman"),("full_model","+ GRU + Kalman + MS")]:
        r=row(name)
        f.write(f"| {label} | {r['GRU']} | {r['Kalman']} | {r['MS']} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['B_LSR5_pct'],2)}% | {fmt(r['C_LSR5_pct'],2)}% | {fmt(r['B_JumpRate_pct'],3)}/{fmt(r['C_JumpRate_pct'],3)}% |\n")

    f.write("\n## Table 2. GRU motion-model design\n\n")
    f.write("| Motion | B MLE | C MLE | B+C MLE | B Speed MAE | C Speed MAE |\n|---|---:|---:|---:|---:|---:|\n")
    for name,label in [("design_motion_none","None"),("design_motion_velocity","Velocity"),("full_model","Quadratic")]:
        r=row(name); f.write(f"| {label} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['B_SpeedError'])} | {fmt(r['C_SpeedError'])} |\n")

    f.write("\n## Table 3. Kalman measurement design\n\n")
    f.write("| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |\n|---|---:|---:|---:|---:|\n")
    for name,label in [("design_kalman_none","No Kalman"),("design_kalman_learned","Learned variance"),("full_model","Fixed variance (selected)")]:
        r=row(name); f.write(f"| {label} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['B_JumpRate_pct'],3)}/{fmt(r['C_JumpRate_pct'],3)}% |\n")

    f.write("\n## Table 4. MS window accuracy-efficiency trade-off\n\n")
    f.write("| Window | Candidates | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for name,label,cands in [("sens_ms_grid4x4","4x4",16),("full_model","6x6",36),("sens_ms_grid8x8","8x8",64)]:
        r=row(name); f.write(f"| {label} | {cands} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['BC_MS_Latency_ms'])} | {fmt(r['MS_FPS'],1)} |\n")

    f.write("\n## Table 5. MeanShift bandwidth accuracy-efficiency trade-off\n\n")
    f.write("| Bandwidth | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |\n|---:|---:|---:|---:|---:|---:|\n")
    for name,label in [("sens_ms_bandwidth3","3 m"),("sens_ms_bandwidth5","5 m"),("full_model","7 m (selected)")]:
        r=row(name); f.write(f"| {label} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['BC_MS_Latency_ms'])} | {fmt(r['MS_FPS'],1)} |\n")

summary_md = suite / "experiment_summary.md"
summary_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
print(f"[TABLE] {csv_path}")
print(f"[TABLE] {md_path}")
PY

  echo "============================================================================================================"
  echo "ALL EXPERIMENTS COMPLETED"
  echo "Results: ${SUITE_ROOT}"
  echo "CSV: ${SUITE_ROOT}/experiment_summary.csv"
  echo "Paper tables: ${SUITE_ROOT}/paper_tables.md"
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
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
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

export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"
export MS_REFERENCE_SIGMA_M="${MS_REFERENCE_SIGMA_M:-4.0}"
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_PRIOR_WEIGHT="${MS_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS_BANDWIDTH_M="${MS_BANDWIDTH_M:-7.0}"
export MS_ENABLED="${MS_ENABLED:-1}"
export MS_GRID_SIZE="${MS_GRID_SIZE:-6}"
export MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}"

echo "============================================================================================================"
echo "v39 architecture: GRU -> Kalman Filter -> MS -> Final Position"
echo "experiment: ${EXPERIMENT_TAG:-single_default}"
echo "category: ${EXPERIMENT_CATEGORY:-single}"
echo "GRU disabled: ${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}"
echo "Kalman mode: ${UAVSAT_EXPERIMENT_KALMAN:-fixed}"
echo "motion: ${UAVSAT_EXPERIMENT_MOTION:-quadratic}"
echo "MS enabled/grid/bandwidth: ${MS_ENABLED}/${MS_GRID_SIZE}/${MS_BANDWIDTH_M}"
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
UAVSAT_EXPERIMENT_KALMAN="${UAVSAT_EXPERIMENT_KALMAN:-fixed}" \
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
d["experiment_motion"] = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "quadratic")
d["experiment_kalman"] = os.environ.get("UAVSAT_EXPERIMENT_KALMAN", "fixed")
d["experiment_disable_gru"] = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "0") == "1"
d["ms_enabled"] = os.environ.get("MS_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
d["ms_grid_size"] = int(os.environ.get("MS_GRID_SIZE", "6"))
d["final_chain"] = "GRU -> Kalman Filter -> MS -> Final Position"
d["final_decoder"] = "MeanShift when MS is enabled; MS output is the final position"
d["ms_search_center"] = "nearest permanent SAT lattice point to the single Kalman posterior"
d["ms_hyperparameters"] = {
    "kalman_sigma_m": float(os.environ.get("MS_KF_SIGMA_M", "4.0")),
    "reference_sigma_m": float(os.environ.get("MS_REFERENCE_SIGMA_M", "4.0")),
    "kalman_prior_weight": float(os.environ.get("MS_KF_PRIOR_WEIGHT", "1.50")),
    "reference_prior_weight": float(os.environ.get("MS_REFERENCE_PRIOR_WEIGHT", "2.50")),
    "bandwidth_m": float(os.environ.get("MS_BANDWIDTH_M", "7.0")),
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] result: ${OUT}/robust_tracker_summary.json"
