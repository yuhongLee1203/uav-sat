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

# Current selected operating point after the pilot runs.
# IMPORTANT: velocity means constant-velocity motion prediction. It does NOT
# mean a two-frame GRU. The GRU input remains three frames in every experiment.
DEFAULT_MOTION="${DEFAULT_MOTION:-velocity}"
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
  echo "ALL GRU experiments use 3 input frames."
  echo "Selected motion: constant velocity (${DEFAULT_MOTION})"
  echo "Selected Kalman: ${DEFAULT_KALMAN} variance"
  echo "Current operating point: MS ${DEFAULT_MS_GRID}x${DEFAULT_MS_GRID}, bandwidth ${DEFAULT_MS_BANDWIDTH} m"
  echo "Fair timing rule: every grid-size point is run sequentially on GPU 5."
  echo "Bandwidth 1..14 m is run sequentially on GPU 6; bandwidth is treated as an accuracy/smoothing parameter, not a runtime parameter."
  echo "GPU plan: 0=architecture/motion/Kalman, 5=MS grid, 6=MS bandwidth"
  echo "output root: ${SUITE_ROOT}"
  echo "============================================================================================================"

  # Full selected model first; safely warms the shared feature cache.
  run_one 0 "full_model" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" "module_ablation"

  # --------------------------------------------------------------------------
  # GPU 0: architecture + motion + Kalman. Same GPU for all related variants.
  # Table 1 intentionally starts at GRU because the paper architecture is
  # GRU -> Kalman -> MS; the fixed visual front-end is not an architecture row.
  # --------------------------------------------------------------------------
  (
    run_one 0 "abl_gru_only"             "${DEFAULT_MOTION}" none    0 0 6 "${DEFAULT_MS_BANDWIDTH}" "module_ablation"
    run_one 0 "abl_gru_kalman"           "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 0 6 "${DEFAULT_MS_BANDWIDTH}" "module_ablation"

    # Table 2: all use THREE image frames. Only the downstream motion equation changes.
    run_one 0 "design_motion_none"        none      "${DEFAULT_KALMAN}" 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "motion_model"
    run_one 0 "design_motion_acceleration" quadratic "${DEFAULT_KALMAN}" 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "motion_model"

    # Table 3: Kalman measurement design; selected fixed variance is full_model.
    run_one 0 "design_kalman_none"        "${DEFAULT_MOTION}" none    0 1 6 "${DEFAULT_MS_BANDWIDTH}" "kalman_design"
    run_one 0 "design_kalman_learned"     "${DEFAULT_MOTION}" learned 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "kalman_design"
  ) & pid0=$!

  # --------------------------------------------------------------------------
  # GPU 5: complete local-window sweep on ONE GPU for fair accuracy/latency.
  # Include 5x5 and 7x7; do not infer the 6x6 balance point from only 4/6/8.
  # --------------------------------------------------------------------------
  (
    run_one 5 "sens_ms_grid4x4" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 4 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
    run_one 5 "sens_ms_grid5x5" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 5 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
    run_one 5 "sens_ms_grid6x6" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 6 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
    run_one 5 "sens_ms_grid7x7" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 7 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
    run_one 5 "sens_ms_grid8x8" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 8 "${DEFAULT_MS_BANDWIDTH}" "ms_window"
  ) & pid5=$!

  # --------------------------------------------------------------------------
  # GPU 6: dense bandwidth sweep on ONE GPU.
  # SAT lattice stride is 32 px; at 0.14 m/px this is ~4.48 m between adjacent
  # candidate centers. 1..14 m covers <<1 lattice spacing through roughly the
  # center-to-edge scale of the 6x6 window, so the complete smoothing trend is visible.
  # --------------------------------------------------------------------------
  (
    for bw in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
      run_one 6 "sens_ms_bandwidth${bw}" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 6 "${bw}.0" "meanshift_bandwidth"
    done
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
        "Motion": d.get("experiment_motion", "velocity"),
        "MS_grid": d.get("MS_GridSize", d.get("ms_grid_size", "-")) if ms_enabled else "-",
        "MS_bandwidth_m": d.get("ms_hyperparameters", {}).get("bandwidth_m", "-") if ms_enabled else "-",
        "B_MLE_m": bm,
        "C_MLE_m": cm,
        "BC_weighted_MLE_m": weighted(bm, cm),
        "B_P90_m": b.get("P90_m", math.nan),
        "C_P90_m": c.get("P90_m", math.nan),
        "BC_weighted_P90_m": weighted(b.get("P90_m", math.nan), c.get("P90_m", math.nan)),
        "B_LSR5_pct": b.get("LSR@5_pct", math.nan),
        "C_LSR5_pct": c.get("LSR@5_pct", math.nan),
        "BC_weighted_LSR5_pct": weighted(b.get("LSR@5_pct", math.nan), c.get("LSR@5_pct", math.nan)),
        "B_LSR15_pct": b.get("LSR@15_pct", math.nan),
        "C_LSR15_pct": c.get("LSR@15_pct", math.nan),
        "B_JumpRate_pct": b.get("JumpRate_pct", math.nan),
        "C_JumpRate_pct": c.get("JumpRate_pct", math.nan),
        "B_MS_Latency_ms": b_lat,
        "C_MS_Latency_ms": c_lat,
        "BC_MS_Latency_ms": bc_lat,
        "MS_FPS": (1000.0 / bc_lat) if bc_lat > 0 else 0.0,
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
    "abl_gru_only", "abl_gru_kalman", "full_model",
    "design_motion_none", "design_motion_acceleration",
    "design_kalman_none", "design_kalman_learned",
    "sens_ms_grid4x4", "sens_ms_grid5x5", "sens_ms_grid6x6", "sens_ms_grid7x7", "sens_ms_grid8x8",
] + [f"sens_ms_bandwidth{i}" for i in range(1, 15)]
rank = {name: i for i, name in enumerate(order)}
rows.sort(key=lambda r: rank.get(r["Experiment"], 999))

columns = [
    "Experiment", "Category", "GRU", "Kalman", "Kalman_mode", "MS", "Motion",
    "MS_grid", "MS_bandwidth_m", "B_MLE_m", "C_MLE_m", "BC_weighted_MLE_m",
    "Delta_vs_Full_pct", "B_P90_m", "C_P90_m", "BC_weighted_P90_m",
    "B_LSR5_pct", "C_LSR5_pct", "BC_weighted_LSR5_pct",
    "B_LSR15_pct", "C_LSR15_pct", "B_JumpRate_pct", "C_JumpRate_pct",
    "B_MS_Latency_ms", "C_MS_Latency_ms", "BC_MS_Latency_ms", "MS_FPS",
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

# Transparent balance rule for Table 4:
# choose the LOWEST-LATENCY window among settings within 0.5% of the best B+C MLE.
grid_names = [f"sens_ms_grid{i}x{i}" for i in range(4, 9)]
grid_rows = [row(name) for name in grid_names]
best_grid_mle = min(float(r["BC_weighted_MLE_m"]) for r in grid_rows)
eligible_grids = [r for r in grid_rows if float(r["BC_weighted_MLE_m"]) <= best_grid_mle * 1.005]
selected_grid_row = min(eligible_grids, key=lambda r: float(r["BC_MS_Latency_ms"]))
selected_grid = int(selected_grid_row["MS_grid"])

# Bandwidth is not selected by latency because the operation count is unchanged.
bw_names = [f"sens_ms_bandwidth{i}" for i in range(1, 15)]
bw_rows = [row(name) for name in bw_names]
selected_bw_row = min(bw_rows, key=lambda r: float(r["BC_weighted_MLE_m"]))
selected_bw = float(selected_bw_row["MS_bandwidth_m"])

selection_path = suite / "selection_summary.json"
selection_path.write_text(json.dumps({
    "grid_selection_rule": "lowest MS latency among grids within 0.5% of the best B+C MLE",
    "selected_grid": selected_grid,
    "best_grid_mle_m": best_grid_mle,
    "bandwidth_selection_rule": "lowest B+C MLE; latency is not used because bandwidth does not change the MeanShift operation count",
    "selected_bandwidth_m": selected_bw,
    "selected_bandwidth_mle_m": float(selected_bw_row["BC_weighted_MLE_m"]),
    "candidate_spacing_note": "SAT stride 32 px; at 0.14 m/px adjacent candidate centers are approximately 4.48 m apart",
}, indent=2), encoding="utf-8")

md_path = suite / "paper_tables.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# v39 Paper Tables\n\n")
    f.write("All GRU variants use three UAV image frames. 'Constant Velocity' and 'Velocity + Acceleration' refer to the downstream motion equation, not the number of input images.\n\n")

    f.write("## Table 1. Progressive architecture ablation\n\n")
    f.write("| Setting | GRU | Kalman | MS | B MLE | C MLE | B+C MLE | B LSR@5 | C LSR@5 | B/C Jump |\n")
    f.write("|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|\n")
    for name, label in [("abl_gru_only","GRU"),("abl_gru_kalman","+ Kalman"),("full_model","+ MS")]:
        r=row(name)
        f.write(f"| {label} | {r['GRU']} | {r['Kalman']} | {r['MS']} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['B_LSR5_pct'],2)}% | {fmt(r['C_LSR5_pct'],2)}% | {fmt(r['B_JumpRate_pct'],3)}/{fmt(r['C_JumpRate_pct'],3)}% |\n")

    f.write("\n## Table 2. Motion prediction model (all use 3-frame GRU input)\n\n")
    f.write("| Motion prediction | Meaning | B MLE | C MLE | B+C MLE |\n|---|---|---:|---:|---:|\n")
    for name,label,meaning in [
        ("design_motion_none","No learned motion","Kalman keeps its own previous velocity"),
        ("full_model","Constant Velocity (selected)","GRU velocity; acceleration term is not used"),
        ("design_motion_acceleration","Velocity + Acceleration","GRU velocity plus acceleration term"),
    ]:
        r=row(name); f.write(f"| {label} | {meaning} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} |\n")

    f.write("\n## Table 3. Kalman measurement design\n\n")
    f.write("| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |\n|---|---:|---:|---:|---:|\n")
    for name,label in [("design_kalman_none","No Kalman"),("design_kalman_learned","Learned variance"),("full_model","Fixed variance (selected)")]:
        r=row(name); f.write(f"| {label} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['B_JumpRate_pct'],3)}/{fmt(r['C_JumpRate_pct'],3)}% |\n")

    f.write("\n## Table 4. MS local-window accuracy-efficiency trade-off\n\n")
    f.write("All rows are measured sequentially on GPU 5. Selection rule: lowest latency among settings within 0.5% of the best B+C MLE.\n\n")
    f.write("| Window | Candidates | B MLE | C MLE | B+C MLE | MS latency (ms) | MS FPS |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for size in range(4, 9):
        name=f"sens_ms_grid{size}x{size}"; r=row(name)
        label=f"{size}x{size}" + (" (selected)" if size == selected_grid else "")
        f.write(f"| {label} | {size*size} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['BC_MS_Latency_ms'])} | {fmt(r['MS_FPS'],1)} |\n")

    f.write("\n## Table 5. MeanShift bandwidth sensitivity\n\n")
    f.write("Adjacent SAT candidate centers are approximately 4.48 m apart. Bandwidth changes the spatial smoothing scale, not the number of MeanShift operations, so latency is intentionally omitted.\n\n")
    f.write("| Bandwidth | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |\n|---:|---:|---:|---:|---:|---:|\n")
    for bw in range(1, 15):
        r=row(f"sens_ms_bandwidth{bw}")
        label=f"{bw} m" + (" (best)" if abs(float(r['MS_bandwidth_m']) - selected_bw) < 1e-9 else "")
        f.write(f"| {label} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_weighted_MLE_m'])} | {fmt(r['BC_weighted_P90_m'])} | {fmt(r['BC_weighted_LSR5_pct'],2)}% |\n")

    f.write("\n## Automatic selection summary\n\n")
    f.write(f"- MS window selected by the predefined accuracy-efficiency rule: **{selected_grid}x{selected_grid}**.\n")
    f.write(f"- Best tested MeanShift bandwidth by B+C MLE: **{selected_bw:.0f} m**.\n")

summary_md = suite / "experiment_summary.md"
summary_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
print(f"[TABLE] {csv_path}")
print(f"[TABLE] {md_path}")
print(f"[SELECT] {selection_path}")
print(f"[SELECT] MS grid={selected_grid}x{selected_grid}, bandwidth={selected_bw:.0f} m")
PY

  echo "============================================================================================================"
  echo "ALL EXPERIMENTS COMPLETED"
  echo "Results: ${SUITE_ROOT}"
  echo "CSV: ${SUITE_ROOT}/experiment_summary.csv"
  echo "Paper tables: ${SUITE_ROOT}/paper_tables.md"
  echo "Selection: ${SUITE_ROOT}/selection_summary.json"
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
echo "GRU temporal input: 3 UAV frames"
echo "experiment: ${EXPERIMENT_TAG:-single_default}"
echo "category: ${EXPERIMENT_CATEGORY:-single}"
echo "GRU disabled: ${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}"
echo "Kalman mode: ${UAVSAT_EXPERIMENT_KALMAN:-fixed}"
echo "motion model: ${UAVSAT_EXPERIMENT_MOTION:-velocity}"
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
UAVSAT_EXPERIMENT_MOTION="${UAVSAT_EXPERIMENT_MOTION:-velocity}" \
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
d["experiment_motion"] = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "velocity")
d["experiment_kalman"] = os.environ.get("UAVSAT_EXPERIMENT_KALMAN", "fixed")
d["experiment_disable_gru"] = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "0") == "1"
d["experiment_frame_count"] = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "3"))
d["ms_enabled"] = os.environ.get("MS_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
d["ms_grid_size"] = int(os.environ.get("MS_GRID_SIZE", "6"))
d["final_chain"] = "GRU -> Kalman Filter -> MS -> Final Position"
d["motion_label"] = {
    "none": "No learned motion",
    "velocity": "Constant Velocity",
    "quadratic": "Velocity + Acceleration",
}.get(d["experiment_motion"], d["experiment_motion"])
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
