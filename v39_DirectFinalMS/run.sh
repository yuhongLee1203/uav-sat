#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SCRIPT_PATH="${ROOT}/run.sh"
BASE_SRC="${ROOT}/base_src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_wc_final}"
SRC="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_wc_final}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DENSE_REF_DIR="${UAVSAT_DENSE_ROUTE_REFERENCE_DIR:-${REPO_ROOT}/frame-reference-exp/references}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V39_WeightedCentroid_PreviousState_MobileNetV3_Forward3x6"
FINAL_ARCH="V39_WeightedCentroid_GRU_Kalman_MS"
DEFAULT_GRID="${DEFAULT_MS_GRID:-5}"
DEFAULT_BW="${DEFAULT_MS_BANDWIDTH:-7.0}"
DEFAULT_MOTION="velocity"
DEFAULT_KALMAN="fixed"

# -----------------------------------------------------------------------------
# Complete paper suite. Unique train-time configurations are retrained on Route A.
# Route B/C are evaluation only. No bandwidth tuning is performed on B/C.
# -----------------------------------------------------------------------------
if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/wc_experiments_${TS}}"
  SHARED_CACHE="${UAVSAT_SHARED_FEATURE_CACHE_DIR:-${ROOT}/output/feature_cache}"
  mkdir -p "${SUITE_ROOT}" "${SHARED_CACHE}"

  run_cfg() {
    local gpu="$1" name="$2" frame_count="$3" kalman="$4" disable_gru="$5"
    local ms_enabled="$6" grid="$7" category="$8" train_mode="$9"
    local ckpt_source="${10:-}" measure_ms="${11:-0}" measure_e2e="${12:-0}"
    local out="${SUITE_ROOT}/${name}"
    local runtime="${SUITE_ROOT}/runtime_${name}"
    echo "[START][GPU ${gpu}] ${name} frame=${frame_count} kalman=${kalman} gru=$((1-disable_gru)) ms=${ms_enabled} grid=${grid} mode=${train_mode}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_RUNTIME_DIR="${runtime}" \
    UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${SHARED_CACHE}" \
    UAVSAT_DENSE_ROUTE_REFERENCE_DIR="${DENSE_REF_DIR}" \
    UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frame_count}" \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_BW}" \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP=30 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP=30 \
    FORCE_RETRAIN_TEMPORAL="$([[ "${train_mode}" == "train" ]] && echo 1 || echo 0)" \
    UAVSAT_TEMPORAL_CKPT_SOURCE="${ckpt_source}" \
    EXPERIMENT_TAG="${name}" \
    EXPERIMENT_CATEGORY="${category}" \
    RUN_ALL_EXPERIMENTS=0 \
    bash "${SCRIPT_PATH}"
    echo "[DONE ][GPU ${gpu}] ${name}"
  }

  echo "============================================================================================================"
  echo "Weighted Centroid -> GRU -> Kalman -> ONE final MS"
  echo "Reference protocol: scheduled_route_reference (B/C current metric coordinate is not an inference input)"
  echo "Bandwidth fixed before B/C evaluation: ${DEFAULT_BW} m"
  echo "Selected operating-point window fixed before this suite: ${DEFAULT_GRID}x${DEFAULT_GRID}"
  echo "============================================================================================================"

  # 0) Canonical model first, alone. This retrains the 3-frame/fixed-Kalman GRU
  # under the Weighted-Centroid front-end and safely warms A/B/C feature caches.
  run_cfg 0 full_model 3 fixed 0 1 "${DEFAULT_GRID}" module_ablation train
  CANON_CKPT="${SUITE_ROOT}/full_model/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  [[ -s "${CANON_CKPT}" ]] || { echo "ERROR: canonical checkpoint missing" >&2; exit 20; }

  # 1) Unique training distributions. All start from scratch on Route A.
  # Same feature cache is read-only now, so these can safely use separate GPUs.
  (
    run_cfg 0 temporal_1frame 1 fixed 0 1 "${DEFAULT_GRID}" temporal_frames train
  ) & p0=$!
  (
    run_cfg 5 temporal_2frame 2 fixed 0 1 "${DEFAULT_GRID}" temporal_frames train
  ) & p5=$!
  (
    run_cfg 6 kalman_none 3 none 0 1 "${DEFAULT_GRID}" kalman_design train
    run_cfg 6 kalman_learned 3 learned 0 1 "${DEFAULT_GRID}" kalman_design train
  ) & p6=$!
  status=0
  wait "${p0}" || status=1
  wait "${p5}" || status=1
  wait "${p6}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: training/evaluation queue failed" >&2; exit 21; }

  NONE_CKPT="${SUITE_ROOT}/kalman_none/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  LEARNED_CKPT="${SUITE_ROOT}/kalman_learned/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  [[ -s "${NONE_CKPT}" && -s "${LEARNED_CKPT}" ]] || { echo "ERROR: ablation checkpoint missing" >&2; exit 22; }

  # 2) Architecture ablation, same GPU. No retraining where train distribution
  # is already represented by one of the checkpoints above.
  run_cfg 0 abl_wc_only 3 none 1 0 "${DEFAULT_GRID}" module_ablation eval
  run_cfg 0 abl_wc_gru 3 none 0 0 "${DEFAULT_GRID}" module_ablation eval "${NONE_CKPT}"
  run_cfg 0 abl_wc_gru_kalman 3 fixed 0 0 "${DEFAULT_GRID}" module_ablation eval "${CANON_CKPT}"

  # 3) Final-MS window sensitivity. Run ALONE and sequentially on GPU5 so the
  # stage latency is comparable. The canonical checkpoint is reused unchanged.
  for g in 4 5 6 7 8; do
    run_cfg 5 "sens_ms_grid${g}x${g}" 3 fixed 0 1 "${g}" ms_window eval "${CANON_CKPT}" 1 0
  done

  # 4) End-to-end runtime benchmark ALONE on GPU0. This includes backbone,
  # Weighted Centroid, GRU, Kalman and the single final MS; disk I/O and image
  # preprocessing are excluded by robust_tracker.py's timer definition.
  run_cfg 0 runtime_e2e 3 fixed 0 1 "${DEFAULT_GRID}" runtime eval "${CANON_CKPT}" 0 1

  # 5) Aggregate, audit, and create paper-ready tables.
  python3 - "${SUITE_ROOT}" "${DEFAULT_GRID}" "${DEFAULT_BW}" <<'PY'
import csv, json, math, sys
from pathlib import Path

suite = Path(sys.argv[1]); selected_grid = int(sys.argv[2]); fixed_bw = float(sys.argv[3])

def read(name):
    p = suite / name / "robust_tracker_summary.json"
    if not p.exists(): raise SystemExit(f"AUDIT FAILED: missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def route_n(d, route):
    p = Path(d[route].get("CSV", ""))
    if p.exists():
        with p.open("r", encoding="utf-8") as f: return max(sum(1 for _ in f)-1, 1)
    return 2276 if route == "route_B" else 1258

def weighted(d, key):
    nb, nc = route_n(d,"route_B"), route_n(d,"route_C")
    return (float(d["route_B"][key])*nb + float(d["route_C"][key])*nc)/(nb+nc)

def audit(name, d, expect_gru=True, expect_ms=True, frame=None, kalman=None):
    errors=[]
    if d.get("reference_protocol") != "scheduled_route_reference": errors.append("reference_protocol")
    if d.get("uses_gt_center_at_inference") is not False: errors.append("GT center used at inference")
    if d.get("experiment_anchor") != "weighted_centroid": errors.append("front decoder flag")
    for r in ("route_B","route_C"):
        rr=d.get(r,{})
        if rr.get("VisualObservationDecoder") != "posterior weighted centroid": errors.append(f"{r} decoder")
        if rr.get("OnlineMeanShiftCount") != (1 if expect_ms else 0): errors.append(f"{r} MeanShift count")
        if rr.get("StaticSatelliteProjectionCache") is not True: errors.append(f"{r} SAT cache")
        if expect_ms and rr.get("FinalMSReusesFrontUAVEmbedding") is not True: errors.append(f"{r} UAV reuse")
    if frame is not None and int(d.get("experiment_frame_count",-1)) != frame: errors.append("frame count")
    if kalman is not None and d.get("experiment_kalman") != kalman: errors.append("Kalman mode")
    if bool(d.get("experiment_disable_gru",False)) == expect_gru: errors.append("GRU switch")
    if errors: raise SystemExit(f"AUDIT FAILED [{name}]: {', '.join(errors)}")

names=["full_model","temporal_1frame","temporal_2frame","kalman_none","kalman_learned",
       "abl_wc_only","abl_wc_gru","abl_wc_gru_kalman"]+[f"sens_ms_grid{i}x{i}" for i in range(4,9)]+["runtime_e2e"]
data={n:read(n) for n in names}

audit("full_model",data["full_model"],True,True,3,"fixed")
audit("temporal_1frame",data["temporal_1frame"],True,True,1,"fixed")
audit("temporal_2frame",data["temporal_2frame"],True,True,2,"fixed")
audit("kalman_none",data["kalman_none"],True,True,3,"none")
audit("kalman_learned",data["kalman_learned"],True,True,3,"learned")
audit("abl_wc_only",data["abl_wc_only"],False,False,3,"none")
audit("abl_wc_gru",data["abl_wc_gru"],True,False,3,"none")
audit("abl_wc_gru_kalman",data["abl_wc_gru_kalman"],True,False,3,"fixed")
for i in range(4,9): audit(f"sens_ms_grid{i}x{i}",data[f"sens_ms_grid{i}x{i}"],True,True,3,"fixed")
audit("runtime_e2e",data["runtime_e2e"],True,True,3,"fixed")

# Check that trained frame-count checkpoints are distinct outputs, not legacy reuse.
for n in ["full_model","temporal_1frame","temporal_2frame","kalman_none","kalman_learned"]:
    ck=suite/n/"checkpoints"/"controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
    if not ck.exists() or ck.is_symlink(): raise SystemExit(f"AUDIT FAILED [{n}]: temporal checkpoint was not freshly trained")

rows=[]
for n,d in data.items():
    b,c=d["route_B"],d["route_C"]
    e2e_b=b.get("EndToEndTiming",{}); e2e_c=c.get("EndToEndTiming",{})
    rows.append({
      "Experiment":n,"Frames":d.get("experiment_frame_count"),"Kalman":d.get("experiment_kalman"),
      "GRU":"no" if d.get("experiment_disable_gru") else "yes","MS":"yes" if d.get("ms_enabled") else "no",
      "MS_grid":d.get("ms_grid_size","-"),"B_MLE_m":b["MLE_m"],"C_MLE_m":c["MLE_m"],
      "BC_MLE_m":weighted(d,"MLE_m"),"BC_P90_m":weighted(d,"P90_m"),"BC_LSR5_pct":weighted(d,"LSR@5_pct"),
      "B_Jump_pct":b["JumpRate_pct"],"C_Jump_pct":c["JumpRate_pct"],
      "MS_latency_ms":weighted(d,"MS_LatencyMean_ms") if float(b.get("MS_LatencyMean_ms",0))>0 else 0.0,
      "E2E_latency_ms":((float(e2e_b.get("mean_ms",0))*route_n(d,"route_B")+float(e2e_c.get("mean_ms",0))*route_n(d,"route_C"))/(route_n(d,"route_B")+route_n(d,"route_C"))) if e2e_b and e2e_c else 0.0,
    })

with (suite/"experiment_summary.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fmt=lambda x,n=3: f"{float(x):.{n}f}"
md=[]
md += ["# Weighted-Centroid Final Ablation Tables","",
       "Protocol: scheduled predefined route references; Route B/C current metric coordinates are used only for evaluation metrics.",""]
md += ["## Table 1. Progressive architecture ablation","",
       "| Setting | GRU | Kalman | MS | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |",
       "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|"]
for n,label in [("abl_wc_only","WC"),("abl_wc_gru","WC + GRU"),("abl_wc_gru_kalman","WC + GRU + Kalman"),("full_model","WC + GRU + Kalman + MS")]:
    d=data[n]; md.append(f"| {label} | {'no' if d.get('experiment_disable_gru') else 'yes'} | {'no' if d.get('experiment_kalman')=='none' else 'yes'} | {'yes' if d.get('ms_enabled') else 'no'} | {fmt(d['route_B']['MLE_m'])} | {fmt(d['route_C']['MLE_m'])} | {fmt(weighted(d,'MLE_m'))} | {fmt(weighted(d,'LSR@5_pct'),2)}% | {fmt(d['route_B']['JumpRate_pct'],3)}/{fmt(d['route_C']['JumpRate_pct'],3)}% |")
md += ["","## Table 2. Temporal input-frame ablation","","| UAV frames | B MLE | C MLE | B+C MLE | B+C P90 |","|---:|---:|---:|---:|---:|"]
for n,k in [("temporal_1frame",1),("temporal_2frame",2),("full_model",3)]:
    d=data[n]; md.append(f"| {k} | {fmt(d['route_B']['MLE_m'])} | {fmt(d['route_C']['MLE_m'])} | {fmt(weighted(d,'MLE_m'))} | {fmt(weighted(d,'P90_m'))} |")
md += ["","## Table 3. Kalman design","","| Kalman setting | B MLE | C MLE | B+C MLE | B/C Jump |","|---|---:|---:|---:|---:|"]
for n,label in [("kalman_none","No Kalman"),("kalman_learned","Learned variance"),("full_model","Fixed variance")]:
    d=data[n]; md.append(f"| {label} | {fmt(d['route_B']['MLE_m'])} | {fmt(d['route_C']['MLE_m'])} | {fmt(weighted(d,'MLE_m'))} | {fmt(d['route_B']['JumpRate_pct'],3)}/{fmt(d['route_C']['JumpRate_pct'],3)}% |")
md += ["","## Table 4. Final-MS local-window accuracy/efficiency","",
       "All five timing rows are executed sequentially on GPU 5 after all other jobs finish.","",
       "| Window | Candidates | B+C MLE | MS latency | MS FPS |","|---|---:|---:|---:|---:|"]
for g in range(4,9):
    d=data[f"sens_ms_grid{g}x{g}"]; lat=weighted(d,"MS_LatencyMean_ms"); tag=" (pre-fixed)" if g==selected_grid else ""
    md.append(f"| {g}x{g}{tag} | {g*g} | {fmt(weighted(d,'MLE_m'))} | {fmt(lat)} ms | {fmt(1000/lat,1)} |")
r=data["runtime_e2e"]; eb=r["route_B"]["EndToEndTiming"]; ec=r["route_C"]["EndToEndTiming"]
nb,nc=route_n(r,"route_B"),route_n(r,"route_C"); e2e=(eb["mean_ms"]*nb+ec["mean_ms"]*nc)/(nb+nc)
md += ["","## Table 5. End-to-end online runtime","",
       "Timing definition: prepared UAV tensor -> backbone -> Weighted Centroid -> GRU -> Kalman -> one final MS -> XY.","",
       f"- Route B: {fmt(eb['mean_ms'])} ms, {fmt(eb['fps'],1)} FPS",
       f"- Route C: {fmt(ec['mean_ms'])} ms, {fmt(ec['fps'],1)} FPS",
       f"- Weighted B+C: **{fmt(e2e)} ms, {fmt(1000/e2e,1)} FPS**",
       "",f"Fixed MeanShift bandwidth for all formal B/C experiments: **{fixed_bw:g} m** (not tuned on B/C in this suite)."]
(suite/"paper_tables.md").write_text("\n".join(md)+"\n",encoding="utf-8")
(suite/"audit_report.json").write_text(json.dumps({"status":"PASS","architecture":"Weighted Centroid -> GRU -> Kalman -> one final MS","reference_protocol":"scheduled_route_reference","selected_grid_prefixed":selected_grid,"bandwidth_m_fixed":fixed_bw,"fresh_temporal_trainings":["full_model","temporal_1frame","temporal_2frame","kalman_none","kalman_learned"]},indent=2),encoding="utf-8")
print("AUDIT PASS")
print(suite/"paper_tables.md")
PY

  echo "============================================================================================================"
  echo "ALL FORMAL ABLATIONS COMPLETED + AUDIT PASS"
  echo "Results: ${SUITE_ROOT}"
  echo "Tables : ${SUITE_ROOT}/paper_tables.md"
  echo "Audit  : ${SUITE_ROOT}/audit_report.json"
  echo "============================================================================================================"
  exit 0
fi

# -----------------------------------------------------------------------------
# Single train/eval runner used by the suite above.
# -----------------------------------------------------------------------------
VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
LOCAL_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LATEST_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing route ${route}" >&2; exit 2; }
  [[ -s "${DENSE_REF_DIR}/${route}.npz" ]] || { echo "ERROR: missing scheduled reference ${DENSE_REF_DIR}/${route}.npz" >&2; exit 2; }
done

rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
cp -a "${BASE_SRC}/." "${SRC}/"
python3 "${ROOT}/patch_direct_finalms.py" "${SRC}/robust_tracker.py" "${SRC}/visual_localizer.py" "${SRC}/config.py"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

MODE="eval"
if [[ "${FORCE_RETRAIN_TEMPORAL:-0}" == "1" ]]; then
  rm -f "${LOCAL_TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
  MODE="train_eval"
elif [[ -n "${UAVSAT_TEMPORAL_CKPT_SOURCE:-}" ]]; then
  [[ -s "${UAVSAT_TEMPORAL_CKPT_SOURCE}" ]] || { echo "ERROR: checkpoint source missing: ${UAVSAT_TEMPORAL_CKPT_SOURCE}" >&2; exit 3; }
  rm -f "${LOCAL_TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
  ln -s "${UAVSAT_TEMPORAL_CKPT_SOURCE}" "${LOCAL_TEMPORAL_CKPT}"
elif [[ "${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" == "1" ]]; then
  MODE="eval"
elif [[ ! -s "${LOCAL_TEMPORAL_CKPT}" ]]; then
  MODE="train_eval"
fi

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"
export MS_REFERENCE_SIGMA_M="${MS_REFERENCE_SIGMA_M:-4.0}"
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_PRIOR_WEIGHT="${MS_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS_BANDWIDTH_M="${MS_BANDWIDTH_M:-${DEFAULT_BW}}"
export MS_ENABLED="${MS_ENABLED:-1}"
export MS_GRID_SIZE="${MS_GRID_SIZE:-${DEFAULT_GRID}}"
export MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}"
export MS_MEASURE_LATENCY="${MS_MEASURE_LATENCY:-0}"

cd "${SRC}"
ARGS=(--mode "${MODE}" --reuse-visual --jitter-m 8)
if [[ "${MODE}" == "train_eval" ]]; then
  ARGS+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
fi

UAVSAT_DEVICE="${DEVICE}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_DENSE_ROUTE_REFERENCE_DIR="${DENSE_REF_DIR}" \
UAVSAT_BACKBONE="${BACKBONE}" \
UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
UAVSAT_REFERENCE_PROTOCOL="${UAVSAT_REFERENCE_PROTOCOL:-scheduled_route_reference}" \
UAVSAT_EXPERIMENT_ANCHOR="${UAVSAT_EXPERIMENT_ANCHOR:-weighted_centroid}" \
UAVSAT_EXPERIMENT_FRAME_COUNT="${UAVSAT_EXPERIMENT_FRAME_COUNT:-3}" \
UAVSAT_EXPERIMENT_MOTION="${UAVSAT_EXPERIMENT_MOTION:-velocity}" \
UAVSAT_EXPERIMENT_KALMAN="${UAVSAT_EXPERIMENT_KALMAN:-fixed}" \
UAVSAT_EXPERIMENT_DISABLE_GRU="${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" \
UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
UAVSAT_MEASURE_LATENCY="${UAVSAT_MEASURE_LATENCY:-0}" \
UAVSAT_LATENCY_WARMUP="${UAVSAT_LATENCY_WARMUP:-30}" \
python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${MODE}.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding="utf-8"))
d["architecture"]=sys.argv[2]
d["experiment_tag"]=os.environ.get("EXPERIMENT_TAG","single")
d["experiment_category"]=os.environ.get("EXPERIMENT_CATEGORY","single")
d["experiment_motion"]=os.environ.get("UAVSAT_EXPERIMENT_MOTION","velocity")
d["experiment_kalman"]=os.environ.get("UAVSAT_EXPERIMENT_KALMAN","fixed")
d["experiment_disable_gru"]=os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU","0")=="1"
d["experiment_frame_count"]=int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT","3"))
d["ms_enabled"]=os.environ.get("MS_ENABLED","1").lower() not in {"0","false","no","off"}
d["ms_grid_size"]=int(os.environ.get("MS_GRID_SIZE","5"))
d["final_chain"]="Weighted Centroid -> GRU -> Kalman Filter -> MS -> Final Position"
d["ms_hyperparameters"]={"bandwidth_m":float(os.environ.get("MS_BANDWIDTH_M","7.0")),"kalman_sigma_m":float(os.environ.get("MS_KF_SIGMA_M","4.0")),"reference_sigma_m":float(os.environ.get("MS_REFERENCE_SIGMA_M","4.0")),"kalman_prior_weight":float(os.environ.get("MS_KF_PRIOR_WEIGHT","1.5")),"reference_prior_weight":float(os.environ.get("MS_REFERENCE_PRIOR_WEIGHT","2.5"))}
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding="utf-8")
PY

echo "[DONE] ${OUT}/robust_tracker_summary.json"
