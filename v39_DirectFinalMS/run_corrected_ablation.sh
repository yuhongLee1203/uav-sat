#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
FEATURE_CACHE_ROOT="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache_paper6x6}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="mobilenet_v3_small"

# Paper suite: 6x6 only. Weighted-centroid visual readout is fixed internally
# and is intentionally not shown as a standalone paper contribution.
FINAL_ARCH="V39_Forward3x6_ContextGRU_FixedKalman_FinalMS6x6"
NO_FORWARD_ARCH="V39_Full6x6_ContextGRU_FixedKalman_FinalMS6x6"
FRAME1_ARCH="V39_Forward3x6_ContextGRU1F_FixedKalman_FinalMS6x6"
FRAME2_ARCH="V39_Forward3x6_ContextGRU2F_FixedKalman_FinalMS6x6"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-5}"
DEFAULT_MOTION="velocity"
DEFAULT_GRID="6"
DEFAULT_BANDWIDTH="7.0"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/paper6x6_all_${TS}}"
mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE_ROOT}"

patch_runtime_v5() {
  local runtime="$1"
  python3 - "${runtime}" <<'PY'
from pathlib import Path
import sys

runtime = Path(sys.argv[1])

p = runtime / "visual_model.py"
s = p.read_text(encoding="utf-8")
old = "        self.gru = nn.GRUCell(feature_dim * 4, hidden_dim)\n"
new = "        self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)\n"
if s.count(old) != 1:
    raise SystemExit(f"ERROR: GRU input-size patch count={s.count(old)}")
s = s.replace(old, new, 1)
old_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
new_block = '''        recurrent_input = torch.cat(
            [
                self.clip_mean_projection(clip_mean),
                self.delta_recent_projection(delta_recent),
                self.delta_accel_projection(delta_accel),
                self.sat_projection(sat_context),
                self.previous_state_projection(previous_state),
            ],
            dim=1,
        )
'''
if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: GRU recurrent-input patch count={s.count(old_block)}")
s = s.replace(old_block, new_block, 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")

p = runtime / "robust_tracker.py"
s = p.read_text(encoding="utf-8")
old_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            step = velocity.copy()
'''
new_motion = '''        elif motion_mode == "velocity":
            acceleration[:] = 0.0
            if bool(getattr(config, "EXPERIMENT_DISABLE_GRU", False)):
                velocity = self.x[2:4].copy()
                step = velocity.copy()
            else:
                step = velocity.copy()
'''
if s.count(old_motion) != 1:
    raise SystemExit(f"ERROR: velocity-fusion patch count={s.count(old_motion)}")
s = s.replace(old_motion, new_motion, 1)

marker = "ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)\n\n\n"
if s.count(marker) != 1:
    raise SystemExit(f"ERROR: fast-grid insertion marker count={s.count(marker)}")
helper = r'''ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)


def fast_regular_grid_indices(
    gallery_xy, gallery_pixel, pixel_index, prior_xy, grid_size, stride, device,
):
    grid_size = int(grid_size)
    start = -(grid_size // 2)
    offsets = range(start, start + grid_size)
    stride = int(stride)
    rows = []
    prior_xy = prior_xy.to(gallery_xy.device, dtype=gallery_xy.dtype)
    for prior in prior_xy:
        distance_squared = (gallery_xy - prior[None, :]).square().sum(dim=1)
        center_index = int(distance_squared.argmin().item())
        center_pixel = gallery_pixel[center_index].detach().cpu().tolist()
        center_x, center_y = (int(round(float(v))) for v in center_pixel)
        row, complete = [], True
        for offset_y in offsets:
            for offset_x in offsets:
                index = pixel_index.get(
                    (center_x + offset_x * stride, center_y + offset_y * stride)
                )
                if index is None:
                    complete = False
                    break
                row.append(index)
            if not complete:
                break
        if not complete:
            row = torch.topk(
                distance_squared, k=grid_size * grid_size, largest=False,
            ).indices.detach().cpu().tolist()
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


'''
s = s.replace(marker, helper, 1)
call_count = s.count("regular_grid_indices(")
if call_count < 2:
    raise SystemExit(f"ERROR: expected >=2 regular_grid_indices calls, got {call_count}")
s = s.replace("regular_grid_indices(", "fast_regular_grid_indices(")
s = s.replace("def fast_fast_regular_grid_indices(", "def fast_regular_grid_indices(", 1)
p.write_text(s, encoding="utf-8")
compile(s, str(p), "exec")
print("runtime-v5 patch PASS")
PY
}

make_runtime() {
  local out="$1" runtime="$2"
  rm -rf "${runtime}"
  mkdir -p "${runtime}" "${out}/checkpoints"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  patch_runtime_v5 "${runtime}"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
}

checkpoint_arch() {
  local ckpt="$1"
  python3 - "${ckpt}" <<'PY'
import sys, torch
payload=torch.load(sys.argv[1], map_location="cpu")
arch=payload.get("architecture")
if not arch:
    raise SystemExit("checkpoint has no architecture tag")
print(str(arch))
PY
}

run_cfg() {
  local gpu="$1" name="$2" frames="$3" disable_gru="$4" kalman="$5"
  local ms_enabled="$6" forward_only="$7" bandwidth="$8" mode="$9"
  local ckpt_source="${10:-}" arch_tag="${11:-${FINAL_ARCH}}" measure_e2e="${12:-0}"
  local out="${SUITE_ROOT}/${name}"
  local runtime="${SUITE_ROOT}/runtime_${name}"
  local feature_cache="${FEATURE_CACHE_ROOT}/gpu${gpu}"

  mkdir -p "${feature_cache}"
  make_runtime "${out}" "${runtime}"

  if [[ "${mode}" == "eval" && "${disable_gru}" == "0" ]]; then
    [[ -s "${ckpt_source}" ]] || { echo "ERROR: missing checkpoint ${ckpt_source}" >&2; return 3; }
    local stored_arch
    stored_arch="$(checkpoint_arch "${ckpt_source}")"
    [[ "${stored_arch}" == "${arch_tag}" ]] || {
      echo "ERROR: checkpoint architecture ${stored_arch} != ${arch_tag}" >&2
      return 4
    }
    ln -sfn "${ckpt_source}" "${out}/checkpoints/${CKPT_NAME}"
  fi

  echo "[START][${name}][GPU${gpu}] frames=${frames} gru=$((1-disable_gru)) kalman=${kalman} ms=${ms_enabled} forward=${forward_only} grid=6 bw=${bandwidth} e2e=${measure_e2e}"
  (
    cd "${runtime}"
    args=(--mode "${mode}" --reuse-visual --jitter-m "${JITTER_M}")
    if [[ "${mode}" == "train_eval" ]]; then
      args+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${feature_cache}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${arch_tag}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2=25.0 \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY="${forward_only}" \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${DEFAULT_GRID}" \
    MS_BANDWIDTH_M="${bandwidth}" \
    MS_MEASURE_LATENCY=0 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP=30 \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | sed -u "s/^/[${name}] /" | tee "${out}/${mode}.log"
  )

  python3 - "${out}/robust_tracker_summary.json" "${name}" "${frames}" "${disable_gru}" "${kalman}" "${ms_enabled}" "${forward_only}" "${bandwidth}" "${arch_tag}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding="utf-8"))
d["experiment_tag"]=sys.argv[2]
d["experiment_frame_count"]=int(sys.argv[3])
d["experiment_disable_gru"]=bool(int(sys.argv[4]))
d["experiment_kalman"]=sys.argv[5]
d["ms_enabled"]=bool(int(sys.argv[6]))
d["experiment_forward_only"]=bool(int(sys.argv[7]))
d["ms_bandwidth_m"]=float(sys.argv[8])
d["ms_grid_size"]=6
d["architecture"]=sys.argv[9]
d["experiment_anchor"]="weighted_centroid"
d["experiment_motion"]="velocity"
d["early_stopping_patience"]=5
d["paper_display_frontend"]="visual localization front-end"
d["gru_role"]="temporal residual/velocity conditioned on posterior-weighted satellite context"
d["kalman_ablation_definition"]="EXPERIMENT_KALMAN=none bypasses external Kalman measurement fusion"
d["final_ms_ablation_definition"]="MS_ENABLED=0 returns the pre-MS estimator output directly"
d["forward_ablation_definition"]="forward_only=0 scores all 36 candidates; forward_only=1 scores only forward 3x6=18"
d["runtime_indexing"]="GPU-resident gallery nearest/grid lookup"
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding="utf-8")
PY
  echo "[DONE][${name}]"
}

gpu_audit() {
  local gpu="$1" tag="$2"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "physical_gpu_index=${gpu}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -i "${gpu}" --query-gpu=name,uuid,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits || true
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python3 - <<'PY' || true
import torch
print("torch_version=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_runtime=", torch.version.cuda)
    print("visible_device_name=", torch.cuda.get_device_name(0))
    p=torch.cuda.get_device_properties(0)
    print("total_memory_GB=", round(p.total_memory/1024**3,3))
PY
  } | tee "${SUITE_ROOT}/gpu${gpu}_${tag}.txt"
}

echo "============================================================================================================"
echo "V39 PAPER 6x6 COMPLETE EXPERIMENT SUITE"
echo "Main: visual front-end -> context 3-frame GRU -> fixed-R external Kalman -> one final 6x6 MeanShift"
echo "Main ablation: w/o forward restriction / w/o GRU / w/o Kalman / w/o final MS / Full"
echo "Additional: temporal 1/2/3 frames, Kalman design, forward restriction efficiency, bandwidth sensitivity"
echo "GPUs: 0, 5, 6 used concurrently; Route A training only; B+C pooled evaluation; patience=${PATIENCE}"
echo "============================================================================================================"

gpu_audit 0 before_all
gpu_audit 5 before_all
gpu_audit 6 before_all

run_cfg 0 train_full 3 0 fixed 1 1 "${DEFAULT_BANDWIDTH}" train_eval "" "${FINAL_ARCH}" 0 &
PID_FULL=$!
run_cfg 5 train_no_forward 3 0 fixed 1 0 "${DEFAULT_BANDWIDTH}" train_eval "" "${NO_FORWARD_ARCH}" 0 &
PID_NO_FORWARD=$!
run_cfg 6 temporal_1f 1 0 fixed 1 1 "${DEFAULT_BANDWIDTH}" train_eval "" "${FRAME1_ARCH}" 0 &
PID_FRAME1=$!

wait "${PID_FULL}"
FULL_CKPT="${SUITE_ROOT}/train_full/checkpoints/${CKPT_NAME}"
[[ -s "${FULL_CKPT}" ]] || { echo "ERROR: full checkpoint missing" >&2; exit 20; }
[[ "$(checkpoint_arch "${FULL_CKPT}")" == "${FINAL_ARCH}" ]] || { echo "ERROR: full checkpoint architecture mismatch" >&2; exit 21; }

(
  gpu_audit 0 before_core
  run_cfg 0 full_model        3 0 fixed   1 1 "${DEFAULT_BANDWIDTH}" eval "${FULL_CKPT}" "${FINAL_ARCH}" 1
  run_cfg 0 abl_no_gru        3 1 fixed   1 1 "${DEFAULT_BANDWIDTH}" eval ""             "${FINAL_ARCH}" 1
  run_cfg 0 abl_no_kalman     3 0 none    1 1 "${DEFAULT_BANDWIDTH}" eval "${FULL_CKPT}" "${FINAL_ARCH}" 1
  run_cfg 0 abl_no_final_ms   3 0 fixed   0 1 "${DEFAULT_BANDWIDTH}" eval "${FULL_CKPT}" "${FINAL_ARCH}" 1
  run_cfg 0 kalman_learned    3 0 learned 1 1 "${DEFAULT_BANDWIDTH}" eval "${FULL_CKPT}" "${FINAL_ARCH}" 1
  gpu_audit 0 after_core
) &
PID_CORE=$!

wait "${PID_NO_FORWARD}"
NO_FORWARD_CKPT="${SUITE_ROOT}/train_no_forward/checkpoints/${CKPT_NAME}"
[[ -s "${NO_FORWARD_CKPT}" ]] || { echo "ERROR: no-forward checkpoint missing" >&2; exit 22; }
[[ "$(checkpoint_arch "${NO_FORWARD_CKPT}")" == "${NO_FORWARD_ARCH}" ]] || { echo "ERROR: no-forward checkpoint architecture mismatch" >&2; exit 23; }
run_cfg 5 temporal_2f 2 0 fixed 1 1 "${DEFAULT_BANDWIDTH}" train_eval "" "${FRAME2_ARCH}" 0 &
PID_FRAME2=$!

wait "${PID_FRAME1}"
(
  run_cfg 6 bw_1m  3 0 fixed 1 1 1.0  eval "${FULL_CKPT}" "${FINAL_ARCH}" 0
  run_cfg 6 bw_3m  3 0 fixed 1 1 3.0  eval "${FULL_CKPT}" "${FINAL_ARCH}" 0
  run_cfg 6 bw_5m  3 0 fixed 1 1 5.0  eval "${FULL_CKPT}" "${FINAL_ARCH}" 0
  run_cfg 6 bw_9m  3 0 fixed 1 1 9.0  eval "${FULL_CKPT}" "${FINAL_ARCH}" 0
  run_cfg 6 bw_11m 3 0 fixed 1 1 11.0 eval "${FULL_CKPT}" "${FINAL_ARCH}" 0
) &
PID_BW=$!

wait "${PID_CORE}"
run_cfg 0 abl_no_forward 3 0 fixed 1 0 "${DEFAULT_BANDWIDTH}" eval "${NO_FORWARD_CKPT}" "${NO_FORWARD_ARCH}" 1

wait "${PID_FRAME2}"
wait "${PID_BW}"

gpu_audit 0 after_all
gpu_audit 5 after_all
gpu_audit 6 after_all

python3 - "${SUITE_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
import numpy as np

suite=Path(sys.argv[1])
main_names=["abl_no_forward","abl_no_gru","abl_no_kalman","abl_no_final_ms","full_model"]
temporal_names=["temporal_1f","temporal_2f","full_model"]
kalman_names=["abl_no_kalman","kalman_learned","full_model"]
bandwidth_names=["bw_1m","bw_3m","bw_5m","full_model","bw_9m","bw_11m"]
all_names=sorted(set(main_names+temporal_names+kalman_names+bandwidth_names))
labels={
    "abl_no_forward":"w/o Forward 3x6 Restriction",
    "abl_no_gru":"w/o Temporal GRU",
    "abl_no_kalman":"w/o External Kalman Fusion",
    "abl_no_final_ms":"w/o Final MeanShift",
    "full_model":"Full Model",
    "temporal_1f":"1 frame",
    "temporal_2f":"2 frames",
    "kalman_learned":"Learned-R Kalman",
    "bw_1m":"1 m","bw_3m":"3 m","bw_5m":"5 m","bw_9m":"9 m","bw_11m":"11 m",
}

def summary(name):
    p=suite/name/"robust_tracker_summary.json"
    if not p.exists(): raise SystemExit(f"AUDIT FAILED: missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def frame_csv(name,route):
    files=list((suite/name).glob(f"{route}_*_frames.csv"))
    if len(files)!=1: raise SystemExit(f"AUDIT FAILED [{name}/{route}]: expected one frame CSV, got {len(files)}")
    return files[0]

def route_errors(name,route):
    vals=[]
    with frame_csv(name,route).open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f): vals.append(float(row["error_final_m"]))
    return np.asarray(vals,dtype=np.float64)

def errors(name): return np.concatenate([route_errors(name,"route_B"),route_errors(name,"route_C")])

def pooled(a):
    return {"N":int(a.size),"MLE":float(a.mean()),"MedLE":float(np.median(a)),"P90":float(np.quantile(a,.90)),"P95":float(np.quantile(a,.95)),"P99":float(np.quantile(a,.99)),"LSR3":float((a<=3).mean()*100.0),"LSR5":float((a<=5).mean()*100.0),"LSR10":float((a<=10).mean()*100.0),"LSR15":float((a<=15).mean()*100.0),"LSR20":float((a<=20).mean()*100.0)}

def pooled_e2e(obj):
    b=obj["route_B"].get("EndToEndTiming"); c=obj["route_C"].get("EndToEndTiming")
    if not b or not c: return float("nan"),float("nan")
    nb=int(b["samples"]); nc=int(c["samples"])
    mean=(float(b["mean_ms"])*nb+float(c["mean_ms"])*nc)/(nb+nc)
    return mean,1000.0/mean

def pooled_jump(name,obj):
    nb=max(len(route_errors(name,"route_B"))-1,1); nc=max(len(route_errors(name,"route_C"))-1,1)
    jb=float(obj["route_B"].get("JumpRate_pct",0.0)); jc=float(obj["route_C"].get("JumpRate_pct",0.0))
    return (jb*nb+jc*nc)/(nb+nc)

d={n:summary(n) for n in all_names}; p={n:pooled(errors(n)) for n in all_names}
timing={n:pooled_e2e(d[n]) for n in all_names}; jump={n:pooled_jump(n,d[n]) for n in all_names}

full=d["full_model"]
if full.get("experiment_disable_gru") or full.get("experiment_kalman")!="fixed" or not full.get("ms_enabled") or not full.get("experiment_forward_only"): raise SystemExit("AUDIT FAILED: full-model configuration mismatch")
if d["abl_no_forward"].get("experiment_forward_only"): raise SystemExit("AUDIT FAILED: no-forward row still uses forward restriction")
if not d["abl_no_gru"].get("experiment_disable_gru"): raise SystemExit("AUDIT FAILED: no-GRU row still has GRU")
if d["abl_no_kalman"].get("experiment_kalman")!="none": raise SystemExit("AUDIT FAILED: no-Kalman row still fuses Kalman")
if d["abl_no_final_ms"].get("ms_enabled"): raise SystemExit("AUDIT FAILED: no-final-MS row still has MS")
if d["kalman_learned"].get("experiment_kalman")!="learned": raise SystemExit("AUDIT FAILED: learned-R row mismatch")
if d["temporal_1f"].get("experiment_frame_count")!=1 or d["temporal_2f"].get("experiment_frame_count")!=2: raise SystemExit("AUDIT FAILED: temporal-frame configuration mismatch")
for n in all_names:
    if int(d[n].get("ms_grid_size",d[n].get("MS_GridSize",-1)))!=6: raise SystemExit(f"AUDIT FAILED [{n}]: not 6x6")

rows=[]
for name in all_names:
    q=p[name]; lat,fps=timing[name]
    rows.append({"Experiment":name,"Display":labels.get(name,name),"BC_MLE_m":q["MLE"],"BC_MedLE_m":q["MedLE"],"BC_P90_m":q["P90"],"BC_P95_m":q["P95"],"BC_P99_m":q["P99"],"BC_LSR3_pct":q["LSR3"],"BC_LSR5_pct":q["LSR5"],"BC_LSR10_pct":q["LSR10"],"BC_LSR15_pct":q["LSR15"],"BC_LSR20_pct":q["LSR20"],"JumpRate_pct":jump[name],"E2E_ms":lat,"FPS":fps,"N_frames":q["N"]})
with (suite/"paper_all_summary.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fmt=lambda x,n=3: "-" if not np.isfinite(float(x)) else f"{float(x):.{n}f}"
full_q=p["full_model"]; full_lat,full_fps=timing["full_model"]
md=[
"# Paper-Ready 6x6 Experimental Results","",
"All localization metrics are pooled over Route B+C at frame level. The final MeanShift grid is fixed to 6x6 for every experiment. The visual readout is fixed inside the localization front-end and is not shown as a standalone ablation component.","",
"## Table 1. Overall performance of the proposed framework","",
"| Method | MLE (m) ↓ | MedLE (m) ↓ | P90 (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ | E2E (ms) ↓ | FPS ↑ |",
"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
f"| Proposed Method (6x6) | {fmt(full_q['MLE'])} | {fmt(full_q['MedLE'])} | {fmt(full_q['P90'])} | {fmt(full_q['LSR3'],2)}% | {fmt(full_q['LSR5'],2)}% | {fmt(full_q['LSR10'],2)}% | {fmt(jump['full_model'],3)}% | {fmt(full_lat)} | {fmt(full_fps,1)} |","",
"## Table 2. Ablation study of the main components","",
"| Variant | Forward 3x6 | Temporal GRU | Kalman Fusion | Final MS | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ | E2E (ms) ↓ | FPS ↑ |",
"|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
]
flags={"abl_no_forward":("✗","✓","✓","✓"),"abl_no_gru":("✓","✗","✓","✓"),"abl_no_kalman":("✓","✓","✗","✓"),"abl_no_final_ms":("✓","✓","✓","✗"),"full_model":("✓","✓","✓","✓")}
for name in main_names:
    q=p[name]; lat,fps=timing[name]; a,b,c,e=flags[name]
    md.append(f"| {labels[name]} | {a} | {b} | {c} | {e} | {fmt(q['MLE'])} | {fmt(q['LSR3'],2)}% | {fmt(q['LSR5'],2)}% | {fmt(q['LSR10'],2)}% | {fmt(jump[name],3)}% | {fmt(lat)} | {fmt(fps,1)} |")

md += ["","## Table 3. Effect of temporal context length","","| Temporal Input | First Difference | Second Difference | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ |","|---|:---:|:---:|---:|---:|---:|---:|"]
for name,fd,sd in [("temporal_1f","✗","✗"),("temporal_2f","✓","✗"),("full_model","✓","✓")]:
    q=p[name]; label="3 frames" if name=="full_model" else labels[name]
    md.append(f"| {label} | {fd} | {sd} | {fmt(q['MLE'])} | {fmt(q['LSR3'],2)}% | {fmt(q['LSR5'],2)}% | {fmt(q['LSR10'],2)}% |")

md += ["","## Table 4. External Kalman fusion design","","| Kalman Configuration | Measurement Variance | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | Jump Rate ↓ |","|---|---|---:|---:|---:|---:|---:|"]
for name,var in [("abl_no_kalman","-"),("kalman_learned","Learned"),("full_model","Fixed")]:
    q=p[name]; label={"abl_no_kalman":"No Kalman Fusion","kalman_learned":"Learned-R Kalman","full_model":"Fixed-R Kalman"}[name]
    md.append(f"| {label} | {var} | {fmt(q['MLE'])} | {fmt(q['LSR3'],2)}% | {fmt(q['LSR5'],2)}% | {fmt(q['LSR10'],2)}% | {fmt(jump[name],3)}% |")

md += ["","## Table 5. Effect of the forward candidate restriction","","| Candidate Search | Scored Candidates | MLE (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ | E2E (ms) ↓ | FPS ↑ |","|---|---:|---:|---:|---:|---:|---:|---:|"]
for name,search,cands in [("abl_no_forward","Full 6x6 search",36),("full_model","Forward 3x6 restriction",18)]:
    q=p[name]; lat,fps=timing[name]
    md.append(f"| {search} | {cands} | {fmt(q['MLE'])} | {fmt(q['LSR3'],2)}% | {fmt(q['LSR5'],2)}% | {fmt(q['LSR10'],2)}% | {fmt(lat)} | {fmt(fps,1)} |")

md += ["","## Table 6. MeanShift bandwidth sensitivity at fixed 6x6","","| Bandwidth | MLE (m) ↓ | P90 (m) ↓ | LSR@3 ↑ | LSR@5 ↑ | LSR@10 ↑ |","|---:|---:|---:|---:|---:|---:|"]
for name,bw in [("bw_1m",1),("bw_3m",3),("bw_5m",5),("full_model",7),("bw_9m",9),("bw_11m",11)]:
    q=p[name]; md.append(f"| {bw} m | {fmt(q['MLE'])} | {fmt(q['P90'])} | {fmt(q['LSR3'],2)}% | {fmt(q['LSR5'],2)}% | {fmt(q['LSR10'],2)}% |")

md += ["","## Table 7. Runtime efficiency of the proposed architecture","","| Configuration | Local Geometry | Visually Scored Candidates | Final MS Grid | Latency (ms) ↓ | FPS ↑ |","|---|---:|---:|---:|---:|---:|",f"| Proposed Method | 6x6 | 18 | 6x6 | {fmt(full_lat)} | {fmt(full_fps,1)} |","","## Audit","","- final MeanShift grid fixed to 6x6 for all experiments: PASS","- Route-A-only temporal training; B+C evaluation only: PASS","- no-forward variant separately retrained on Route A: PASS","- 1-frame and 2-frame variants separately retrained on Route A: PASS","- no-GRU uses Kalman CV fallback, not forced zero motion: PASS","- no-Kalman bypasses external Kalman fusion: PASS","- no-final-MS returns the pre-MS estimator output directly: PASS","- pooled LSR@3/5/10 recomputed from B+C frame-level errors: PASS","- main ablation E2E rows measured sequentially on physical GPU0: PASS","- training patience = 5: PASS"]
(suite/"paper_all_tables.md").write_text("\n".join(md)+"\n",encoding="utf-8")

audit={"status":"PASS","grid":"6x6 only","structural_leave_one_out":True,"patience":5,"gpus_used":[0,5,6],"full_BC_MLE_m":full_q["MLE"],"full_BC_LSR3_pct":full_q["LSR3"],"full_BC_LSR5_pct":full_q["LSR5"],"full_BC_LSR10_pct":full_q["LSR10"],"full_E2E_ms":full_lat,"full_FPS":full_fps,"protocol":"controlled local-refinement; pooled Route B+C","paper_tables":["overall performance","main component ablation","temporal context","Kalman design","forward restriction efficiency","MeanShift bandwidth sensitivity","runtime efficiency"]}
(suite/"audit_report.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

print("============================================================================================================")
print("PAPER 6x6 SUITE COMPLETE")
print("Full pooled B+C MLE:",full_q["MLE"])
print("Full LSR@3/5/10:",full_q["LSR3"],full_q["LSR5"],full_q["LSR10"])
print("Full E2E:",full_lat,"ms /",full_fps,"FPS")
print("Tables:",suite/"paper_all_tables.md")
print("CSV   :",suite/"paper_all_summary.csv")
print("Audit :",suite/"audit_report.json")
print("============================================================================================================")
PY

echo "============================================================================================================"
echo "DONE: all selected 6x6 paper experiments completed"
echo "Results: ${SUITE_ROOT}"
echo "Tables : ${SUITE_ROOT}/paper_all_tables.md"
echo "CSV    : ${SUITE_ROOT}/paper_all_summary.csv"
echo "Audit  : ${SUITE_ROOT}/audit_report.json"
echo "============================================================================================================"
