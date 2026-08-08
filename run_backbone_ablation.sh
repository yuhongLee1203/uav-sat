#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WINDOW="${RTL_TEMPORAL_WINDOW:-5}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-40}"
JITTER_M="${JITTER_M:-12}"

declare -A GPU
GPU[vgg16]="${GPU_VGG16:-0}"
GPU[resnet18]="${GPU_RESNET18:-5}"
GPU[mobilenet_v3_small]="${GPU_MOBILENET:-6}"

BACKBONES=(vgg16 resnet18 mobilenet_v3_small)

echo "=============================================================="
echo "BACKBONE ABLATION"
echo "T2-only RTL-CRF | temporal window = ${WINDOW}"
echo "VGG16             -> physical GPU ${GPU[vgg16]}"
echo "ResNet18           -> physical GPU ${GPU[resnet18]}"
echo "MobileNetV3-Small  -> physical GPU ${GPU[mobilenet_v3_small]}"
echo "=============================================================="
echo
echo "Each backbone trains a FRESH Route-A-only visual head."
echo "No MobileCLIP task checkpoint is reused."
echo

pids=()
labels=()

for backbone in "${BACKBONES[@]}"; do
    output="outputs/backbone_ablation_${backbone}_t2only_w${WINDOW}"
    mkdir -p "${output}"

    echo "Launching ${backbone} on GPU ${GPU[$backbone]} ..."

    CUDA_VISIBLE_DEVICES="${GPU[$backbone]}" \
    RTL_BACKBONE="${backbone}" \
    RTL_TEMPORAL_WINDOW="${WINDOW}" \
    nohup python3 run_backbone_ablation.py \
        --mode train_eval \
        --visual-epochs "${VISUAL_EPOCHS}" \
        --epochs "${TEMPORAL_EPOCHS}" \
        --jitter-m "${JITTER_M}" \
        --eval-split all \
        > "${output}/train_eval.log" 2>&1 &

    pids+=("$!")
    labels+=("${backbone}")
done

echo
echo "Started:"
for i in "${!pids[@]}"; do
    echo "  ${labels[$i]} PID=${pids[$i]}"
done
echo

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[OK] ${labels[$i]} finished"
    else
        echo "[FAIL] ${labels[$i]} failed; check its train_eval.log" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one training job failed. Comparison was not run." >&2
    exit 1
fi

echo
echo "=============================================================="
echo "All accuracy experiments finished. Running speed benchmark..."
echo "=============================================================="

mkdir -p outputs/backbone_speed

# Run speed tests sequentially so they do not interfere with one another.
# Use one physical GPU after the training jobs have completed.
SPEED_GPU="${GPU_SPEED:-0}"

for backbone in mobileclip2_s2 vgg16 resnet18 mobilenet_v3_small; do
    echo
    echo "Speed: ${backbone}"
    CUDA_VISIBLE_DEVICES="${SPEED_GPU}" \
    python3 benchmark_backbone_speed.py \
        --backbone "${backbone}" \
        --warmup 30 \
        --iters 200 \
        --batch-size 64
done

echo
echo "=============================================================="
echo "ACCURACY + SPEED SUMMARY"
echo "=============================================================="

python3 - <<'PY'
from pathlib import Path
import json

window = int(__import__("os").environ.get("RTL_TEMPORAL_WINDOW", "5"))
backbones = ["vgg16", "resnet18", "mobilenet_v3_small"]

rows = []
for b in backbones:
    summary_path = Path(
        f"outputs/backbone_ablation_{b}_t2only_w{window}/"
        "robust_tracker_summary.json"
    )
    speed_path = Path(f"outputs/backbone_speed/{b}.json")

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not speed_path.exists():
        raise FileNotFoundError(speed_path)

    s = json.loads(summary_path.read_text(encoding="utf-8"))
    speed = json.loads(speed_path.read_text(encoding="utf-8"))

    route_b = s["routes"]["route_B"]["RTL_CRF"]
    route_c = s["routes"]["route_C"]["RTL_CRF"]

    rows.append({
        "backbone": b,
        "B_MLE_m": route_b["MLE_m"],
        "B_P90_m": route_b["P90_m"],
        "B_RPE_m": route_b["RPE_m"],
        "B_Jump_pct": route_b["JumpRate_pct"],
        "C_MLE_m": route_c["MLE_m"],
        "C_P90_m": route_c["P90_m"],
        "C_RPE_m": route_c["RPE_m"],
        "C_Jump_pct": route_c["JumpRate_pct"],
        "UAV_ms_per_frame": speed["uav_online_batch1"]["ms_per_image"],
        "UAV_backbone_FPS": speed["uav_online_batch1"]["images_per_second"],
        "backbone_params": speed["backbone_parameters"],
    })

mobileclip_speed_path = Path("outputs/backbone_speed/mobileclip2_s2.json")
mobileclip_speed = None
if mobileclip_speed_path.exists():
    ms = json.loads(mobileclip_speed_path.read_text(encoding="utf-8"))
    mobileclip_speed = {
        "UAV_ms_per_frame": ms["uav_online_batch1"]["ms_per_image"],
        "UAV_backbone_FPS": ms["uav_online_batch1"]["images_per_second"],
        "backbone_params": ms["backbone_parameters"],
    }

# Reuse the already-completed current MobileCLIP2-S2 T2-only 5-frame result
# as the accuracy baseline when it exists. No retraining is needed.
mobileclip_summary_path = Path(
    f"outputs/strict_train_A_test_BC_t2only_w{window}/robust_tracker_summary.json"
)
if mobileclip_summary_path.exists() and mobileclip_speed is not None:
    s = json.loads(mobileclip_summary_path.read_text(encoding="utf-8"))
    rb = s["routes"]["route_B"]["RTL_CRF"]
    rc = s["routes"]["route_C"]["RTL_CRF"]
    rows.insert(0, {
        "backbone": "mobileclip2_s2",
        "B_MLE_m": rb["MLE_m"],
        "B_P90_m": rb["P90_m"],
        "B_RPE_m": rb["RPE_m"],
        "B_Jump_pct": rb["JumpRate_pct"],
        "C_MLE_m": rc["MLE_m"],
        "C_P90_m": rc["P90_m"],
        "C_RPE_m": rc["RPE_m"],
        "C_Jump_pct": rc["JumpRate_pct"],
        "UAV_ms_per_frame": mobileclip_speed["UAV_ms_per_frame"],
        "UAV_backbone_FPS": mobileclip_speed["UAV_backbone_FPS"],
        "backbone_params": mobileclip_speed["backbone_params"],
    })

print()
print(
    f"{'backbone':<22}"
    f"{'B MLE':>10}"
    f"{'B RPE':>10}"
    f"{'B Jump%':>11}"
    f"{'C MLE':>10}"
    f"{'C RPE':>10}"
    f"{'C Jump%':>11}"
    f"{'UAV ms':>10}"
    f"{'FPS':>10}"
)
print("-" * 104)

for r in rows:
    print(
        f"{r['backbone']:<22}"
        f"{r['B_MLE_m']:>10.3f}"
        f"{r['B_RPE_m']:>10.3f}"
        f"{r['B_Jump_pct']:>11.3f}"
        f"{r['C_MLE_m']:>10.3f}"
        f"{r['C_RPE_m']:>10.3f}"
        f"{r['C_Jump_pct']:>11.3f}"
        f"{r['UAV_ms_per_frame']:>10.3f}"
        f"{r['UAV_backbone_FPS']:>10.1f}"
    )

if not mobileclip_summary_path.exists():
    print()
    print(
        "NOTE: existing MobileCLIP accuracy summary was not found at "
        f"{mobileclip_summary_path}; MobileCLIP speed was still measured."
    )

out = {
    "temporal_window": window,
    "accuracy_and_speed": rows,
    "mobileclip2_s2_speed": mobileclip_speed,
    "mobileclip_existing_summary": (
        str(mobileclip_summary_path)
        if mobileclip_summary_path.exists()
        else None
    ),
}
out_path = Path("outputs/backbone_ablation_comparison.json")
out_path.write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print()
print(f"saved: {out_path}")
PY
