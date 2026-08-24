#!/usr/bin/env bash
set -euo pipefail

EXP="/yh/study/uav-sat/v36-exp"
cd "$EXP"

# Builds/reuses only the shared Route-A/fixed-gallery image layout.  It does
# not invoke DenseUAV, Sample4Geo, or Game4Loc and does not modify their runs.
python3 prepare_native_paper_dataset.py

log_dir="$EXP/logs/native-unreported"
mkdir -p "$log_dir"
tag="$(date +%Y%m%d_%H%M%S)"

if [[ -e "$EXP/outputs/native-unreported/InfoGeo/summary.json" || -e "$EXP/outputs/native-unreported/Bearing-UAV/summary.json" ]]; then
    echo "Refusing to overwrite a completed unreported-paper result. Move the completed output directory only if you deliberately want a new run." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=5 nohup python3 -u repro_papers/run_infogeo_routea.py \
    --epochs 12 --batch-size 16 --eval-batch-size 32 \
    > "$log_dir/infogeo_${tag}.log" 2>&1 &
infogeo_pid=$!

CUDA_VISIBLE_DEVICES=6 nohup python3 -u repro_papers/run_bearinguav_routea.py \
    --epochs 60 --batch-size 16 \
    > "$log_dir/bearinguav_${tag}.log" 2>&1 &
bearing_pid=$!

echo "InfoGeo GPU 5 PID: $infogeo_pid"
echo "Bearing-UAV GPU 6 PID: $bearing_pid"
echo "Logs: $log_dir/infogeo_${tag}.log"
echo "Logs: $log_dir/bearinguav_${tag}.log"
