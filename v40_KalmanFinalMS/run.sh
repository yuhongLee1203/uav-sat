#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"

# 最終階段只保留 Kalman 輸出位置作為 MS2 的空間限制。
# 參考點在 MS2 中的權重固定為 0，因此不參與最後 MeanShift 的分數。
# 為了在移除 reference prior 後仍避免農田重複紋理把 MeanShift 拉遠，
# 將原本分散在 Kalman prior + reference prior 的空間約束集中到 Kalman prior。
export MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-4.0}"
export MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-3.5}"
export MS2_REFERENCE_PRIOR_WEIGHT=0.0
export MS2_REFERENCE_SIGMA_M="${MS2_REFERENCE_SIGMA_M:-4.0}"
export MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-5.0}"

rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "============================================================================================================"
echo "v40 Kalman FinalMS"
echo "流程：MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final"
echo "MS2 搜尋中心：Kalman 輸出位置對應的最近 SAT lattice point"
echo "MS2 分數：UAV-SAT 視覺相似度 + Kalman 位置先驗"
echo "MS2 reference prior：停用（weight = 0）"
echo "============================================================================================================"

UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_DEVICE="${DEVICE}" \
JITTER_M="${JITTER_M}" \
bash "${REPO_ROOT}/v39_DirectFinalMS/run.sh"

python3 - "${OUT}/robust_tracker_summary.json" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = "V40_KalmanFinalMS_MobileNetV3_MS1_GRU_Kalman_MS2"
d["final_chain"] = "MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final"
d["second_kalman_update"] = "none"
d["ms2_reference_prior"] = "disabled; reference prior weight = 0"
d["persistent_navigation_state"] = "single Kalman posterior"
d["ms2_search_center"] = "nearest permanent SAT lattice point to the single Kalman posterior"
d["ms2_score"] = "UAV-SAT visual likelihood + Kalman spatial prior only"
d["final_decoder"] = "Kalman-centered full 6x6 Soft MeanShift; MeanShift output is final"
d["ms2_hyperparameters"] = {
    "kalman_sigma_m": float(os.environ.get("MS2_KF_SIGMA_M", "4.0")),
    "kalman_prior_weight": float(os.environ.get("MS2_KF_PRIOR_WEIGHT", "3.5")),
    "reference_prior_weight": 0.0,
    "bandwidth_m": float(os.environ.get("MS2_BANDWIDTH_M", "5.0")),
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] v40 結果：${OUT}/robust_tracker_summary.json"
