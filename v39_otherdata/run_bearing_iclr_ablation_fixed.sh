#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
python3 v39_otherdata/patch_bearing_iclr_main_alignment.py v39_otherdata/bearing_iclr_ablation.py
python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py v39_otherdata/patch_bearing_iclr_main_alignment.py
bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
