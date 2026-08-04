#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
python3 robust_tracker.py "$@"