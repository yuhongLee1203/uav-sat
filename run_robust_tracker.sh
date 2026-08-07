#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# This no-position-scale experiment intentionally reuses the exact same
# Route-A-only visual retrieval checkpoint as the original /10 experiment.
# Only RTL-CRF is retrained, so the ablation isolates POSITION_SCALE_M.
BASELINE_VISUAL="outputs/strict_train_A_test_BC/checkpoints/visual_retrieval_A_only.pt"
TARGET_VISUAL="outputs/strict_train_A_test_BC_no_position_scale/checkpoints/visual_retrieval_A_only.pt"

mkdir -p "$(dirname "$TARGET_VISUAL")"

if [[ ! -f "$TARGET_VISUAL" ]]; then
    if [[ ! -f "$BASELINE_VISUAL" ]]; then
        echo "ERROR: missing baseline Route-A-only visual checkpoint:" >&2
        echo "  $BASELINE_VISUAL" >&2
        echo "Train the original strict A->B/C visual model first, or copy that checkpoint here." >&2
        exit 2
    fi
    cp -p "$BASELINE_VISUAL" "$TARGET_VISUAL"
    echo "Copied fixed Route-A-only visual checkpoint for fair no-scale ablation:"
    echo "  $BASELINE_VISUAL"
    echo "  -> $TARGET_VISUAL"
fi

# robust_tracker.py currently always calls train_visual_retrieval_a_only() in
# train/train_eval mode. With --visual-epochs 0, it must enter resume mode so
# train_visual_retrieval_a_only() loads the existing best visual checkpoint
# instead of trying to run a zero-epoch training loop with best_state=None.
HAS_RESUME=0
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        HAS_RESUME=1
        break
    fi
done

if [[ $HAS_RESUME -eq 1 ]]; then
    exec python3 robust_tracker.py "$@"
else
    exec python3 robust_tracker.py --resume "$@"
fi
