#!/usr/bin/env python3
import json
import sys

import config_backbone as experiment_config

# The current repository imports modules using the literal name "config".
# Route those imports to this independent ablation config.
sys.modules["config"] = experiment_config

import visual_model_backbone as experiment_visual_model
sys.modules["visual_model"] = experiment_visual_model

# IMPORTANT:
# We deliberately reuse the repository's existing data.py,
# visual_localizer.py and robust_tracker.py.  Therefore candidate construction,
# Route-A-only training, B/C evaluation, Fixed HardMS and RTL-CRF code remain
# identical.  Only config + frozen image backbone are replaced.
import robust_tracker


def patch_summary_protocol():
    path = experiment_config.OUTPUT_DIR / "robust_tracker_summary.json"
    if not path.exists():
        return

    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["protocol"] = (
        f"Public frozen {experiment_config.BACKBONE_NAME} backbone; "
        "randomly initialized retrieval heads trained/validated only on "
        "Route A; T2-only RTL-CRF trained/validated only on Route A; "
        "Route B/C used only after training for evaluation; "
        "GT+jitter controlled local prior"
    )
    summary["backbone_key"] = experiment_config.BACKBONE_KEY
    summary["temporal_window"] = int(
        experiment_config.TEMPORAL_WINDOW
    )
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(
        "BACKBONE ABLATION | "
        f"backbone={experiment_config.BACKBONE_KEY} | "
        f"feature_dim={experiment_config.CLIP_DIM} | "
        f"window={experiment_config.TEMPORAL_WINDOW}",
        flush=True,
    )
    robust_tracker.main()
    patch_summary_protocol()
