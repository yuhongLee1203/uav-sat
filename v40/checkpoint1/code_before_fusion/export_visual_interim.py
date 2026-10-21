"""Snapshot a running backbone experiment for validation without stopping it.

Use only with the exact source checkpoint passed to that training process.
The exported weights are immutable, so later training cannot change an eval.
"""
import argparse
import json
from pathlib import Path

import torch

from train_visual_backbone_gtloss import state_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        parser.error("refusing to overwrite an existing snapshot")
    source = torch.load(args.source_checkpoint, map_location="cpu")
    if (source.get("training_protocol") not in (
            "global_gallery_gt_loss_only", "global_gallery_gt_selected_train_patches")
            or source.get("official96_train_only") is not True
            or source.get("previous_task_checkpoint_loaded") is not False):
        parser.error("source must be a strict GT-loss-only official96 checkpoint")
    before = args.interim.stat()
    interim = torch.load(args.interim, map_location="cpu")
    after = args.interim.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError("interim changed while reading; retry after epoch save")
    source.update({
        "model": interim["model"], "best_model": interim["model"],
        "backbone_state": interim["backbone_state"],
        "backbone_state_sha256": state_sha256(interim["backbone_state"]),
        "backbone_fine_tuned": True, "gallery": interim["gallery"],
        "best_score": float(interim["validation_mle_m"]),
        "interim_epoch": int(interim["epoch"]),
        "interim_source": str(args.interim.resolve()),
        "training_source_checkpoint": str(args.source_checkpoint.resolve()),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, args.output)
    print(json.dumps({"checkpoint": str(args.output),
                      "epoch": source["interim_epoch"],
                      "metadata_validation_MLE_m": source["best_score"]}))


if __name__ == "__main__":
    main()
