"""Controlled evaluation for the one-frame delayed KG/GK experiment.

This file does NOT retrain the delayed pair model. It loads the existing
`delayed_pair_ms_kg_gk_experiment.py` checkpoints and evaluates the same
one-frame delayed formulation with a frame-aligned reference-centered local
visual search on Routes B/C.

For pair [I(t), I(t+1)]:
  reference(t) -> centered 6x6 MS on I(t) -> FINAL x_t
  FINAL x_t -> KG/GK with [I(t), I(t+1)] -> provisional x'_(t+1)

Thus the reported FINAL position is still the delayed-MS output, while the
provisional next-frame prediction is measured separately. No checkpoint or
output from the autonomous delayed, segmented, curriculum, or route-tube
experiments is overwritten.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker as rt
import delayed_pair_ms_kg_gk_experiment as delayed

ARCH_CHOICES = ("KG", "GK")


def _output_dir(arch):
    return (
        Path(config.BACKBONE_OUTPUT_DIR)
        / "delayed_pair_gtcenter_controlled_center6x6"
        / arch.lower()
    )


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rt.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


@torch.no_grad()
def evaluate(arch, visual, model, route_name, cache, device):
    model.eval()
    state = delayed._make_state(route_name, visual)

    rows = []
    final_errors = []
    provisional_errors = []

    for index in range(len(cache) - 1):
        uav_current = cache.uav_clip[index:index + 1].to(device).float()
        uav_next = cache.uav_clip[index + 1:index + 2].to(device).float()

        reference_current = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        reference_next = cache.gt_xy[index + 1].cpu().numpy().astype(np.float64)

        final_current_t, provisional_next_t, _, trace = delayed.pair_step(
            arch,
            model,
            visual,
            uav_current,
            uav_next,
            reference_current,
            state,
            device,
        )

        final_current = final_current_t[0].detach().cpu().numpy().astype(np.float64)
        provisional_next = provisional_next_t[0].detach().cpu().numpy().astype(np.float64)

        final_error = float(np.linalg.norm(final_current - reference_current))
        provisional_error = float(np.linalg.norm(provisional_next - reference_next))
        final_errors.append(final_error)
        provisional_errors.append(provisional_error)

        row = {
            "pair_current_index": int(index),
            "pair_next_index": int(index + 1),
            "current_frame_id": int(cache.frame_ids[index]),
            "next_frame_id": int(cache.frame_ids[index + 1]),
            "reference_current_x": float(reference_current[0]),
            "reference_current_y": float(reference_current[1]),
            "search_center_x": float(reference_current[0]),
            "search_center_y": float(reference_current[1]),
            "search_center_error_m": 0.0,
            "final_ms_x": float(final_current[0]),
            "final_ms_y": float(final_current[1]),
            "final_error_m": final_error,
            "reference_next_x": float(reference_next[0]),
            "reference_next_y": float(reference_next[1]),
            "provisional_next_x": float(provisional_next[0]),
            "provisional_next_y": float(provisional_next[1]),
            "provisional_next_error_m": provisional_error,
            "ms_candidate_count": int(trace["candidate_count"]),
        }
        for name in ("kalman_current_xy", "gru_next_xy", "kalman_next_xy"):
            if name in trace:
                v = trace[name][0].detach().cpu().numpy().astype(np.float64)
                row[name + "_x"] = float(v[0])
                row[name + "_y"] = float(v[1])
        rows.append(row)

    outdir = _output_dir(arch)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / (
        "%s_%s_delayed_gtcenter_controlled_frames.csv" % (route_name, arch.lower())
    )
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = rt.metric_summary(final_errors)
    summary.update({
        "Architecture": arch,
        "Route": route_name,
        "Protocol": "one-frame delayed controlled reference-centered evaluation",
        "LatencyFrames": 1,
        "SearchCenterMLE_m": 0.0,
        "FinalMS_MLE_m": float(np.mean(final_errors)),
        "ProvisionalNextMLE_m": float(np.mean(provisional_errors)),
        "FrameAlignedReferencePriorEval": True,
        "FinalOutput": "MeanShift on I(t), finalized when pair [I(t), I(t+1)] is processed",
        "ProvisionalOutput": "KG/GK one-step x'_(t+1)",
        "MeanShiftCandidateCount": 36,
        "ForwardSearch": False,
        "CSV": str(csv_path),
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCH_CHOICES, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = rt.resolve_device(args.device)
    rt.set_seed(int(config.SEED))
    if not config.VISUAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.VISUAL_CHECKPOINT)

    visual = delayed.FrozenVisualLocalizer(device)
    model = delayed.load_architecture(args.arch, device)

    result = {
        "architecture": args.arch,
        "train_route": "route_A",
        "test_routes": ["route_B", "route_C"],
        "protocol": "one-frame delayed controlled reference-centered evaluation",
        "reused_checkpoint": str(delayed._checkpoint_path(args.arch)),
        "frame_aligned_reference_prior_eval": True,
        "meanshift_candidate_count": 36,
        "forward_search": False,
        "results": {},
    }

    for route_name in ("route_B", "route_C"):
        cache = build_cache(route_name, visual, device)
        summary = evaluate(args.arch, visual, model, route_name, cache, device)
        result["results"][route_name] = summary
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    outdir = _output_dir(args.arch)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
