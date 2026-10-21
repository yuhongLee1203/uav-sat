"""Run a fixed-checkpoint City A validation study of GT-assisted patch search.

Every row evaluates the same frames and weights. GT-assisted rows are oracle
diagnostics: current-frame labels enter inference and are never paper test scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FLAGS = (
    "V40_GT_FRONT18_ORACLE", "V40_GT_FRONT18_PATCH_ORACLE",
    "V40_GT_FINAL_MS_ORACLE_CENTER", "V40_GT_FINAL_MS_PATCH_ORACLE",
)
VARIANTS = {
    "no_gt": (),
    "no_gt_no_gru": (),
    "no_gt_no_kalman": (),
    "no_gt_no_ms": (),
    "gt_reference_front18": (FLAGS[0],),
    "gt_reference_front18_patch": (FLAGS[0], FLAGS[1]),
    "gt_reference_front18_patch_final_center": (FLAGS[0], FLAGS[1], FLAGS[2]),
    "gt_reference_visual_prior_w1": (FLAGS[0], FLAGS[2]),
    "gt_reference_visual_prior_w2": (FLAGS[0], FLAGS[2]),
    "gt_reference_visual_prior_w4": (FLAGS[0], FLAGS[2]),
    "gt_reference_all_patches": FLAGS,
}
VARIANT_ENV = {
    "no_gt_no_gru": {"UAVSAT_EXPERIMENT_DISABLE_GRU": "1"},
    "no_gt_no_kalman": {"UAVSAT_EXPERIMENT_KALMAN": "none"},
    "no_gt_no_ms": {"MS_ENABLED": "0"},
    "gt_reference_visual_prior_w1": {"V40_GT_FINAL_MS_PRIOR_WEIGHT": "1"},
    "gt_reference_visual_prior_w2": {"V40_GT_FINAL_MS_PRIOR_WEIGHT": "2"},
    "gt_reference_visual_prior_w4": {"V40_GT_FINAL_MS_PRIOR_WEIGHT": "4"},
}


def summarize_csv(path: Path) -> dict:
    with path.open(newline="") as handle:
        frames = list(csv.DictReader(handle))
    if not frames:
        raise ValueError(f"No frames in {path}")
    loc = [math.hypot(float(r["final_x"]) - float(r["gt_x"]),
                      float(r["final_y"]) - float(r["gt_y"])) for r in frames]
    heading = [abs((float(r["estimated_heading_deg"]) -
                    float(r["gt_heading_deg"]) + 180) % 360 - 180) for r in frames]
    top1 = [math.hypot(float(r["raw_top1_x"]) - float(r["gt_x"]),
                       float(r["raw_top1_y"]) - float(r["gt_y"])) for r in frames]
    return {
        "frames": len(frames),
        "MLE_m": statistics.fmean(loc),
        "LSR@15_pct": 100 * sum(x <= 15 for x in loc) / len(loc),
        "MHE_deg": statistics.fmean(heading),
        "HSR@15_pct": 100 * sum(x <= 15 for x in heading) / len(heading),
        "local_raw_top1_within_4m_pct": 100 * sum(x <= 4 for x in top1) / len(top1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True,
                        help="Existing fixed-route validation run_settings.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    source = json.loads(args.settings.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in args.variants:
        target = args.output / name
        existing = target / "report.json"
        if not existing.exists():
            env = os.environ.copy()
            env.update(source["environment"])
            env.update({flag: "1" if flag in VARIANTS[name] else "0" for flag in FLAGS})
            env.update(VARIANT_ENV.get(name, {}))
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "2"
            command = [sys.executable, str(HERE / "evaluate_official96_validation.py"),
                       "--checkpoint", source["temporal_checkpoint"],
                       "--city", "citya", "--output", str(target)]
            print(f"[RUN] {name}", flush=True)
            with (target.parent / f"{name}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
        frame_files = list(target.glob("citya/*_frames.csv"))
        if len(frame_files) != 1:
            raise RuntimeError(f"Expected one City A validation frame CSV in {target}")
        report = json.loads(existing.read_text())
        result = summarize_csv(frame_files[0])
        if not math.isclose(result["MLE_m"], report["MLE_m"], abs_tol=1e-5):
            raise RuntimeError(f"CSV/report MLE mismatch in {target}")
        result["current_frame_GT_model_input"] = report["current_frame_GT_model_input"]
        result["checkpoint"] = report["checkpoint"]
        result["visual_checkpoint"] = report["visual_checkpoint"]
        records[name] = result
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[DONE] {name}: MLE={result['MLE_m']:.3f}m", flush=True)
    rows = ["# City A fixed-route GT reference search ablation", "",
            "Same validation frames and frozen model weights. Component rows disable modules",
            "at inference without retraining. GT rows use current-frame GT during",
            "inference; they are oracle diagnostics. Local top1@4m is not official Recall@1.", "",
            "| Variant | GT input | Local top1@4m % | LSR@15 % | HSR@15 % | MLE m | MHE deg | Frames |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in records.items():
        rows.append(f"| {name} | {int(row['current_frame_GT_model_input'])} | "
                    f"{row['local_raw_top1_within_4m_pct']:.2f} | {row['LSR@15_pct']:.2f} | "
                    f"{row['HSR@15_pct']:.2f} | {row['MLE_m']:.3f} | "
                    f"{row['MHE_deg']:.2f} | {row['frames']} |")
    rows += ["", "SR@20, SPL and NE require a navigation rollout; this validation",
             "episode contains localization frames only.", ""]
    (args.output / "summary.md").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
