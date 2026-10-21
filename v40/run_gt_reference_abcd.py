"""Exploratory validation on the previously assembled four-city c4m episodes.

The assembled episodes are synthetic and some historical weights used GT priors
during training. GT-assisted evaluations use current-frame labels at inference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_gt_reference_ablation import summarize_csv


HERE = Path(__file__).resolve().parent
CITIES = ("citya", "cityb", "cityc", "cityd")
VARIANTS = {
    "no_gt": {},
    "gt_reference_prior_w4": {
        "V40_GT_FRONT18_ORACLE": "1", "V40_GT_FINAL_MS_ORACLE_CENTER": "1",
        "V40_GT_FINAL_MS_PRIOR_WEIGHT": "4"},
    "gt_patch_oracle": {
        "V40_GT_FRONT18_ORACLE": "1", "V40_GT_FRONT18_PATCH_ORACLE": "1",
        "V40_GT_FINAL_MS_ORACLE_CENTER": "1", "V40_GT_FINAL_MS_PATCH_ORACLE": "1"},
}
GT_FLAGS = ("V40_GT_FRONT18_ORACLE", "V40_GT_FRONT18_PATCH_ORACLE",
            "V40_GT_FINAL_MS_ORACLE_CENTER", "V40_GT_FINAL_MS_PATCH_ORACLE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    source = json.loads(args.settings.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    shared = args.suite / "shared" / "frames3"
    visual = (shared / "visual_retrieval_A_only.pt").resolve()
    temporal = (shared / "controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt").resolve()
    if not visual.is_file() or not temporal.is_file():
        raise FileNotFoundError("Missing shared four-city checkpoints")
    summary = {}
    for city in CITIES:
        summary[city] = {}
        for variant in args.variants:
            target = args.output / city / variant
            report_file = target / "report.json"
            if not report_file.exists():
                env = os.environ.copy()
                env.update(source["environment"])
                env.update({flag: "0" for flag in GT_FLAGS})
                env.pop("V40_GT_FINAL_MS_PRIOR_WEIGHT", None)
                env.update(VARIANTS[variant])
                env["V40_BASE_CITYA"] = str((args.suite / "citya").resolve())
                env["V40_PREPARED_ROOT"] = str(args.suite.resolve())
                env["V40_VISUAL_CHECKPOINT_FILE"] = str(visual)
                env["V40_FEATURE_CACHE_ROOT"] = str((args.output / "cache").resolve())
                env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "2"
                command = [sys.executable, str(HERE / "evaluate_official96_validation.py"),
                           "--checkpoint", str(temporal), "--city", city,
                           "--output", str(target)]
                print("[RUN]", city, variant, flush=True)
                with (args.output / f"{city}_{variant}.log").open("w") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True)
            report = json.loads(report_file.read_text())
            frames = list((target / city).glob("*_frames.csv"))
            if len(frames) != 1:
                raise RuntimeError(f"Missing validation frame CSV in {target}")
            row = summarize_csv(frames[0])
            row.update({"city": city, "variant": variant,
                        "current_frame_GT_model_input": report["current_frame_GT_model_input"]})
            summary[city][variant] = row
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("[DONE]", city, variant, f"MLE={row['MLE_m']:.3f}m", flush=True)
    lines = ["# Historical shared ABCD checkpoint on synthetic c4m validation episodes", "",
             "These are GT-selected synthetic episodes, not recorded flights or official Table 6.",
             "GT variants consume each validation frame's label during inference.",
             "The historical shared checkpoint was trained under a different protocol than",
             "the fixed-route City A pilot. Its results must not be pooled with that pilot.", "",
             "| City | Variant | Frames | Local top1@4m % | LSR@15 % | HSR@15 % | MLE m | MHE deg |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for city, variants in summary.items():
        for name, row in variants.items():
            lines.append(f"| {city} | {name} | {row['frames']} | "
                         f"{row['local_raw_top1_within_4m_pct']:.2f} | "
                         f"{row['LSR@15_pct']:.2f} | {row['HSR@15_pct']:.2f} | "
                         f"{row['MLE_m']:.3f} | {row['MHE_deg']:.2f} |")
    lines += ["", "Official R@1 requires the four reference tiles per independent query.",
              "SR@20, SPL and NE require navigation rollouts and cannot be inferred",
              "from these localization frames.", ""]
    (args.output / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
