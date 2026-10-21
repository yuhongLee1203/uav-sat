"""Validate already-running visual trials after they finish; never use test GT.

This does not start or stop training. Each GPU worker waits for its specified
training process and then evaluates the saved best model on the fixed val route.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="GPU:training_PID:absolute_training_output_dir")
    parser.add_argument("--settings", type=Path, required=True,
                        help="run_settings.json from a completed validation run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new immutable evaluation directory")
    settings = json.loads(args.settings.read_text())
    if settings.get("validation_only") is not True or settings.get("tta_rotations") != [0]:
        parser.error("require validation-only settings without TTA")
    if settings.get("GT_front18_oracle") or settings.get("GT_final_MS_center_oracle"):
        parser.error("automatic model selection forbids current-frame GT oracles")
    runs = []
    for value in args.run:
        gpu, pid, root = value.split(":", 2)
        if gpu not in ("0", "5", "6"):
            parser.error("only authorized GPUs 0,5,6 are allowed")
        runs.append((gpu, int(pid), Path(root).resolve()))
    if len({gpu for gpu, _, _ in runs}) != len(runs):
        parser.error("one worker per GPU")
    args.output.mkdir(parents=True)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    def worker(spec):
        gpu, pid, root = spec
        result = {"trial": str(root), "gpu": gpu}
        report_path = root / "report.json"
        deadline = time.monotonic() + args.timeout_hours * 3600
        try:
            while not report_path.is_file():
                if time.monotonic() > deadline:
                    raise RuntimeError("timed out waiting for training report")
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    if not report_path.is_file():
                        raise RuntimeError("training exited without a final report")
                    break
                time.sleep(5)
            training = json.loads(report_path.read_text())
            checkpoint = Path(training["checkpoint"]).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            output = args.output / root.name
            output.mkdir()
            env = os.environ.copy()
            env.update(settings["environment"])
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "V40_VISUAL_CHECKPOINT_ROOT": str(checkpoint.parent.parent),
                        "V40_FEATURE_CACHE_ROOT": str(root / "validation_cache")})
            command = [sys.executable, "-u", str(Path(__file__).with_name(
                "evaluate_official96_validation.py")), "--checkpoint",
                settings["temporal_checkpoint"], "--output", str(output)]
            print("[VALIDATION START]", root.name, "gpu=" + gpu, flush=True)
            with (output / "run.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            validation = json.loads((output / "report.json").read_text())
            result.update({"metadata_validation_MLE_m": training["best_val_global_top1_MLE_m"],
                           "GT_selected_training_patches": bool(training.get(
                               "best_checkpoint_GT_selected_patches", False)),
                           "route_validation_MLE_m": validation["MLE_m"],
                           "route_validation_LSR15_pct": validation["LSR@15_pct"],
                           "route_validation_4m_gate": validation["MLE_m"] <= 4.0,
                           "report": str(output / "report.json"),
                           "status": "complete"})
        except Exception as error:
            result.update({"status": "failed", "error": str(error)})
        print("[VALIDATION RESULT]", json.dumps(result), flush=True)
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runs)) as pool:
        futures = [pool.submit(worker, run) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            summary = {"validation_only": True, "test_evaluated": False,
                       "official_table6_reproduction": False, "results": results}
            temporary = args.output / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
