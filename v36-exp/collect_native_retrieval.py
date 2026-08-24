#!/usr/bin/env python3
"""Convert native retrieval outputs into distance metrics without re-ranking."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


def metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "frames": int(len(errors)),
        "MLE_m": float(errors.mean()),
        "Median_m": float(np.median(errors)),
        "P90_m": float(np.quantile(errors, 0.90)),
        "P95_m": float(np.quantile(errors, 0.95)),
        "LSR@3_pct": float((errors <= 3).mean() * 100),
        "LSR@5_pct": float((errors <= 5).mean() * 100),
        "LSR@10_pct": float((errors <= 10).mean() * 100),
        "LSR@15_pct": float((errors <= 15).mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--query-root", type=Path,
        help="Native U1652 query_drone root; uses the dataset's actual os.walk order and validates every query label.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-output", type=Path,
        help="Optional directory for per-route frame_id/GT/prediction coordinate files.",
    )
    parser.add_argument("--format", choices=("torch", "denseuav-mat"), default="torch")
    args = parser.parse_args()

    if args.format == "torch":
        payload = torch.load(args.features, map_location="cpu")
        query_features = payload["query_features"].float().numpy()
        query_labels = payload["query_labels"].long().numpy()
        gallery_features = payload["gallery_features"].float().numpy()
        gallery_labels = payload["gallery_labels"].long().numpy()
        query_fps = float(payload.get("query_fps", float("nan")))
        query_paths = payload.get("query_paths")
    else:
        payload = loadmat(args.features)
        query_features = np.asarray(payload["query_f"], dtype=np.float32)
        query_labels = np.asarray(payload["query_label"]).reshape(-1).astype(np.int64)
        gallery_features = np.asarray(payload["gallery_f"], dtype=np.float32)
        gallery_labels = np.asarray(payload["gallery_label"]).reshape(-1).astype(np.int64)
        query_fps = float(np.asarray(payload.get("query_fps", [[float("nan")]])).reshape(-1)[0])
        query_paths = np.asarray(payload.get("query_path", [])).reshape(-1)

    # Native University evaluators mark gallery locations that are not positive
    # classes for any test query as -1 (junk) and remove them before ranking.
    # Apply the identical rule before converting Top-1 to metre error.
    valid_gallery = gallery_labels >= 0
    if not np.any(valid_gallery):
        raise RuntimeError("native feature dump contains no non-junk gallery rows")
    gallery_features = gallery_features[valid_gallery]
    gallery_labels = gallery_labels[valid_gallery]
    scores = query_features @ gallery_features.T
    predicted_labels = gallery_labels[scores.argmax(axis=1)]
    manifest = json.loads(args.manifest.read_text())
    xy = {int(key): np.asarray(value["xy"], dtype=np.float64) for key, value in manifest["gallery"].items()}
    errors = [float(np.linalg.norm(xy[int(pred)] - xy[int(gt)])) for pred, gt in zip(predicted_labels, query_labels)]
    if args.trajectory_output is not None:
        if query_paths is not None and len(query_paths) == len(query_labels):
            records = []
            native_labels = []
            for raw_path in query_paths:
                path = Path(str(raw_path))
                stem = path.stem
                if "_f" not in stem:
                    raise RuntimeError(f"cannot parse route/frame from native query path: {path}")
                route, frame_text = stem.rsplit("_f", 1)
                records.append({"route": route, "frame_id": int(frame_text), "file": path.name})
                native_labels.append(int(path.parent.name))
            native_labels = np.asarray(native_labels, dtype=np.int64)
            if not np.array_equal(native_labels, query_labels):
                raise RuntimeError("stored native query paths do not match feature labels")
        elif args.query_root is not None:
            # U1652DatasetEval preserves insertion order from get_data(), whose
            # implementation walks class directories and files without sorting.
            # Reproduce that exact native order instead of assuming manifest order.
            data = {}
            for root, dirs, _ in os.walk(args.query_root, topdown=False):
                for name in dirs:
                    class_path = Path(root) / name
                    class_files = []
                    for _, _, files in os.walk(class_path, topdown=False):
                        class_files = files
                    data[name] = {"path": class_path, "files": class_files}
            records = []
            native_labels = []
            for sample_id, item in data.items():
                for filename in item["files"]:
                    stem = Path(filename).stem
                    if "_f" not in stem:
                        raise RuntimeError(f"cannot parse route/frame from native query name: {filename}")
                    route, frame_text = stem.rsplit("_f", 1)
                    records.append({"route": route, "frame_id": int(frame_text), "file": filename})
                    native_labels.append(int(sample_id))
            native_labels = np.asarray(native_labels, dtype=np.int64)
            if not np.array_equal(native_labels, query_labels):
                mismatch = np.flatnonzero(native_labels != query_labels)
                first = int(mismatch[0]) if len(mismatch) else -1
                raise RuntimeError(
                    f"native query order/feature labels mismatch at row {first}: "
                    f"filesystem={native_labels[first]} feature={query_labels[first]}"
                )
        else:
            records = manifest["queries"]
        if len(records) != len(predicted_labels):
            raise RuntimeError(
                f"manifest has {len(records)} queries but feature dump has {len(predicted_labels)} rows"
            )
        args.trajectory_output.mkdir(parents=True, exist_ok=True)
        predicted_xy = np.asarray([xy[int(label)] for label in predicted_labels], dtype=np.float32)
        gt_xy = np.asarray([xy[int(label)] for label in query_labels], dtype=np.float32)
        errors_np = np.asarray(errors, dtype=np.float32)
        for route in ("route_B", "route_C"):
            indices = np.asarray(
                [index for index, record in enumerate(records) if record["route"] == route],
                dtype=np.int64,
            )
            np.savez_compressed(
                args.trajectory_output / f"{route}_predictions.npz",
                frame_id=np.asarray([records[index]["frame_id"] for index in indices], dtype=np.int32),
                gt_xy=gt_xy[indices], pred_xy=predicted_xy[indices], error_m=errors_np[indices],
            )
    combined = metrics(errors)
    if np.isfinite(query_fps):
        combined["FPS"] = query_fps
    result = {
        "method": args.method,
        "protocol": manifest["protocol"],
        "distance_definition": "Top-1 predicted gallery centre minus positive gallery centre",
        "combined": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
