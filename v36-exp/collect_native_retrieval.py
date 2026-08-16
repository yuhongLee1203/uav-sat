#!/usr/bin/env python3
"""Convert native retrieval outputs into distance metrics without re-ranking."""

import argparse
import json
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("torch", "denseuav-mat"), default="torch")
    args = parser.parse_args()

    if args.format == "torch":
        payload = torch.load(args.features, map_location="cpu")
        query_features = payload["query_features"].float().numpy()
        query_labels = payload["query_labels"].long().numpy()
        gallery_features = payload["gallery_features"].float().numpy()
        gallery_labels = payload["gallery_labels"].long().numpy()
        query_fps = float(payload.get("query_fps", float("nan")))
    else:
        payload = loadmat(args.features)
        query_features = np.asarray(payload["query_f"], dtype=np.float32)
        query_labels = np.asarray(payload["query_label"]).reshape(-1).astype(np.int64)
        gallery_features = np.asarray(payload["gallery_f"], dtype=np.float32)
        gallery_labels = np.asarray(payload["gallery_label"]).reshape(-1).astype(np.int64)
        query_fps = float(np.asarray(payload.get("query_fps", [[float("nan")]])).reshape(-1)[0])

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
