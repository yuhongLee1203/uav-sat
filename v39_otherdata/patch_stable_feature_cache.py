#!/usr/bin/env python3
"""Key UAV feature caches by encoder state, not checkpoint file mtime."""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_stable_feature_cache.py ROBUST_TRACKER")
    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "import csv\nimport json\n",
        "import csv\nimport hashlib\nimport json\n",
        "hashlib import",
    )
    text = replace_once(
        text,
        '''    stat = config.VISUAL_CHECKPOINT.stat()
    signature = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "backbone": str(config.BACKBONE_KEY),
    }
''',
        '''    # Only UAV encoder/head state affects this cache. Satellite gallery,
    # city origin and checkpoint file mtime do not.
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    model_state = checkpoint.get("best_model") or checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise RuntimeError("visual checkpoint has no model state for cache identity")
    digest = hashlib.sha256()
    for key in sorted(model_state):
        value = model_state[key]
        digest.update(str(key).encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    signature = {
        "uav_model_sha256": digest.hexdigest(),
        "backbone": str(config.BACKBONE_KEY),
        "origin_lat": float(checkpoint["origin_lat"]),
        "origin_lon": float(checkpoint["origin_lon"]),
        "route_manifest_sha256": hashlib.sha256(
            (Path(root) / "manifest.csv").read_bytes()
        ).hexdigest(),
    }
''',
        "stable cache signature",
    )
    text = replace_once(
        text,
        '''        # v34-v36 caches predate the shared-cache backbone key. Checkpoint
        # size+mtime are the actual frozen-encoder identity used by those files.
        if (
            cached_signature.get("size") == signature["size"]
            and cached_signature.get("mtime_ns") == signature["mtime_ns"]
            and cached_signature.get("backbone", str(config.BACKBONE_KEY)) == signature["backbone"]
        ):
''',
        '''        if (
            cached_signature.get("uav_model_sha256")
            == signature["uav_model_sha256"]
            and cached_signature.get("backbone") == signature["backbone"]
            and cached_signature.get("origin_lat") == signature["origin_lat"]
            and cached_signature.get("origin_lon") == signature["origin_lon"]
            and cached_signature.get("route_manifest_sha256")
            == signature["route_manifest_sha256"]
        ):
''',
        "stable cache validation",
    )
    path.write_text(text, encoding="utf-8")
    print("[PATCH OK] stable UAV model-state feature cache")


if __name__ == "__main__":
    main()
