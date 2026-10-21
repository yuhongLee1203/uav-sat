#!/usr/bin/env python3
"""Patch generated runtime for resumable no-GT route-reference adaptation."""
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
        raise SystemExit("usage: patch_route_reference_runtime.py ROBUST_TRACKER")
    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")

    old_resume = '''        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("best_score", float("inf")))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))
        print("resume temporal training from epoch %d" % start_epoch, flush=True)
'''
    new_resume = '''        model.load_state_dict(payload["model"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        if bool(getattr(config, "ROUTE_REFERENCE_ONLY", False)):
            # This is a new training stage with a different no-GT validation
            # protocol. Carry weights but not the old validation score or
            # exhausted patience counter from another episode.
            # The first episode starts from the older controlled checkpoint.
            # Later episodes carry the same optimizer and model weights, while
            # their local validation score/patience starts fresh.
            if bool(payload.get("episode_optimizer_compatible", False)):
                optimizer.load_state_dict(payload["optimizer"])
            best_score = float("inf")
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            print(
                "resume route-reference adaptation from epoch %d "
                "with episode optimizer/validation reset" % start_epoch,
                flush=True,
            )
        else:
            optimizer.load_state_dict(payload["optimizer"])
            best_score = float(payload.get("best_score", float("inf")))
            best_state = payload.get("best_model")
            patience = int(payload.get("patience", 0))
            print("resume temporal training from epoch %d" % start_epoch, flush=True)
'''
    text = replace_once(text, old_resume, new_resume, "route-reference stage reset")

    old_ratio = '''            acquisition_teacher_ratio = (
                1.0
                + fraction * (float(config.ACQ_TEACHER_FINAL) - 1.0)
            )

        kf = RouteKalman(0.0, 0.0)
'''
    new_ratio = '''            acquisition_teacher_ratio = (
                1.0
                + fraction * (float(config.ACQ_TEACHER_FINAL) - 1.0)
            )
        if bool(getattr(config, "ROUTE_REFERENCE_ONLY", False)):
            episode_ratio = getattr(config, "EPISODE_ACQ_TEACHER_RATIO", None)
            if episode_ratio is not None:
                acquisition_teacher_ratio = float(episode_ratio)
            else:
                adaptation_span = max(int(epochs) - int(start_epoch), 1)
                adaptation_elapsed = max(0, int(epoch) - int(start_epoch))
                acquisition_teacher_ratio = max(
                    0.0, 1.0 - adaptation_elapsed / float(adaptation_span)
                )

        kf = RouteKalman(0.0, 0.0)
'''
    text = replace_once(text, old_ratio, new_ratio, "teacher decay")

    old_teacher = '''            teacher_select = True

            uav_clip = cache.uav_clip[index : index + 1].to(device).float()
'''
    new_teacher = '''            if bool(getattr(config, "ROUTE_REFERENCE_ONLY", False)):
                # Deterministic scheduled sampling: the last adaptation epoch
                # uses exactly the same autonomous selector as validation.
                teacher_bucket = ((index - train_start) * 37 + epoch * 17) % 1000
                teacher_select = teacher_bucket < int(
                    round(1000.0 * acquisition_teacher_ratio)
                )
            else:
                teacher_select = True

            uav_clip = cache.uav_clip[index : index + 1].to(device).float()
'''
    text = replace_once(text, old_teacher, new_teacher, "scheduled acquisition teacher")

    path.write_text(text, encoding="utf-8")
    print("[PATCH OK] route-reference stage reset + scheduled acquisition")


if __name__ == "__main__":
    main()
