#!/usr/bin/env python3
"""Compatibility entrypoint for Core V5-Restore.

The completed local Formal-V5 runner has a generated ``train_frames`` argument
that is not present in the GitHub template parser.  Core V5-Restore is always a
3-frame Full model, so inject train_frames=3 before delegating to the real local
runner.  This file does not modify the Formal-V5 runner itself.
"""
from __future__ import annotations

import inspect
import re

import bearing_core_v5_restore as core


def _ensure_compat(args):
    # Formal-V5 generated train_full() requires this field.  Core V5-Restore is
    # intentionally fixed to the trained 3-frame Full architecture.
    if not hasattr(args, "train_frames"):
        args.train_frames = 3
    else:
        args.train_frames = 3

    # Keep epochs_per_route aligned with the temporal training budget if a
    # generated runner expects it.
    if not hasattr(args, "epochs_per_route"):
        args.epochs_per_route = int(getattr(args, "temporal_epochs", 100))

    return args


def _audit_callable(fn, args, label):
    """Report args.* fields referenced directly by the generated entrypoint.

    This is a compatibility audit, not a brittle hard failure for fields that
    may be assigned internally.  The known generated-only field train_frames is
    injected above before execution.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        print(f"[CORE-V5 COMPAT] {label}: source inspection unavailable; continuing")
        return
    refs = sorted(set(re.findall(r"args\.([A-Za-z_][A-Za-z0-9_]*)", src)))
    missing = [name for name in refs if not hasattr(args, name)]
    print(f"[CORE-V5 COMPAT] {label} args refs={refs}")
    print(f"[CORE-V5 COMPAT] train_frames={getattr(args, 'train_frames', None)}")
    if missing:
        print(f"[CORE-V5 COMPAT] {label} unresolved direct args refs before call: {missing}")


_ORIG_TRAIN_FULL = core.ab.train_full
_ORIG_EVALUATE = core.ab.evaluate


def _train_full(args):
    _ensure_compat(args)
    _audit_callable(_ORIG_TRAIN_FULL, args, "train_full")
    return _ORIG_TRAIN_FULL(args)


def _evaluate(args):
    _ensure_compat(args)
    _audit_callable(_ORIG_EVALUATE, args, "evaluate")
    return _ORIG_EVALUATE(args)


core.ab.train_full = _train_full
core.ab.evaluate = _evaluate


if __name__ == "__main__":
    core.main()
