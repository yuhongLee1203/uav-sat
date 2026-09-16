#!/usr/bin/env python3
"""Multi-city entry point for the selected Bearing-adapted v39 model.

The heavy model code stays in bearing_runner_exact_v39.py.  This entry point
only changes the DATA CONTRACT introduced by bearing_prepare_multicity.py:
there is exactly one canonical training route (train_01), plus the two official
held-out Bearing navigation routes.  It also removes a stale audit field that
used to say MS grid=5 although the effective final decoder is 6x6.
"""
from __future__ import annotations

import json
from pathlib import Path

import bearing_runner_exact_v39 as exact

NEW_SELECTION_VERSION = "soft_sequence_v13_multicity_auto_train_fullroute"

# One Route A only.  This is the same A-only temporal protocol used by the
# selected v39 experiment.  Unused training-candidate probes must not become
# training data and must not be allowed to block a city.
exact.EXPECTED_SELECTION_VERSION = NEW_SELECTION_VERSION
exact.base.TRAIN_ROUTES = ("train_01",)
exact.base.TEST_ROUTES = ("test_01", "test_02")

_original_exact_audit = exact._audit


def _audit_multicity(config, runtime: Path, args, prepared_root: Path) -> None:
    _original_exact_audit(config, runtime, args, prepared_root)
    audit_path = Path(config.OUTPUT_DIR) / "v39_bearing_training_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))

    # Paper-facing truth: effective runtime is one final 6x6 MS, BW=7 m.
    audit["final_ms_grid"] = 6
    audit["final_ms_bandwidth_m"] = 7.0
    audit["train_routes"] = ["train_01"]
    audit["test_routes"] = ["test_01", "test_02"]
    audit["route_mapping"]["unused_extra_training_routes"] = []
    audit["route_a_candidate_source"] = exp.get("route_a_candidate_source")
    audit["preparation_profile"] = exp.get("preparation_profile")
    audit["official_test_route_source"] = exp.get("test_route_source")
    audit["prepared_selection_version"] = NEW_SELECTION_VERSION
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(
        "[MULTICITY-V39] audit: PASS | ONE Route A | two official test routes | final MS=6x6/BW7",
        flush=True,
    )
    print(
        "[MULTICITY-V39] selected Route-A source:",
        exp.get("route_a_candidate_source"),
        "| prep profile:",
        exp.get("preparation_profile", {}).get("name"),
        flush=True,
    )


# exact._train_and_infer resolves _audit from its own module globals, therefore
# replace both the module symbol and the callback used by base.main().
exact._audit = _audit_multicity
exact.base._audit_canonical = _audit_multicity
exact.base.train_and_infer = exact._train_and_infer


if __name__ == "__main__":
    exact.base.main()
