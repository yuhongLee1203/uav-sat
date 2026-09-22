#!/usr/bin/env python3
"""Held-out component evaluation with frozen global route-reference profile."""
from __future__ import annotations

# Patch the shared ablation module before importing the existing global-profile
# evaluator. This keeps the same selected parameters but changes the reference
# contract to route_reference for every component row.
import bearing_iclr_ablation_route_reference  # noqa: F401
import eval_with_global_final_output_kalman_profile as global_eval


if __name__ == "__main__":
    global_eval.main()
