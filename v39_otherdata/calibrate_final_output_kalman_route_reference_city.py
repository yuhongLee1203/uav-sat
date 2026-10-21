#!/usr/bin/env python3
"""Global ABCD Route-A calibration under no-GT route-reference inference."""
from __future__ import annotations

# Import first: this patches the shared bearing_iclr_ablation module object.
import bearing_iclr_ablation_route_reference  # noqa: F401
import calibrate_final_output_kalman_global_city as global_cal


if __name__ == "__main__":
    # global_cal patches the same already route-reference-aware ablation module
    # with the pooled ABCD cadence/base, then delegates to the measured
    # train/validation-only calibration routine.
    global_cal.cal.main()
