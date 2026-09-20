#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_formal_v5_smooth_output.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

marker = "FORMAL_V5_BOUNDED_MS_OUTPUT"
if marker in s:
    print("[SMOOTH PATCH] already applied")
    raise SystemExit(0)

old = '''            ms_xy = ms_xy_t[0].detach().cpu().numpy().astype(np.float64)
            ms_support = float(ms_support_t[0].item())
            ms_mode_count = int((ms_mode_weights_t[0] > 0).sum().item())
            ms_s, ms_e, _ = route.project_xy_local(ms_xy, preferred_leg)
            final_se = np.asarray([ms_s, ms_e], dtype=np.float64)
            final_xy = ms_xy.copy()
'''

new = '''            ms_xy = ms_xy_t[0].detach().cpu().numpy().astype(np.float64)
            ms_support = float(ms_support_t[0].item())
            ms_mode_count = int((ms_mode_weights_t[0] > 0).sum().item())

            # FORMAL_V5_BOUNDED_MS_OUTPUT
            # Kalman is the temporally stable estimator.  The final MeanShift is
            # allowed to refine it, but is no longer allowed to replace it with
            # an unrestricted local mode.  This is part of inference itself;
            # the plotting script still draws raw final_x/final_y without any
            # display smoothing.
            ms_raw_residual = ms_xy - kalman_xy
            ms_raw_residual_norm = float(np.linalg.norm(ms_raw_residual))
            ms_output_residual_cap_m = max(
                float(env.get("MS_OUTPUT_RESIDUAL_CAP_M", "3.0")), 0.0
            )
            ms_output_base_blend = float(np.clip(
                float(env.get("MS_OUTPUT_BLEND", "0.35")), 0.0, 1.0
            ))

            if ms_raw_residual_norm > 1e-9 and ms_output_residual_cap_m > 0.0:
                residual_scale = min(
                    1.0, ms_output_residual_cap_m / ms_raw_residual_norm
                )
            elif ms_output_residual_cap_m <= 0.0:
                residual_scale = 0.0
            else:
                residual_scale = 1.0

            # Multiple surviving modes indicate ambiguity.  In that case trust
            # the temporal Kalman state more strongly instead of letting the
            # output bounce among neighboring visual modes from frame to frame.
            mode_damping = 1.0 / math.sqrt(max(ms_mode_count, 1))
            ms_output_effective_blend = float(
                ms_output_base_blend * mode_damping
            )
            bounded_ms_residual = ms_raw_residual * residual_scale
            refined_xy = (
                kalman_xy
                + ms_output_effective_blend * bounded_ms_residual
            )

            ms_s, ms_e, _ = route.project_xy_local(refined_xy, preferred_leg)
            final_se = np.asarray([ms_s, ms_e], dtype=np.float64)
            final_xy = refined_xy.copy()
'''

if s.count(old) != 1:
    raise SystemExit(
        f"smooth output patch failed: final MeanShift output block matches={s.count(old)}"
    )
s = s.replace(old, new, 1)

csv_anchor = '''                "ms_shift_from_kalman_m": float(ms_shift_from_kalman_m),
                "ms_latency_ms": float(ms_latency_ms),
'''
csv_new = '''                "ms_shift_from_kalman_m": float(ms_shift_from_kalman_m),
                "ms_raw_shift_from_kalman_m": float(ms_raw_residual_norm) if ms_enabled else 0.0,
                "ms_output_effective_blend": float(ms_output_effective_blend) if ms_enabled else 0.0,
                "ms_output_residual_cap_m": float(ms_output_residual_cap_m) if ms_enabled else 0.0,
                "ms_latency_ms": float(ms_latency_ms),
'''
if s.count(csv_anchor) != 1:
    raise SystemExit(
        f"smooth output patch failed: CSV anchor matches={s.count(csv_anchor)}"
    )
s = s.replace(csv_anchor, csv_new, 1)

summary_anchor = '''    summary["MS_Definition"] = "exactly one final local Soft MeanShift after the Kalman estimator"
'''
summary_new = '''    summary["MS_Definition"] = "one final local Soft MeanShift whose bounded residual refines the Kalman estimator"
    summary["MS_OutputRefinement"] = "bounded residual to Kalman with ambiguity-aware mode damping"
    summary["MS_OutputBlend"] = float(__import__("os").environ.get("MS_OUTPUT_BLEND", "0.35"))
    summary["MS_OutputResidualCap_m"] = float(__import__("os").environ.get("MS_OUTPUT_RESIDUAL_CAP_M", "3.0"))
'''
if s.count(summary_anchor) != 1:
    raise SystemExit(
        f"smooth output patch failed: summary anchor matches={s.count(summary_anchor)}"
    )
s = s.replace(summary_anchor, summary_new, 1)

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[SMOOTH PATCH] PASS: Kalman + bounded ambiguity-aware MeanShift residual")
