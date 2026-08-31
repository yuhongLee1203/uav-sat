"""Corrected route-tube recovery wrapper.

This v2 keeps the training/evaluation implementation from
`delayed_pair_route_tube_recovery_experiment.py` but replaces its route guard
with a stricter geometry rule:

* hard corridor is LATERAL distance to the ordered planned route;
* progress is separately bounded by the causal forward window;
* a proposal that lies past the forward-window endpoint is recovered even if it
  is geometrically close to the same straight route;
* backward motion past the current progress anchor is also recovered.

No frame-aligned B/C reference is used by this guard.
"""

import numpy as np

import delayed_pair_route_tube_recovery_experiment as v1


class CausalRouteTube(v1.CausalRouteTube):
    """Planned-route tube with separate lateral and longitudinal constraints."""

    def constrain(self, raw_xy):
        p = np.asarray(raw_xy, dtype=np.float64).reshape(2)
        previous_index = int(self.last_index)
        start = previous_index
        end = min(len(self.xy), start + self.max_forward_steps + 1)
        local = self.xy[start:end]
        if len(local) == 0:
            local = self.xy[-1:]
            start = len(self.xy) - 1
            end = len(self.xy)

        local_i = int(np.linalg.norm(local - p[None, :], axis=1).argmin())
        index = start + local_i
        route_xy = self.xy[index].copy()
        tangent = self.tangent(index)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        delta = p - route_xy

        lateral_distance = float(abs(np.dot(delta, normal)))
        along_residual = float(np.dot(delta, tangent))

        # If the nearest allowed route point is the forward boundary and the raw
        # prediction still lies beyond that boundary, the actual prediction has
        # exceeded the per-frame progress cap even when lateral distance is tiny.
        at_forward_boundary = bool(index == end - 1 and end < len(self.xy))
        forward_limited = bool(
            at_forward_boundary and along_residual > 0.5 * self.spacing_m
        )

        # The route tracker is causal/non-backtracking. A proposal clearly behind
        # the current anchor is also recovered instead of letting the search walk
        # backward and destabilize progress.
        at_backward_boundary = bool(index == start and previous_index > 0)
        backward_limited = bool(
            at_backward_boundary and along_residual < -0.5 * self.spacing_m
        )

        lateral_limited = bool(lateral_distance > self.hard_width_m)
        triggered = bool(lateral_limited or forward_limited or backward_limited)

        protected = route_xy.copy() if triggered else p.copy()
        progress_delta_m = float((index - previous_index) * self.spacing_m)
        self.last_index = max(previous_index, index)

        return {
            "raw_xy": p,
            "protected_xy": protected,
            "route_xy": route_xy,
            "route_distance_m": lateral_distance,
            "lateral_distance_m": lateral_distance,
            "along_residual_m": along_residual,
            "lateral_limited": lateral_limited,
            "forward_limited": forward_limited,
            "backward_limited": backward_limited,
            "triggered": triggered,
            "route_index": int(self.last_index),
            "progress_delta_m": progress_delta_m,
            "tangent": self.tangent(self.last_index),
        }


# The imported train/eval functions resolve CausalRouteTube from the v1 module's
# globals at runtime, so replacing it here upgrades both training and evaluation
# without duplicating the full experiment implementation.
v1.CausalRouteTube = CausalRouteTube


if __name__ == "__main__":
    v1.main()
