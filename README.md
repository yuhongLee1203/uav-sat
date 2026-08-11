# Continuous-Progress Visual RNN v11

This version removes the unstable discrete topology state used by v8-v10.

## Architecture

```text
known W0 + ordered mission waypoints
        |
continuous route progress s
        |
route heading + previous RNN heading residual
        |
existing 6x6 local SAT lattice
        |
keep forward half only = 18 patches
        |
current UAV image <-> current SAT images
        |
learned visual candidate refinement
        |
hard selected image candidate
        |
plain nn.RNNCell
        |
move gate + heading residual + uncertainty + next hidden state
        |
image-supported route-progress update
        |
0 <= delta_s <= 3 m/frame
        |
second-order polynomial inertia CAP
        |
visual progress s
        |
1D progress-only Kalman (no velocity)
        |
route(s) -> final XY
```

## Important properties

- No LSTM.
- No GRU.
- No PREVIOUS/CURRENT/NEXT leg classifier.
- No waypoint transition classifier.
- No per-frame leg argmax.
- No advance/rollback oscillation.
- No fixed nominal speed.
- No velocity state.
- Zero movement is valid.
- Hard maximum movement is 3 m/frame.
- The RNN receives image-derived features only.
- GT/GPS is never a network input or inference search center.
- Teacher forcing is 0.00 from epoch 1.
- B/C waypoint frame_index is not used.
- The active waypoint pair comes only from continuous monotonic route progress.
- The RNN hidden state is passed directly from one frame to the next.
- Closed-loop Route-A validation chooses the best temporal checkpoint and
  supports early stopping.

## Direction

The base search direction is the tangent from the current waypoint toward the
next waypoint. The RNN outputs an image-derived bounded heading residual.
The NEXT frame uses:

`search heading = route heading + previous RNN heading residual`

Therefore the CSV/video exposes:

- `route_heading_deg`
- `heading_residual_deg`
- `estimated_heading_deg`
- `search_heading_deg`

At a 180-degree waypoint turn, continuous progress crosses the waypoint only
once. The route heading then flips to the next waypoint direction; there is no
discrete topology classifier that can oscillate every frame.

## Polynomial inertia

The temporal second-order prediction is:

`delta_poly = 2*delta_(t-1) - delta_(t-2)`

It is used only as a maximum allowed movement cap. It can never push the UAV
forward unless the current UAV/SAT visual candidate also supports forward
progress.

## GT usage

GT exists in `RouteCache` because Route-A needs supervised labels and B/C needs
evaluation metrics. Prediction is completed before the B/C GT row is read.
The temporal checkpoint records:

- `current_gt_as_model_input = False`
- `previous_gt_as_model_input = False`
- `test_gt_as_model_input = False`
- `teacher_forcing_ratio = 0.0`
