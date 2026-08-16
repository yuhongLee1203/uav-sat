# Dense route-reference experiment (GPU 0)

This launcher restores the previously trainable frame-reference architecture:
one causal-heading forward 3x6 SAT window, SoftMS, 3-frame GRU, quadratic
polynomial motion, four prediction heads, and the external RouteKalman.

The failed multi-window route bank is disabled. The only protocol change is the
source of the frame reference:

1. `build_dense_route_references.py` freezes Route A/B/C into a frame-indexed
   reference manifest under `references/`.
2. At frame `t`, the tracker reads reference `R_t` from that fixed manifest and
   centres the single 3x6 window there.
3. Current-frame GT is not read by inference to select the window, cap progress,
   cap step size, or teacher-select a visual result.
4. During Route-A training, GT remains the Smooth-L1 supervision label for the
   current visual measurement and the `t -> t+1` polynomial landing position.
5. During Route-B/C evaluation, GT is used only to calculate metrics.

Important research limitation: the repository contains sparse GPS-derived turn
waypoints, not an independently recorded dense autopilot mission plan. The
dense frame manifests are therefore frozen offline from the dataset trajectory.
They test a known time-indexed route-reference setting; they must not be
described as references obtained from an independent flight plan.

Run:

```bash
bash frame-reference-exp/run_frame_reference_gpu3.sh
```

Despite its historical filename, the launcher uses GPU 0. Results are written
to `frame-reference-exp/outputs/scheduled_route_single3x6_gpu0/` and the log to
`frame-reference-exp/logs/scheduled_route_single3x6_gpu0.log`.
