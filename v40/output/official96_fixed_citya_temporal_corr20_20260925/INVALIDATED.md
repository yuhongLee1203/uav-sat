Do not select this run for the final model.

The old monotonic route-coordinate supervision mapped two shifted training
episodes to incorrect position targets (XY reconstruction MLE 145.894 m and
270.332 m). The correction-bound trials were interrupted after discovering
this issue. Existing logs are retained as experimental history.

Replacement trials use raw Cartesian GT only inside the loss, with the
unchanged model output converted through the runtime waypoint geometry.
Replacement output names contain `temporal_xy_corr`.
