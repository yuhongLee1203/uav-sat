# GPS-derived route waypoint manifests

Each `route_*_waypoints.json` gives an ordered start, GPS-derived turn waypoints, and end. Each adjacent pair defines one straight leg for later segment-wise training/inference. The figures are mandatory visual checks before using the manifests.

The source telemetry has no PX4 mission-item list, so these are geometrically derived GPS waypoint estimates rather than confirmed autopilot command waypoints.
