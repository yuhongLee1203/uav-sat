# GPS-derived route waypoint manifests

Each `route_*_waypoints.json` gives ordered start, GPS-derived turn waypoints, and end. Each adjacent pair defines one straight leg. The figures are visual checks before using the manifests.

The source telemetry contains sampled GPS but no PX4 mission-item list; these are geometrically derived GPS waypoint estimates, not asserted flight-controller command coordinates.
