# Root v33 backbone experiments

This directory benchmarks only the architecture implemented by the files in
the `/yh/study/uav-sat` root. It does not import or execute the old `CRF/`
experiment.

Pipeline measured:

`prepared UAV image -> visual backbone -> local retrieval -> ThreeFrameRouteStateGRU -> second-order polynomial -> forward 3x6 visual measurement -> external RouteKalman -> final XY`

Run:

```bash
cd /yh/study/uav-sat
bash backbone-exp/run_v33_backbone_experiments.sh
```

Results are written to `backbone-exp/outputs/`.
