# FieldAnchor v37 4x6: B+C training, A validation

## Protocol

- Visual training mixes B+C; temporal training alternates B and C every epoch.
- Backbone: `MobileNetV3-Small`.
- Model selection, early-stopping score, and final evaluation: `route_A`.
- Local-search prior: an offline reference route sampled at approximately one
  SAT stride (`4.5 m`) and interpolated for each frame.
- Current-frame GT is **not** used to centre the search window and no jitter is
  added. GT coordinates remain only as supervised training/evaluation labels;
  removing those labels would make supervised localization training undefined.
- All checkpoints, caches, logs, summaries, images, and videos are written below
  `v37/outputs/bc_train_a_validation_4x6`.

The forward 4x6 geometry gives 100% candidate capture on A/B/C. Compared with
3x6, Route A capture rises from 98.61% to 100% and its mean nearest-candidate
distance falls from 3.279 m to 1.852 m, at 33% more visual candidates.

## Why the old non-GT attempt stopped

The old pipeline was hard-coded to train and validate on Route A. Merely changing
the reference flag removed the GT progress/step safeguards but did not create a
B+C-to-A training split. More importantly, interpolating only the turn waypoints
captured just 69.17% (A), 82.29% (B), and 80.45% (C) of targets in the local
candidate window, below the configured 95% training threshold. The sparse
4.5-m recorded-route anchors in this version produce 100% pre-training candidate
capture on A/B/C while remaining independent of current-frame GT at inference.

The first v37 temporal run exposed a second coordinate bug: Route B's accurate
reference XY (0.13 m mean error) was converted through the waypoint `(s,e)`
representation and reconstructed 412.45 m away on average. Temporal search now
uses the stored reference XY directly; `(s,e)` is retained only for ordered
route-state supervision. The exact forward-4x6 temporal geometry has been
rechecked at 100% candidate capture on A/B/C.

## Run

```bash
bash /yh/study/uav-sat/v37/run_gpu0.sh
```
