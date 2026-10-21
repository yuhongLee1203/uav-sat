# Official96 metadata, fixed-waypoint synthetic episodes (v40)

`Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv`
contains geotagged, independent images; it has no timestamp or episode ID.
Consequently, these experiments are **not** the original Bearing-UAV Table 6
and must not be described as recorded UAV trajectories.

The `prepare_official96_fixed_routes.py` adapter keeps the local ID96
seed-42 train/validation/test split. It uses GT **offline** to choose images
near predeclared v39 waypoint lines and order them into synthetic episodes.
The held-out waypoint files themselves are fixed before those images are
chosen; unlike the earlier `official_rsi96_seed42_temporal_v1` adapter, they
are not sampled from validation/test GT. No episode crosses a city or split.
The City A fixed manifests have 66 train, 63 validation, 64 test_01 and 89
test_02 frames, with no repeated image paths between the four episodes.

The model modules remain the v39 MobileNetV3-Small visual model, three-frame
GRU, front-18 local candidates, RouteKalman and MeanShift. Current-frame GT
never enters the forward pass or search center. Training uses GT only inside
losses; visual training scores the full fixed satellite gallery rather than
GT-selected patches. `prepare_official96_fixed_train_extra.py` can create
train-only 20/30 m-step episodes, but its pilot did not improve validation.

City A baseline (fixed waypoint, original 44.8 m SAT crop): best validation
MLE 17.364 m; test_01 10.336 m; test_02 8.891 m. Increasing Kalman
correction and tuning MeanShift on validation lowered validation to 11.378 m;
stronger image-based candidate selection lowered it to 10.566 m; two-rotation
test-time averaging reached 9.998 m with an *older* visual checkpoint. That
checkpoint records a different held-out test manifest hash, so the 9.998 m
number is exploratory and is **not eligible** for selecting a fixed-route
model. Fresh checkpoints now use the fixed-route split hash, which evaluation
checks before running. None of these are new held-out test claims. The 4 m
gate has not passed; do not run or report ABCD as a 4 m model yet.

Relevant scripts:

- `prepare_official96_fixed_routes.py`: fixed held-out waypoint adapter.
- `train_visual_global_gtloss.py`: whole-gallery GT-loss-only visual heads.
- `train_visual_backbone_gtloss.py`: optional shared MobileNetV3-Small weight
  fine-tuning; refreshes the entire satellite gallery after each epoch.
- `train_multicity_routes.py`: shared GRU/Kalman episode training, state reset
  at each route boundary. Use `--skip-test` during model selection.
- `evaluate_official96_validation.py`: validation-only inference and parameter
  sweeps; optional TTA with isolated feature cache.
- `evaluate_official96_fixed_test.py`: one-shot held-out test after selecting
  the configuration on validation.

All outputs are under `v40/output/`; the original v39 files and results were
left untouched.

The temporal supervision now compares the predicted measurement in XY against
the original metadata XY, using a differentiable copy of the existing route
coordinate conversion. The former monotonic GT projection produced incorrect
targets on two off-route training episodes (145.894/270.332 m reconstruction
MLE). Affected runs contain `INVALIDATED.md`; their logs remain available.
Motion labels now use raw frame displacement expressed in the model's own
route frame. No GT chooses the model frame or forward search candidates.

`UAVSAT_CORR_PARALLEL_M` / `UAVSAT_CORR_CROSS_M` control existing GRU residual
output scales (legacy defaults: 0.75/0.50 m). New temporal checkpoints record
these values, and evaluation rejects inconsistent values. Episode backbone
caches have separate persistent directories to prevent one training route
from overwriting another's cache.

Visual backbone training supports `--live-sat-topk` to backpropagate through
image-selected satellite patches as well as UAV images. Selection uses image
scores and uniform sampling, and all gallery positions remain in the loss.
The full gallery is refreshed after each epoch because backbone weights change.
`--uav-quarter-turn-augment` rotates only training images around their centers;
the position labels stay unchanged. Neither option changes network modules.

2026-09-25 ongoing validation (not held-out test): the fixed epoch-1 rotation
augmentation snapshot from `official96_fixed_citya_joint8_rotation_snapshot1_20260925`
gave route MLE **8.312805 m**, P90 **15.281563 m**, LSR@15 **88.888889%**.
It uses the standard last8 temporal checkpoint, Q-progress/Q-cross 10, route
frame smoothing radius 0, and no inference TTA. The complete parameter record
is `output/official96_fixed_citya_val_joint8_rot1_20260925/run_settings.json`.
The current frame actually runs through MobileNetV3-Small; GRU, Kalman and
final MeanShift are enabled. This remains above the requested 4 m threshold.

`export_visual_interim.py` can snapshot an in-progress run's best weights for
validation without stopping training. It must receive that run's exact source
checkpoint. `finish_citya_joint_validation.py` waits for the three named live
training processes, evaluates their final best visual weights on the same fixed
validation episode, and writes all outcomes to `summary.json`. It neither
loads the held-out test images nor claims equivalence to official Table 6.

The City A same-checkpoint GT oracle diagnosis is in
`output/official96_fixed_citya_oracle_diagnostic_20260925/report.md`.
GT-centered front18 yields 7.447 m route MLE; choosing the GT-nearest front18
patch yields 6.551 m; also centering and choosing the GT-nearest final-MS
patch yields 1.718 m. These are **GT-input validation oracles**, not valid
image-only model scores. The best completed no-GT trial from the first
eight-epoch continuation is 8.071 m route validation MLE. The second
training series includes GT-nearest satellite patches only in its training
gradient calculation; checkpoint provenance records this, while held-out
inference still receives no frame GT.

Using the same no-GT City A visual checkpoint after eight more epochs, moving
the final MS lattice center halfway between the visual anchor and the Kalman
estimate and removing its Kalman spatial prior lowered fixed-route validation
MLE to **7.202326 m**. The record is
`output/official96_fixed_citya_val_joint8_blend0_20260925/report.json`.
The matching `run_settings.json` records all environment parameters. This is
still a validation result; the held-out test routes remain unevaluated.

The City B fixed-route adapter required a 50 m observation-selection radius
to get at least 24 frames on its 5% validation split (40 m yielded 17 frames).
Its waypoints remained predeclared and independent of selected GT. The
prepared City B episode lengths are train 66, validation 64, test_01 64,
test_02 72. City B uses a different radius from City A and its scores must
state that construction difference.
