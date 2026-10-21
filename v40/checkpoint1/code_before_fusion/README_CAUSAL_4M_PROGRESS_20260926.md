# City A causal localization progress (2026-09-26)

Objective: no current-frame GT at inference; GT supervises training loss only. The
pipeline remains 18 local satellite patches → GRU → Kalman → final Mean Shift.
All numbers below are on City A fixed `val_01` (63 observations); the held-out
`test_02` route has not been used for tuning. Repeated selection on this one
validation route can overfit, so these values are development results.

| Configuration | City A val MLE | Report |
| --- | ---: | --- |
| Previous strict no-GT baseline | 7.202 m | `output/official96_fixed_citya_val_joint8_blend0_20260925/report.json` |
| Rotation-augmented visual checkpoint (epoch 1), 4 rotation TTA, final MS grid 18, bandwidth 25 m, score temperature 0.7, acquisition raw visual weight 20, weak route prior (weight 0.25, sigma 20 m) | **4.618 m** | `output/causal_4m_visual_augmented_mild_epoch1_prior_20260926_a/report.json` |
| Same checkpoint and settings with a 24×24 final candidate grid | 4.615 m | `output/causal_4m_augmented_grid_20260926_a/grid24_bw25/report.json` |

Both selected reports record `valid_no_GT_inference_result: true` and
`current_frame_GT_model_input: false`. The visual checkpoint is
`output/causal_4m_visual_augmented_mild_epoch1_snapshot/checkpoints/citya/visual_retrieval_A_only.pt`;
the temporal checkpoint is
`output/causal_4m_search_20260926_a/temporal_grid/best_temporal_interim.pt`.
The exact environment and checkpoint paths are in the report's
`run_settings.json`.

A one-step training label alignment error was fixed in
`train_multicity_routes.py`: the target for `next_step_se[t]` is now the
future displacement `GT[t+1] - GT[t]`. Checkpoints trained with this fix were
tested, but have not beaten the current best. New GT-loss-only visual runs with
rotation augmentation and one temporal run with an extra image-derived front18
hypothesis are still in progress under `output/causal_4m_visual_augmented_*`
and `output/causal_4m_temporal_extra_image_20260926_a`.

The 4.0 m target has **not** been reached. Frames 10 and 29 each contribute
about 28–32 m error even though the causal acquisition bank contains a nearby
GT candidate. The learned visual matching currently selects the wrong local
window with high confidence on both frames. This is the main remaining issue.

When the two visual runs finish, `finish_augmented_validation.py` writes their
full-route validation reports to `output/causal_4m_visual_augmented_route_20260926_a`
and `output/causal_4m_visual_augmented_mild_route_20260926_a` automatically.

A GT-independence perturbation audit passed: shifting every validation GT by
`[1234, -987]` m changed the measured error but changed predictions and search
centers by exactly `0.0` m. See
`output/causal_4m_best_gt_independence_audit_20260926_a/audit.json`.
The audit checks the core route pipeline without TTA; TTA itself averages only
transformed UAV images.

Two one-epoch GT-loss-only continuation trials with training seeds 2047 and
2059 are running from the best route-validated visual checkpoint. Their automatic
route reports will be under `output/causal_4m_visual_seed2047_route_20260926_a`
and `output/causal_4m_visual_seed2059_route_20260926_a`. These are exploratory
validation runs, not an independent test result.
