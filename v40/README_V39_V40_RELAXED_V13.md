# v39 four-city GT-assisted method with applicable v40 fixes

Run from any directory:

```bash
bash /yh/study/uav-sat/v40/run_v39_v40_relaxed_v13_ablation_3gpu_v2.sh
```

This run uses the **original** `soft_sequence_v13_multicity_auto_train_fullroute`
4 m prepared routes, not the later Official96 8 m reassembly. The quoted
w/o GRU 4.275, w/o Kalman 3.763, w/o final Mean Shift 6.785, and Full
3.759 m are pooled over City A/B/C/D's eight test episodes (1,482 frames).
They are historical baselines; changes below define a new experiment.

The command creates a new `v40/output/v39_v40_relaxed_v13_*` directory and
copies only prepared data from the original run. No old model checkpoint or
Kalman calibration file is copied. Three shared ABCD models are trained from
scratch: 3 frames on GPU 0, 2 frames on GPU 5, 1 frame on GPU 6. Model selection
and Kalman calibration use the training route's internal validation portion.
The 1/2-frame groups start their test evaluation on GPUs 6/5 as soon as their
training and city-gallery binding finishes, while the 3-frame group continues.
Once the 3-frame model finishes, its four city galleries are rebuilt in
parallel on GPUs 0/5/6. The remaining variants then run across those GPUs.
Component and final-window ablations
remove or vary a component at inference using the newly trained 3-frame model;
they are not separately retrained architectures.

## v40 audit applied to this v39 runtime

- The original fixed longitudinal and lateral motion/Kalman caps are replaced
  by bounds derived only from each city's `train_01` maximum XY frame step.
  The route-frame smoothing radius is zero for the independent-image pseudo
  routes. The 3x6 front search and 6x6 final grid remain the declared v39
  architecture, not tunable motion caps.
- The current test frame's GT no longer truncates predicted progress, the
  Kalman update, or the allowed frame step. GT **still selects the controlled
  search reference patch**, as the user requested. This is a GT-assisted
  result, not an image-only or causal no-GT result.
- The temporal measurement loss uses original metadata XY through a
  differentiable copy of the route coordinate conversion; motion labels use
  raw adjacent metadata XY displacement in the route frame. On these original
  training routes, monotonic GT projection round trips differed from metadata
  XY by 0.41/0.51/0.41/0.53 m on average for A/B/C/D; maxima were
  6.65/9.42/6.83/8.10 m.
- Frame-count feature caches are isolated, and the command refuses an
  existing output directory so a previous checkpoint cannot be resumed.

The final output is `v39_v40_relaxed_ablation.md` with MLE and LSR@5 for Full,
component removal, temporal input, and MeanShift windows. The existing v39
`paper_ablation_tables.md` adds pooled R@1*, LSR@15, HSR@15, and MHE.
`PROTOCOL.txt` and the training/evaluation logs record the run. R@1* is a
same-quadrant proxy, not four-tile retrieval Recall@1. The v13 episodes are
synthetic spatial sequences; their test routes have already been inspected in
past experiments, so they are not untouched paper-final held-out tests.

The v40 no-GT causal bank and whole-gallery visual protocol are different
experiments. Importing those switches here would remove the requested GT
reference and change the v39 architecture, so this command does not claim their
no-GT scores. The 3.759 m baseline is not guaranteed after retraining and
changing the loss and limits.
