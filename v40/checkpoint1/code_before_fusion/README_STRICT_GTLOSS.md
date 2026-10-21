# Strict GT-loss-only City A pilot

The v40 code retains the v39 `AllMapGeoCLIP`/`ThreeFrameRouteStateGRU`
classes and fixes the backbone to ImageNet-pretrained MobileNetV3-Small.
Current-frame UAV images go through this backbone; the previous two frame
features come from a cache. The route path remains front-18 visual patches,
GRU, Kalman, final MeanShift, output.

Training uses GT coordinates only to construct visual and temporal losses.
The visual stage scores the entire fixed satellite gallery on every forward
pass. It does not select candidates using GT. Its task heads start randomly;
no GT-prior-trained task checkpoint is loaded. The temporal stage also starts
randomly and uses causal search centres and predicted recurrent/Kalman state.
At inference, GT is read only after prediction to compute metrics.

The prepared `train_01`, `test_01` and `test_02` routes are custom subsets of
the Bearing-UAV-90K images. They are not native continuous flights. The
official c4m metadata has independent cross-view rows, including four SAT
patches from the GT block, and does not itself provide ordered three-frame
episodes. Therefore these route results are **not** official Table 6 results.

Run the complete strict City A pilot with:

```bash
bash v40/run_strict_gtloss_citya_pilot.sh
```

Set `V40_DENSE_OFFBANK_LOSS=1` to keep temporal supervision active when the
causal search window misses the GT. This changes only the training loss, not
the model inputs or architecture. Feature caches are reused under
`v40/cache/visual90k`; set `V40_VISUAL_SAMPLES`, `V40_VISUAL_EPOCHS`,
`V40_TEMPORAL_EPOCHS`, `V40_VISUAL_GPU` and `V40_TEMPORAL_GPU` as needed.

The 10,000-image, three-epoch City A visual pilot reached 53.318 m independent
metadata-validation global top-1 MLE. Two epochs of strict temporal training
reached 79.022 m on City A `test_01` and 80.905 m on `test_02`. These are
exploratory debugging numbers, not an independent final paper estimate and
far above the requested 4 m threshold. An ABCD/c4m final run should not be
represented as validated until a held-out strict pilot meets the chosen
criterion.

Increasing to 20,000 images and 15 visual epochs improved global validation
top-1 MLE to 13.366 m. The corresponding two-epoch causal route experiment
with the original narrow search and Kalman limits still yielded 73.801 m on
`test_01` and 74.489 m on `test_02`. A narrower 2 m Gaussian loss target was
worse than the default 4 m target in the matched visual experiment.

Code audit found the strict no-GT branch had disabled visual step relaxation
and capped the Kalman step at 6 m even though prepared-route frame cadence can
exceed that. The v40 fix retains the Kalman filter, raises its default step cap
to 14 m, and permits bounded image-driven correction. With a 13-hypothesis,
60 m causal bank and the 20k/15-epoch visual checkpoint, the exploratory
City A route MLE is **7.929 m** (`test_01`) and **7.697 m** (`test_02`)
after two temporal epochs and bounded Kalman correction. It is not <4 m.
Six or ten temporal epochs, and two alternative MeanShift settings, did not
improve both routes. These test routes have now been used for debugging and
must not be called untouched final test data in a paper.

The integrated pilot script was rerun end to end after the Kalman fix. Its
reproduced result is 7.937 m (`test_01`) and 7.560 m (`test_02`), saved under
`v40/output/strict_runner_verify_20260924/temporal/`. Distance-regularized
visual losses improved `test_01` to 7.213 m at best but worsened `test_02`
to 8.210 m, so they are not the default. No tested strict setting met <4 m on
both routes. Do not launch or report an ABCD/c4m Table-6-style final run as
validated on the basis of these City A experiments.
