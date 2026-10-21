# Original 3.759 m GT-assisted localization retest

Run the original City A/B/C/D v13 test again on GPUs 0, 5, and 6:

    bash /yh/study/uav-sat/v39_otherdata/run_original_3759_inference_all_3gpu.sh

The wrapper copies the historical checkpoints and prepared 4 m synthetic routes
to a new result directory. It then reruns all ten variants on 1,482 test frames.
This is inference only; it does not retrain the model. The command prints the
new result directory at completion.

Use original_3759_all_metrics.md as the localization result. It reports
LSR@3, LSR@5, LSR@15, and MLE for Full, component removal, temporal input,
and MeanShift window experiments. publication_audit.md gives whole-route
bootstrap intervals and records the test-time GT-assisted protocol.

The saved heading head was trained against motion direction between neighboring
points on the synthetic route. The Bearing-UAV image metadata instead provides
yaw_deg, and the two target definitions do not agree. Therefore the wrapper
excludes official HSR/MHE from its main report. A legacy table with synthetic
route-motion heading diagnostics is renamed to
route_motion_heading_diagnostic.md. It must not be submitted as Bearing-UAV
image-heading performance.

The model did not output an official four-tile retrieval ranking or a closed-loop
navigation rollout, so Recall@1, SR@20, SPL, and NE are also excluded.
Current-frame GT is used to center the search and select the acquisition
hypothesis. These results must be described as a GT-assisted/oracle experiment,
not no-GT visual localization.
