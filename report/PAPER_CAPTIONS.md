# Draft figure captions

**Main MLE and LSR@15 comparison.** Published UAV-view values for University-1652, SUES-200, DenseUAV, GTA-UAV, Bearing-UAV, and Bearing-UAV with weather augmentation are taken from Liu et al., CVPR 2026, Table 2. Our Method is evaluated on a separate 1,482-frame synthetic-route replay with a supplied per-frame GT-assisted local/path reference. Protocols are not directly comparable; bars provide context rather than a controlled benchmark ranking.

**Component ablation.** Mean localization error and localization success within 3 m for four matched 1,482-frame variants. Final spatial refinement has the largest observed effect. GRU removal also degrades both metrics. Kalman removal changes aggregate MLE by 0.004 m and LSR@3 by 0.61 percentage points.

**Temporal input.** Separately trained one-, two-, and three-frame variants, each retaining recurrent hidden state. Three frames obtain the lowest MLE and highest LSR@3 on the GT-assisted replay.

**Final neighborhood accuracy and runtime.** Final Mean Shift is evaluated with 4×4–8×8 candidate windows on the same 1,482 observations. The original Full model uses 6×6; accuracy changes little beyond 5×5, whereas measured Mean Shift time rises with window size. Timing values follow the replay's timing summary.

**City-level performance.** MLE, LSR@5 and P90 for the Full model on the eight GT-assisted City A–D test routes. City C has the highest MLE and P90 in this set.

**Reference improvement.** Per-city localization error and LSR@5 for the supplied GT-centered jittered reference and Full prediction. The reference is an oracle-assisted input, not an independent no-GT trajectory.

**Error CDF.** Cumulative fraction of the same 1,482 observations below each localization-error threshold for reference, w/o GRU, w/o final refinement, and Full.

**Qualitative routes.** Two held-out synthetic routes per city with GT, per-frame GT-assisted reference centers, and Full predictions. Start and end GT points are marked by circles and crosses.
