# Historical shared ABCD checkpoint on synthetic c4m validation episodes

These are GT-selected synthetic episodes, not recorded flights or official Table 6.
GT variants consume each validation frame's label during inference.
The historical shared checkpoint was trained under a different protocol than
the fixed-route City A pilot. Its results must not be pooled with that pilot.

| City | Variant | Frames | Local top1@4m % | LSR@15 % | HSR@15 % | MLE m | MHE deg |
|---|---|---:|---:|---:|---:|---:|---:|
| citya | no_gt | 73 | 0.00 | 0.00 | 9.59 | 301.494 | 72.39 |
| citya | gt_reference_prior_w4 | 73 | 8.22 | 100.00 | 17.81 | 2.719 | 61.92 |
| citya | gt_patch_oracle | 73 | 100.00 | 100.00 | 17.81 | 1.805 | 61.83 |
| cityb | no_gt | 80 | 0.00 | 0.00 | 11.25 | 177.709 | 95.46 |
| cityb | gt_reference_prior_w4 | 80 | 3.75 | 100.00 | 23.75 | 2.653 | 54.08 |
| cityb | gt_patch_oracle | 80 | 100.00 | 100.00 | 26.25 | 1.578 | 50.84 |
| cityc | no_gt | 77 | 0.00 | 0.00 | 6.49 | 308.949 | 70.35 |
| cityc | gt_reference_prior_w4 | 77 | 7.79 | 100.00 | 22.08 | 2.854 | 55.54 |
| cityc | gt_patch_oracle | 77 | 100.00 | 100.00 | 23.38 | 1.689 | 53.43 |
| cityd | no_gt | 72 | 0.00 | 0.00 | 9.72 | 271.988 | 69.91 |
| cityd | gt_reference_prior_w4 | 72 | 11.11 | 100.00 | 18.06 | 2.315 | 53.22 |
| cityd | gt_patch_oracle | 72 | 100.00 | 100.00 | 19.44 | 1.628 | 49.11 |

Official R@1 requires the four reference tiles per independent query.
SR@20, SPL and NE require navigation rollouts and cannot be inferred
from these localization frames.
