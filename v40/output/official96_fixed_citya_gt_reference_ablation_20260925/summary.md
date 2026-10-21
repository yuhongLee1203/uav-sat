# City A fixed-route GT reference search ablation

Same validation frames and frozen model weights. Component rows disable modules
at inference without retraining. GT rows use current-frame GT during
inference; they are oracle diagnostics. Local top1@4m is not official Recall@1.

| Variant | GT input | Local top1@4m % | LSR@15 % | HSR@15 % | MLE m | MHE deg | Frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_gt | 0 | 22.22 | 92.06 | 9.52 | 7.202 | 58.56 | 63 |
| no_gt_no_gru | 0 | 17.46 | 92.06 | 9.52 | 7.608 | 58.54 | 63 |
| no_gt_no_kalman | 0 | 26.98 | 90.48 | 9.52 | 7.917 | 58.55 | 63 |
| no_gt_no_ms | 0 | 22.22 | 76.19 | 9.52 | 10.888 | 58.63 | 63 |
| gt_reference_front18 | 1 | 30.16 | 93.65 | 9.52 | 6.805 | 58.55 | 63 |
| gt_reference_front18_patch | 1 | 100.00 | 92.06 | 9.52 | 6.449 | 58.55 | 63 |
| gt_reference_front18_patch_final_center | 1 | 100.00 | 95.24 | 9.52 | 6.383 | 58.55 | 63 |
| gt_reference_visual_prior_w1 | 1 | 30.16 | 98.41 | 9.52 | 5.404 | 58.55 | 63 |
| gt_reference_visual_prior_w2 | 1 | 30.16 | 98.41 | 9.52 | 4.861 | 58.60 | 63 |
| gt_reference_visual_prior_w4 | 1 | 30.16 | 100.00 | 9.52 | 3.987 | 58.60 | 63 |
| gt_reference_all_patches | 1 | 100.00 | 100.00 | 9.52 | 1.718 | 58.62 | 63 |

SR@20, SPL and NE require a navigation rollout; this validation
episode contains localization frames only.
