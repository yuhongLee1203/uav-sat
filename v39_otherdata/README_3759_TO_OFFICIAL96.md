# Reproduce 3.759 m and transfer its weights to Official96 temporal episodes

Run one command from any directory:

```bash
bash /yh/study/uav-sat/v39_otherdata/run_official96_gt_from_3759_3gpu.sh
```

The source is `output/abcd_shared_20260922_121010`: its V7 component table
reports pooled 1/2/3-frame MLE 3.883/3.921/3.759 m on 1,482 frames. The
source uses `soft_sequence_v13_multicity_auto_train_fullroute` episodes with
4 m waypoint sampling, `controlled_gt_jitter`, three shared ABCD frame-count
checkpoints, city-specific satellite galleries, and the recorded
`positive_three_frame_path_primary_v7` Kalman profiles.

The command copies these exact weights and first re-evaluates the original
4 m episodes. It checks all nine pooled MLE/LSR@3/LSR@5 values before transferring the
weights to the previously reassembled Official96 c4m episodes. Transfer only
rebuilds each city's satellite gallery for the new route origin; no model
training occurs. GPU 0/5/6 run the independent gallery rebuilds and all ten
full/component/frame/grid evaluations concurrently. Outputs are separate:

- `output/official96_gt_3759_transfer_20260925/reproduction_v13_4m/reproduced_temporal_3_5.md`
- `output/official96_gt_3759_transfer_20260925/reproduction_v13_4m/paper_ablation_tables.md`
- `output/official96_gt_3759_transfer_20260925/official96_temporal_8m/official96_gt_all_metrics.md`

The original 3.759 m is a result on the 4 m v13 episodes. Official96 uses
8 m target spacing and much larger actual gaps between independent images.
Direct City A Full transfer measured 112.457/154.912 m on its two Official96
test episodes, so the transfer command does not promise 3.759 m on the new
data. The source training routes reuse 20 images in Official96 test episodes;
`source_train_overlap_audit.json` records those paths. Consequently the
transfer output cannot be treated as an independent held-out score.

The combined report's R@1* is a same-quadrant proxy from final XY, not the
official four-tile retrieval Recall@1. Its SR@20/SPL/NE are offline route
proxies because the reassembled episodes are not recorded closed-loop flights.
