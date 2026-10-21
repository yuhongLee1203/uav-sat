# City A fixed-route validation: GT oracle diagnosis

All rows use the same MobileNetV3-Small visual snapshot
`official96_fixed_citya_joint8_rotation_snapshot1_20260925`, the same three-frame
GRU checkpoint `official96_fixed_citya_temporal_last8_standard_20260925`, the
same 63-frame validation episode, Kalman Q progress/cross 10, route-frame
smoothing radius 0, and no TTA. No held-out test route was evaluated.

| Current-frame GT use | Front18 raw Top1 MLE | Front18 SoftMS MLE | Kalman MLE | Final MLE | Source |
|---|---:|---:|---:|---:|---|
| None | 8.357 m | 8.328 m | 11.099 m | 8.313 m | [normal](../official96_fixed_citya_val_joint8_rot1_20260925/report.json) |
| Center the front18 window | 6.554 m | 6.274 m | 8.037 m | 7.447 m | [front18 center oracle](../official96_fixed_citya_val_gtfront18_oracle_20260925/report.json) |
| Center front18 and choose its nearest GT patch | 1.966 m | 1.966 m | 4.914 m | 6.551 m | [front18 patch oracle](../official96_fixed_citya_val_gtfront18_patch_oracle_20260925/report.json) |
| Above, plus center final MS candidates at GT | 1.966 m | 1.966 m | 4.914 m | 5.910 m | [final MS center oracle](../official96_fixed_citya_val_gt_all_mscenter_20260925/report.json) |
| Above, plus choose final MS nearest GT patch | 1.966 m | 1.966 m | 4.914 m | 1.718 m | [final MS patch oracle](../official96_fixed_citya_val_gt_all_mspatch_20260925/report.json) |

Each GT oracle row feeds the current validation frame's GT into candidate
selection, and is **not a valid no-GT inference result**. The nearest-patch
oracle modifies the similarity logit for the GT-nearest existing patch so it
is selected; it does not represent learned localization. The 1.718 m result
is a dataset and grid diagnostic only. It cannot be presented as a model
result, a test score, or Bearing-UAV Table 6 reproduction.

The front18 nearest-center quantization error is about 1.97 m, while the
normal visual decoder has about 8.33 m MLE. Kalman and the final MS add bias
even when the first patch is correct. Thus the remaining no-GT work is to
improve visual correspondence and reduce filter bias without test GT.
