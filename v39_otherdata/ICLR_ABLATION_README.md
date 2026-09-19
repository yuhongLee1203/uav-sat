# ICLR Bearing-UAV ablation

The paper-facing architecture is fixed before held-out evaluation:

`6x6 geometry -> heading-guided forward 3x6 (18 scored patches) -> 3-frame GRU -> fixed-R Kalman -> one final 6x6 MeanShift -> XY`

- Route A is the only supervised training route.
- Routes B/C preserve the existing `controlled_gt_jitter` protocol and its GT/reference behavior. The ablation runner does not change the Bearing-UAV dataset loader, route preparation, or labels.
- The front visual posterior is summarized by a weighted centroid. MeanShift appears exactly once, after Kalman.
- The default final window is 6x6 (36 candidates), matching the selected V39 setting. Windows 4x4--8x8 are reported as sensitivity analysis, not selected on B/C.
- Component removals reuse the exact same trained full checkpoint and held-out frames.
- If an ablation beats Full, the audit reports `FULL_TREND_CHECK=FAIL`; it never edits or suppresses a result.

One-command run:

```bash
cd /yh/study/uav-sat
git fetch origin v39_otherdata
git show origin/v39_otherdata:v39_otherdata/bearing_iclr_ablation.py > v39_otherdata/bearing_iclr_ablation.py
git show origin/v39_otherdata:v39_otherdata/build_iclr_ablation_tables.py > v39_otherdata/build_iclr_ablation_tables.py
git show origin/v39_otherdata:v39_otherdata/run_bearing_iclr_ablation.sh > v39_otherdata/run_bearing_iclr_ablation.sh
chmod +x v39_otherdata/bearing_iclr_ablation.py v39_otherdata/build_iclr_ablation_tables.py v39_otherdata/run_bearing_iclr_ablation.sh
TEMPORAL_EPOCHS=80 PATIENCE=4 UPLOAD_RESULTS=1 bash v39_otherdata/run_bearing_iclr_ablation.sh
```

GPU scheduling is `citya -> cityd` on GPU 0, `cityb` on GPU 5, and `cityc` on GPU 6.
