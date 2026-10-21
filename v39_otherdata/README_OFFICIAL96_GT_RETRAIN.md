# Official96 c4m temporal GT experiment: train from scratch and evaluate

One command, from any directory:

```bash
bash /yh/study/uav-sat/v39_otherdata/run_official96_gt_retrain_all_3gpu.sh
```

This command creates a new timestamped output directory. It copies only the
previously prepared City A/B/C/D Official96 temporal episodes, checks the
seed-42 76,500/4,500/9,000 split, and starts with no checkpoints. It trains
three shared ABCD models from random initialization: 3 frames on GPU 0,
2 frames on GPU 5, and 1 frame on GPU 6. Each frame setting visits the city
training episodes in sequence; validation occurs inside training. No checkpoint
from the historical 3.759 m run or earlier Official96 runs is imported.

After training, the command evaluates all ten variants on the Official96 test
episodes: Full, without GRU, without Kalman, without MeanShift, 1 frame,
2 frames, and MeanShift windows 4x4, 5x5, 7x7, and 8x8. The default Full window
is 6x6. Evaluation also uses GPUs 0, 5, and 6. All variants use the existing
`controlled_gt_jitter` inference protocol.

The command prints the new output directory. Its main outputs are:

- `paper_ablation_tables.md`: pooled component, temporal, and window ablations.
- `official96_gt_all_metrics.md`, `.json`, `.csv`: pooled LSR@3/5/15,
  HSR@15, MLE, MHE, R@1*, and offline SR@20/SPL/NE proxies.
- `logs/train_frames{1,2,3}_gpu{6,5,0}.log` and evaluation logs.
- `FRESH_TRAINING_PROTOCOL.txt`: input and training provenance.

R@1* is a same-quadrant score from final XY, rather than official four-tile
retrieval Recall@1. SR@20, SPL, and NE are offline route proxies because the
reassembled images do not contain a recorded closed-loop flight. The original
3.759 m belongs to different 4 m routes; a freshly trained Official96 run is
a new experiment and must be reported with its own measured error.
