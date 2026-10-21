# City A image-only adaptation (supervised warm start)

Training uses no frame GT; test GT is used only for metrics.

| Design | Train loss | test_01 MLE | test_02 MLE |
|---|---:|---:|---:|
| visual_pseudo | 0.4521 | 47.258 | 88.304 |
| temporal_cycle | 0.3438 | 213.354 | 342.086 |
| confidence_gated | 0.2432 | 262.662 | 351.565 |
