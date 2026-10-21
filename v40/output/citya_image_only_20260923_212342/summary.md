# City A image-only adaptation (supervised warm start)

Training uses no frame GT; test GT is used only for metrics.

| Design | Train loss | test_01 MLE | test_02 MLE |
|---|---:|---:|---:|
| visual_pseudo | 0.1564 | 193.271 | 334.920 |
| temporal_cycle | 0.7403 | 194.500 | 335.362 |
| confidence_gated | 0.1046 | 193.441 | 335.075 |
