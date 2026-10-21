# V36 backbone comparison

Error is pooled over Route B+C frames (not an average of two route averages). Latency is a sample-count-weighted B+C value. Cache is excluded from latency: it is used only for training/evaluation feature reuse, while the timed path runs the real backbone.

| Backbone | Status | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 | LSR@15 | E2E mean (ms) | E2E p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobileclip2_s2 | complete | 3.634 | 6.720 | 46.52 | 75.75 | 98.42 | 99.41 | 188.55 | 191.44 |
| resnet18 | complete | 4.077 | 8.228 | 44.31 | 65.65 | 98.27 | 100.00 | 95.07 | 97.85 |
| mobilenet_v3_small | complete | 4.224 | 8.274 | 42.02 | 63.70 | 96.41 | 100.00 | 88.83 | 91.60 |
| vgg16 | complete | 4.277 | 8.771 | 41.51 | 63.72 | 97.28 | 100.00 | 137.35 | 147.53 |
