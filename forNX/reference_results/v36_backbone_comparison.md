# V36 backbone comparison

Error is pooled over Route B+C frames (not an average of two route averages). Latency is a sample-count-weighted B+C value. Cache is excluded from latency: it is used only for training/evaluation feature reuse, while the timed path runs the real backbone.

| Backbone | Status | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 | LSR@15 | E2E mean (ms) | E2E p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobileclip2_s2 | complete | 3.634 | 6.722 | 46.55 | 75.72 | 98.42 | 99.41 | 68.40 | 73.19 |
| resnet18 | complete | 4.077 | 8.225 | 44.31 | 65.62 | 98.27 | 100.00 | 24.76 | 30.34 |
| mobilenet_v3_small | complete | 4.224 | 8.272 | 42.02 | 63.72 | 96.41 | 100.00 | 25.32 | 27.21 |
| vgg16 | complete | 4.277 | 8.771 | 41.51 | 63.72 | 97.28 | 100.00 | 20.88 | 21.72 |
