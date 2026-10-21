# c4m Official96 reassembled temporal episodes, GT-assisted v39 ablation

R@1* is a same-quadrant proxy from final XY. Official Recall@1 needs a
dedicated four-reference-tile ranking output. SR@20/SPL/NE below are offline
synthetic-route proxies, because these images are not a recorded flight or
closed-loop navigation rollout. GT enters inference through controlled jitter.

| Variant | R@1* ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE m ↓ | MHE deg ↓ | SR@20 proxy ↑ | SPL proxy ↑ | NE proxy m ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 56.67% | 31.87% | 17.50% | 99.777 | 66.77 | 25.00% | 21.87% | 224.399 |
| no_gru | 56.46% | 31.98% | 17.40% | 99.829 | 66.73 | 25.00% | 21.60% | 224.313 |
| no_kalman | 56.98% | 32.19% | 17.71% | 99.689 | 66.80 | 25.00% | 21.93% | 224.398 |
| no_ms | 58.12% | 37.92% | 17.60% | 98.447 | 66.96 | 25.00% | 19.15% | 223.259 |
| frames1 | 56.77% | 31.77% | 17.50% | 99.775 | 66.78 | 25.00% | 21.95% | 224.405 |
| frames2 | 56.46% | 31.87% | 17.40% | 99.776 | 66.77 | 25.00% | 21.88% | 224.409 |
| grid4 | 56.88% | 33.54% | 17.50% | 99.577 | 67.06 | 25.00% | 21.13% | 224.710 |
| grid5 | 56.46% | 32.71% | 17.40% | 99.690 | 67.14 | 25.00% | 21.70% | 224.399 |
| grid7 | 56.04% | 31.56% | 17.60% | 99.824 | 66.67 | 25.00% | 22.10% | 224.397 |
| grid8 | 56.04% | 31.25% | 17.81% | 99.892 | 66.52 | 25.00% | 22.24% | 224.397 |
