# Report-ready localization results

## 1. Main comparison

| Method | MLE (m) ↓ | LSR@15 (%) ↑ | Source/protocol |
| --- | ---: | ---: | --- |
| University-1652 | 33.15 | 15.11 | Bearing-UAV paper, UAV view |
| SUES-200 | 30.83 | 15.76 | Bearing-UAV paper, UAV view |
| DenseUAV | 28.79 | 16.54 | Bearing-UAV paper, UAV view |
| GTA-UAV | 28.43 | 27.96 | Bearing-UAV paper, UAV view |
| Bearing-UAV | 8.61 | 89.36 | Bearing-UAV paper, UAV view |
| Bearing-UAV + WA | 7.48 | 92.88 | Bearing-UAV paper, UAV view, weather augmentation |
| Our Method | **3.759** | **100.00** | Original v13 replay, **current-frame GT-assisted local/path reference** |

The six published values are from [Bearing-UAV Table 2](https://arxiv.org/abs/2603.22153). **The protocols are not directly comparable:** Our Method receives a local/path reference whose current center and acquisition selection use GT during evaluation. Do not state that this table proves superiority under the published methods' input conditions. Figures: [MLE](figures/main/main_mle.pdf), [LSR@15](figures/main/main_lsr15.pdf).

## 2. Component removal

| Variant | MLE (m) ↓ | LSR@3 (%) ↑ |
| --- | ---: | ---: |
| w/o GRU | 4.275 | 32.66 |
| w/o Kalman | 3.763 | 41.30 |
| w/o Final Refinement | 6.785 | 15.45 |
| Full Model | **3.759** | **41.90** |

On these paired frames, final refinement makes the largest measured difference (6.785→3.759 m and 15.45→41.90%); GRU removal raises MLE by 0.516 m. Kalman removal changes aggregate MLE by only 0.004 m and LSR@3 by 0.61 percentage points. The prior route-level interval for the Full-minus-no-Kalman MLE difference includes zero, so this is a small observed difference rather than firm evidence of an accuracy gain. [Figure](figures/ablation/component_ablation.pdf).

## 3. Explicit temporal input

| Input | MLE (m) ↓ | LSR@3 (%) ↑ |
| --- | ---: | ---: |
| 1 frame | 3.883 | 39.81 |
| 2 frames | 3.921 | 39.14 |
| 3 frames | **3.759** | **41.90** |

Three frames perform best in these separately trained variants. Persistent recurrent state is active for all three rows, so explicit frame context contributes beyond that state on this replay. [Figure](figures/ablation/temporal_window.pdf).

## 4. Final neighborhood

| Final window | MLE (m) ↓ | LSR@3 (%) ↑ | Mean Shift time (ms) ↓ |
| --- | ---: | ---: | ---: |
| 4×4 | 4.168 | 37.65 | 4.04 |
| 5×5 | 3.763 | 41.90 | 5.00 |
| 6×6 (Full) | **3.759** | **41.90** | 7.52 |
| 7×7 | 3.738 | 42.04 | 9.50 |
| 8×8 | 3.738 | 42.04 | 10.27 |

Accuracy largely saturates at 5×5. The original Full configuration uses 6×6 and is 0.004 m lower in MLE than 5×5, but its reported MS time is 2.52 ms higher. A 7×7 window is another 0.021 m lower at 1.98 ms more than 6×6. Thus 6×6 is the paper's chosen accuracy/latency compromise; the plotted numbers should be reported rather than claiming it is the unique optimum. [Accuracy figure](figures/ablation/final_neighborhood_accuracy.pdf), [runtime figure](figures/ablation/final_neighborhood_runtime.pdf).

## 5. Per-city and per-route results

| City | Frames | Full MLE (m) ↓ | Full P90 (m) ↓ | Full LSR@5 (%) ↑ | Reference MLE (m) ↓ | Reference LSR@5 (%) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| City A | 380 | 3.653 | 6.696 | 74.47 | 5.390 | 21.05 |
| City B | 317 | 3.708 | 6.652 | 72.87 | 5.596 | 9.78 |
| City C | 374 | 3.944 | 7.494 | 70.59 | 5.365 | 22.99 |
| City D | 411 | 3.728 | 6.781 | 76.64 | 5.190 | 29.44 |
| Pooled | **1,482** | **3.759** | — | **73.75** | **5.372** | **21.46** |

City C is hardest by MLE and P90, while City A has the lowest MLE. The city MLE range is 0.290 m; the equal-city standard deviation is 0.110 m. Full improves MLE against the GT-assisted reference in all eight routes. See the [per-city plots](figures/city), [baseline plots](figures/baseline) and [route table](tables/per_route_reference_improvement.csv). The reference baseline uses GT-centered jitter and must be named that way in the paper.

## 6. Error CDF and trajectories

[Error CDF](figures/baseline/error_cdf.pdf) uses the actual paired frame errors for GT-assisted reference, w/o GRU, w/o final refinement and Full, for thresholds 0–20 m. No curve is reconstructed from aggregate means. The City [A](figures/trajectory/citya_routes.pdf), [B](figures/trajectory/cityb_routes.pdf), [C](figures/trajectory/cityc_routes.pdf), and [D](figures/trajectory/cityd_routes.pdf) route figures overlay GT, GT-assisted reference center, and Full prediction on both held-out routes in each city.

## Reporting boundary

These results were replayed with historical v13 weights; they were **not retrained for this figure export**. Current-frame GT assists the acquisition. They cannot be relabeled as the v40 no-GT inference result, and the main comparison should retain the visible protocol warning. Source details: [PROTOCOL.txt](../v39_otherdata/output/original3759_inference_20260925_220434_1615534/PROTOCOL.txt), [original audit](../v39_otherdata/output/original3759_inference_20260925_220434_1615534/publication_audit.md), and [source manifest](data/source_manifest.json).
