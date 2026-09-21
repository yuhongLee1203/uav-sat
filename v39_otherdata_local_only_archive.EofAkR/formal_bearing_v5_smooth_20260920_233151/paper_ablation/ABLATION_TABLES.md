# Paper-facing ablation tables

> 1/2/3-frame temporal-context rows use separately trained temporal checkpoints.
> Kalman / MeanShift / search-policy / decoder / heading-feedback / jitter / grid rows are one-factor inference-component or sensitivity ablations using the Full checkpoint.
> All held-out nav50/nav51 results are measured outputs; no row is edited to force Full to win.

## Core Components

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/o GRU (inference removal) | inference/component or sensitivity ablation using Full checkpoint | 4.158 | 0.500 | 7.053 | 68.792 | 99.932 | -0.068 | 37.725 | 0.322 | 37.042 | -0.339 | 0.882 | 24.670 | 99.050 | 54.923 | 18.207 |
| w/o Kalman | inference/component or sensitivity ablation using Full checkpoint | 3.658 | -0.001 | 6.535 | 76.526 | 100.000 | 0.000 | 37.374 | -0.030 | 37.313 | -0.068 | 0.746 | 24.926 | 99.050 | 50.577 | 19.772 |
| w/o final MeanShift | inference/component or sensitivity ablation using Full checkpoint | 6.658 | 3.000 | 11.497 | 34.600 | 98.168 | -1.832 | 38.613 | 1.209 | 36.703 | -0.678 | 7.598 | 27.168 | 99.050 | 44.418 | 22.513 |
| w/o learned heading feedback | inference/component or sensitivity ablation using Full checkpoint | 3.659 | 0.001 | 6.502 | 76.934 | 100.000 | 0.000 | 37.398 | -0.005 | 37.246 | -0.136 | 0.746 | 24.657 | 99.050 | 50.755 | 19.702 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |

## Temporal Context Retrained

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 frame (retrained temporal) | retrained temporal-context ablation | 3.641 | -0.018 | 6.392 | 76.934 | 100.000 | 0.000 | 37.537 | 0.133 | 37.585 | 0.204 | 0.475 | 24.575 | 98.982 | 53.501 | 18.691 |
| 2 frames (retrained temporal) | retrained temporal-context ablation | 3.634 | -0.024 | 6.379 | 77.544 | 100.000 | 0.000 | 37.468 | 0.064 | 37.585 | 0.204 | 0.543 | 24.622 | 99.050 | 52.921 | 18.896 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |

## Search Policy

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full 6x6 scoring (36 candidates) | inference/component or sensitivity ablation using Full checkpoint | 3.448 | -0.211 | 6.273 | 77.341 | 100.000 | 0.000 | 37.463 | 0.059 | 37.110 | -0.271 | 0.543 | 26.051 | 100.000 | 60.099 | 16.639 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |

## Visual Anchor

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top-1 visual anchor | inference/component or sensitivity ablation using Full checkpoint | 4.348 | 0.690 | 7.657 | 66.214 | 99.796 | -0.204 | 37.811 | 0.407 | 36.771 | -0.611 | 3.053 | 26.045 | 98.982 | 53.542 | 18.677 |
| Posterior-weighted visual anchor | inference/component or sensitivity ablation using Full checkpoint | 3.884 | 0.226 | 6.875 | 73.338 | 99.864 | -0.136 | 37.560 | 0.156 | 36.906 | -0.475 | 1.085 | 25.469 | 99.050 | 54.172 | 18.460 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |

## Prior Jitter Sensitivity

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prior jitter 0 m | inference/component or sensitivity ablation using Full checkpoint | 3.050 | -0.609 | 5.882 | 83.039 | 100.000 | 0.000 | 37.513 | 0.109 | 37.517 | 0.136 | 0.136 | 23.796 | 99.932 | 52.056 | 19.210 |
| Prior jitter 4 m | inference/component or sensitivity ablation using Full checkpoint | 3.101 | -0.557 | 5.893 | 83.107 | 100.000 | 0.000 | 37.335 | -0.069 | 37.313 | -0.068 | 0.407 | 25.098 | 99.932 | 58.481 | 17.100 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |
| Prior jitter 12 m | inference/component or sensitivity ablation using Full checkpoint | 4.810 | 1.152 | 8.138 | 55.767 | 100.000 | 0.000 | 37.754 | 0.350 | 37.042 | -0.339 | 1.357 | 22.664 | 70.828 | 52.052 | 19.212 |
| Prior jitter 16 m | inference/component or sensitivity ablation using Full checkpoint | 6.493 | 2.835 | 11.032 | 36.771 | 98.575 | -1.425 | 38.330 | 0.926 | 36.499 | -0.882 | 2.849 | 31.851 | 36.364 | 53.045 | 18.852 |

## Final Ms Grid

| Label | AblationProtocol | MLE_m | DeltaMLE_m_vsFull | P90_m | LSR@5_pct | LSR@15_pct | DeltaLSR15_pp_vsFull | MHE_deg | DeltaMHE_deg_vsFull | HSR@15_pct | DeltaHSR15_pp_vsFull | JumpRate_pct | MaxFinalStep_m | SelectedCapture_pct | InferenceMean_ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Final MS grid 4x4 | inference/component or sensitivity ablation using Full checkpoint | 4.131 | 0.473 | 7.457 | 68.860 | 99.864 | -0.136 | 37.647 | 0.243 | 37.110 | -0.271 | 2.035 | 25.114 | 99.050 | 52.939 | 18.890 |
| Final MS grid 5x5 | inference/component or sensitivity ablation using Full checkpoint | 3.659 | 0.000 | 6.502 | 76.662 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 52.674 | 18.985 |
| SoftMS visual anchor (Full) | trained Full baseline | 3.658 | 0.000 | 6.502 | 76.866 | 100.000 | 0.000 | 37.404 | 0.000 | 37.381 | 0.000 | 0.746 | 24.657 | 99.050 | 47.553 | 21.029 |
| Final MS grid 7x7 | inference/component or sensitivity ablation using Full checkpoint | 3.636 | -0.022 | 6.434 | 77.544 | 100.000 | 0.000 | 37.384 | -0.020 | 37.381 | 0.000 | 0.543 | 24.647 | 99.050 | 53.811 | 18.583 |
| Final MS grid 8x8 | inference/component or sensitivity ablation using Full checkpoint | 3.637 | -0.022 | 6.435 | 77.544 | 100.000 | 0.000 | 37.384 | -0.020 | 37.381 | 0.000 | 0.543 | 24.646 | 99.050 | 58.062 | 17.223 |

