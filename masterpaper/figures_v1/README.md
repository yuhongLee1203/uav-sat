# FieldAnchorFINAL exported result figures

These files fill the four reserved positions in `FieldAnchorFINAL.pdf`; an improved
optional replacement for the existing Figure 10 is also included.

| PDF figure | Export |
|---|---|
| Figure 21 | `fig21_per_frame_localization_error` |
| Figure 22 | `fig22_error_ecdf` |
| Figure 23 | `fig23_visual_response_cases` |
| Figure 24 | `fig24_turn_closeups` |
| Optional Figure 10 replacement | `fig10_full_trajectory` |

All four figures use the final versioned run at
`outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman`.
Figures 21, 22 and 24 are drawn from its Route-B/C per-frame CSV outputs.
Figure 21 compares the per-frame MS visual estimate against the final constrained
Kalman estimate. Figure 22 compares their complete held-out error distributions;
it is not the redundant Route-B/C split curve previously proposed for this slot.
Figure 23 also re-evaluates the selected frames with the actual frozen visual
checkpoint `checkpoints/visual_retrieval_A_only.pt` on GPU 0, then overlays the
resulting forward-3×6 candidate probabilities on the original satellite
orthomosaic. The paired UAV images are the original dataset frames recorded in
those outputs.

Regenerate with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 /yh/study/uav-sat/masterpaper/generate_final_experiment_figures.py
```
