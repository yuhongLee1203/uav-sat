# FieldAnchorFINAL figure audit

Audit target: `masterpaper/FieldAnchorFINAL.pdf` (61 PDF pages).

| Figures | PDF status | Action |
|---|---|---|
| 1--4 | Embedded data/dataset figures | Already present; no placeholder. |
| 5--9 | Embedded method diagrams | Already present; no placeholder. |
| 10 | Embedded final-trajectory image | Already present. A clearer real-output replacement with chronological start/end markers is also exported as `fig10_full_trajectory.{png,pdf}`. |
| 11--20 | Embedded experimental charts | Already present; no placeholder. |
| 21 | Reserved placeholder on PDF page 55 | Replaced by `fig21_per_frame_localization_error.{png,pdf}`. |
| 22 | Reserved placeholder on PDF page 56 | Replaced by the non-redundant MS-vs-final distribution plot `fig22_error_ecdf.{png,pdf}`. |
| 23 | Reserved placeholder on PDF page 56 | Replaced by `fig23_visual_response_cases.{png,pdf}`. |
| 24 | Reserved placeholder on PDF page 57 | Replaced by `fig24_turn_closeups.{png,pdf}`. |

The audit searched the rendered PDF and extracted text for `Reserved`,
`replace with`, `placeholder`, `to be inserted`, and `TBD`. The only four
result-figure placeholders are Figures 21--24. The PDF has not itself been
rewritten, because this directory is the requested hand-off location for the
replacement figures.
