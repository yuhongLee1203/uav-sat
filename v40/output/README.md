# v40 output index

Current no-current-frame-GT CityA validation best: `causal_4m_visual_seed2059_route_20260926_a/report.json` (4.402 m). Its visual weight is in `causal_4m_visual_seed2059_20260926_a/checkpoints/citya/`; the temporal weight comes from `causal_4m_search_20260926_a/temporal_grid/`. The GT-independence audit is in `causal_4m_best_gt_independence_audit_20260926_a/`.

A portable copy of best weights, original code, run settings, and experiment history is at `../checkpoint1/`. The CSV there lists all 139 causal report files before obsolete causal experiments were removed. The `official96_*`, `gt_*`, and `v39_v40_relaxed_*` directories are earlier GT or paper ablations; they were left intact to avoid losing previously requested research results.

The new one-MLP experiment writes to `fused_temporal_citya_grid_parallel_20260926/` after running `python3 v40/run_fused_temporal_citya.py` from the repository root.
