#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: patch_forward5x5_15.py <robust_tracker.py> <config.py>")

tracker_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
tracker = tracker_path.read_text(encoding="utf-8")
config = config_path.read_text(encoding="utf-8")

old_cfg = '''ACQ_LOCAL_GRID_SIZE = 6
'''
new_cfg = '''ACQ_LOCAL_GRID_SIZE = int(os.environ.get("UAVSAT_ACQ_LOCAL_GRID_SIZE", "5"))
'''
if config.count(old_cfg) != 1:
    raise SystemExit(f"ERROR: ACQ_LOCAL_GRID_SIZE block count={config.count(old_cfg)}")
config = config.replace(old_cfg, new_cfg, 1)

old_forward_cfg = '''FORWARD_SEARCH_ROWS = 3
FORWARD_SEARCH_COLS = 6
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
'''
new_forward_cfg = '''FORWARD_SEARCH_ROWS = int(os.environ.get("UAVSAT_FORWARD_SEARCH_ROWS", "3"))
FORWARD_SEARCH_COLS = int(os.environ.get("UAVSAT_FORWARD_SEARCH_COLS", "5"))
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
'''
if config.count(old_forward_cfg) != 1:
    raise SystemExit(f"ERROR: forward-search config block count={config.count(old_forward_cfg)}")
config = config.replace(old_forward_cfg, new_forward_cfg, 1)

old_guard = '''def forward_3x6_candidate_batch(visual, uav_clip, center_xy, heading_rad, grid_size=6):
    """Score only the causal-heading forward half of the original 6x6 grid.

    The full 6x6 geometry is used only to decide which 18 centers are forward.
    Satellite embeddings, cosine logits, posterior probabilities, Top-1, and
    SoftMS is computed only for the selected 18 candidates.  When heading is
    aligned with a gallery axis this is exactly the front 3 rows x 6 columns.
    """
    grid_size = int(grid_size)
    if grid_size != 6:
        raise ValueError("forward_3x6_candidate_batch expects base grid_size=6")
    keep_count = int(config.FORWARD_SEARCH_CANDIDATE_COUNT)
    if keep_count != 18:
        raise ValueError("forward search must keep exactly 3x6=18 candidates")
'''
new_guard = '''def forward_3x6_candidate_batch(visual, uav_clip, center_xy, heading_rad, grid_size=5):
    """Score the heading-forward 15 candidates from a 5x5 local geometry.

    The full 5x5 geometry is used only to rank locations by projection onto the
    causal heading. Exactly the top 3x5=15 candidates are encoded/scored; rear
    candidates are excluded from UAV-SAT similarity and front SoftMS.
    The legacy function name is retained for source/checkpoint compatibility.
    """
    grid_size = int(grid_size)
    expected_grid = int(config.ACQ_LOCAL_GRID_SIZE)
    if grid_size != expected_grid:
        raise ValueError(
            f"forward local search expects base grid_size={expected_grid}, got {grid_size}"
        )
    keep_count = int(config.FORWARD_SEARCH_CANDIDATE_COUNT)
    expected_keep = int(config.FORWARD_SEARCH_ROWS) * int(config.FORWARD_SEARCH_COLS)
    if keep_count != expected_keep or keep_count != 15:
        raise ValueError(
            f"forward search must keep exactly 3x5=15 candidates, got {keep_count}"
        )
    if grid_size != 5:
        raise ValueError(f"selected V39 method requires 5x5 geometry, got {grid_size}x{grid_size}")
'''
if tracker.count(old_guard) != 1:
    raise SystemExit(f"ERROR: legacy forward-3x6 guard count={tracker.count(old_guard)}")
tracker = tracker.replace(old_guard, new_guard, 1)

tracker = tracker.replace('summary["VisualObservationDecoder"] = "forward 3x6 soft mean shift"','summary["VisualObservationDecoder"] = "forward 15-of-5x5 soft mean shift"')
tracker = tracker.replace('Forward-3x6 Soft MeanShift visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final 6x6 MeanShift -> final XY.','Forward 15-of-5x5 Soft MeanShift visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final 5x5 MeanShift -> final XY.')
tracker = tracker.replace('Forward 3x6 SoftMS -> GRU -> Kalman -> ONE final 6x6 MS -> Final','Forward 15-of-5x5 SoftMS -> GRU -> Kalman -> ONE final 5x5 MS -> Final')
tracker = tracker.replace('Forward-3x6 decoder','Forward-15-of-5x5 decoder')
tracker = tracker.replace('Forward 3x6 candidates','Forward 15-of-5x5 candidates')
tracker = tracker.replace('Confidence from the actual 6x6 local posterior, never hypothesis count.','Confidence from the actual forward-15 local posterior, never hypothesis count.')
tracker = tracker.replace('compare the best raw\n    # UAV-SAT similarity of every 3x6 window.','compare the best raw\n    # UAV-SAT similarity of every forward-15 local window.')

if 'forward 3x6 soft mean shift' in tracker:
    raise SystemExit("ERROR: old forward 3x6 decoder summary still remains")
if 'forward search must keep exactly 3x6=18 candidates' in tracker:
    raise SystemExit("ERROR: old 18-candidate guard still remains")

compile(tracker, str(tracker_path), "exec")
compile(config, str(config_path), "exec")
tracker_path.write_text(tracker, encoding="utf-8")
config_path.write_text(config, encoding="utf-8")
print("[OK] V39 NX online search: 5x5 geometry -> heading-forward 15 -> SoftMS")
