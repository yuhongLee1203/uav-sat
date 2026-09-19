#!/usr/bin/env python3
from pathlib import Path
import sys
if len(sys.argv)!=2:
    raise SystemExit('usage: patch_routea_localization_objective.py ROBUST_TRACKER.py')
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
old='''    mle = float(np.mean(errors))
    speed_mae = float(np.mean(speed_errors))
    progress_mae = float(np.mean(progress_errors))
    heading_mae_deg = float(np.mean(heading_errors)) if heading_errors else float("inf")
    capture_pct = float(np.mean(captures) * 100.0)
    bank_capture_pct = float(np.mean(bank_captures) * 100.0)
    score = (
        mle
        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed_mae
        + float(config.EARLY_SCORE_PROGRESS_WEIGHT) * progress_mae
        + float(config.EARLY_SCORE_HEADING_WEIGHT) * heading_mae_deg
        + float(config.EARLY_SCORE_MISS_WEIGHT) * (100.0 - capture_pct)
    )
'''
new='''    mle = float(np.mean(errors))
    p90 = float(np.quantile(errors, 0.90))
    speed_mae = float(np.mean(speed_errors))
    progress_mae = float(np.mean(progress_errors))
    heading_mae_deg = float(np.mean(heading_errors)) if heading_errors else float("inf")
    capture_pct = float(np.mean(captures) * 100.0)
    bank_capture_pct = float(np.mean(bank_captures) * 100.0)
    # Route-A-only checkpoint selection: prioritize localization itself.
    # B/C never participates in this score. P90 discourages a checkpoint that
    # improves the mean by introducing a long error tail.
    p90_weight = float(__import__("os").environ.get("UAVSAT_ROUTEA_P90_WEIGHT", "0.20"))
    score = (
        mle
        + p90_weight * p90
        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed_mae
        + float(config.EARLY_SCORE_PROGRESS_WEIGHT) * progress_mae
        + float(config.EARLY_SCORE_HEADING_WEIGHT) * heading_mae_deg
        + float(config.EARLY_SCORE_MISS_WEIGHT) * (100.0 - capture_pct)
    )
'''
if s.count(old)!=1:
    raise SystemExit(f'localization objective patch count={s.count(old)}')
s=s.replace(old,new,1)
s=s.replace('''        "p90": float(np.quantile(errors, 0.90)),''','''        "p90": p90,''',1)
compile(s,str(p),'exec'); p.write_text(s,encoding='utf-8')
print('[PATCH OK] Route-A checkpoint score includes localization MLE + P90; B/C untouched')
