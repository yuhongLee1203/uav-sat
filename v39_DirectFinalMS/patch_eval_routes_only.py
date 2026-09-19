#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_eval_routes_only.py ROBUST_TRACKER.py")
p=Path(sys.argv[1])
s=p.read_text(encoding='utf-8')
old='''    for route_name in ["route_B", "route_C"]:\n'''
new='''    _route_env = __import__("os").environ.get("UAVSAT_EVAL_ROUTES", "route_B,route_C")\n    _eval_routes = [x.strip() for x in _route_env.split(",") if x.strip()]\n    for _name in _eval_routes:\n        if _name not in config.ROUTE_NAMES:\n            raise ValueError("Unknown UAVSAT_EVAL_ROUTES entry: %s" % _name)\n    all_summary["eval_routes"] = list(_eval_routes)\n    for route_name in _eval_routes:\n'''
if old not in s:
    if 'UAVSAT_EVAL_ROUTES' in s:
        print('[PATCH OK] selectable eval routes already present')
        raise SystemExit(0)
    raise SystemExit('eval-route loop not found')
s=s.replace(old,new,1)
if 'UAVSAT_GRU_FUSION_GAIN' in s or 'UAVSAT_GRU_CORRECTION_GAIN' in s or 'UAVSAT_GRU_MOTION_GAIN' in s:
    raise SystemExit('refusing tracker containing GRU inference gates')
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')
print('[PATCH OK] UAVSAT_EVAL_ROUTES added; no GRU gating added')
