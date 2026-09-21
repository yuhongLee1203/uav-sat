#!/usr/bin/env python3
"""Let the plotter read legacy nav50/nav51 summary keys as test_01/test_02."""
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = '    for route in args.routes:\n        audit[route] = render(route, root, out, summaries[route])'
new = '''    legacy_alias = {"test_01": "nav50", "test_02": "nav51"}
    for route in args.routes:
        summary_key = route if route in summaries else legacy_alias.get(route)
        if summary_key not in summaries:
            raise KeyError(f"missing summary for {route}; keys={sorted(summaries)}")
        audit[route] = render(route, root, out, summaries[summary_key])'''
if new not in s:
    if s.count(old) != 1:
        raise SystemExit(f"plot alias patch target count={s.count(old)}")
    s = s.replace(old, new, 1)
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[PLOT ROUTE ALIAS PATCH OK]', p)
