import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = ["# Supervised-training / GT-free-test sweep", "",
         "All variants use one shared A+B temporal model. Test GT is metrics-only.", "",
         "| Trial | City | Test route | MLE (m) | P90 (m) | LSR@15 |",
         "|---|---|---|---:|---:|---:|"]
for name in ("causal", "mixed", "teacher"):
    report = json.loads((root / name / "report.json").read_text())
    for city, routes in report["results"].items():
        for route, result in routes.items():
            lines.append("| %s | %s | %s | %.3f | %.3f | %.2f%% |" % (
                name, city, route, result["MLE_m"], result["P90_m"],
                result["LSR@15_pct"]))
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
