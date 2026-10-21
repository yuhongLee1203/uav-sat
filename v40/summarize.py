import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads((root / name / "result.json").read_text())
        for name in ("visual_pseudo", "temporal_cycle", "confidence_gated")]
lines = ["# City A image-only adaptation (supervised warm start)", "",
         "Training uses no frame GT; test GT is used only for metrics.", "",
         "| Design | Train loss | test_01 MLE | test_02 MLE |", "|---|---:|---:|---:|"]
for r in rows:
    def mle(name):
        summary = r.get(name, {})
        for key in ("MLE_m", "mle_m", "mean_error_m", "mle"):
            if key in summary:
                return "%.3f" % summary[key]
        return "see result.json"
    lines.append("| %s | %.4f | %s | %s |" % (r["design"], r["image_only_loss"][-1],
                                                 mle("route_B"), mle("route_C")))
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
