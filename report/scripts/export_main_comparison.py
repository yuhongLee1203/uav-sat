"""Export the seven-row comparison as one paper-ready Markdown file."""

import csv
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
SOURCE = REPORT / "tables" / "main_comparison.csv"
OUTPUT = REPORT / "MAIN_COMPARISON.md"
EXPECTED = [
    "University-1652", "SUES-200", "DenseUAV", "GTA-UAV",
    "Bearing-UAV", "Bearing-UAV + WA", "Our Method",
]


def main():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["Method"] for row in rows] != EXPECTED:
        raise ValueError("Unexpected methods or ordering in main_comparison.csv")
    if not all(row["source"] for row in rows):
        raise ValueError("A comparison row is missing its source")

    lines = [
        "# Main Comparison Table", "",
        "| Method | LSR@15 (%) ↑ | MLE (m) ↓ |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        method = "FieldAnchor" if row["Method"] == "Our Method" else row["Method"]
        lines.append(
            f"| {method} | {float(row['LSR@15_pct']):.2f} | "
            f"{float(row['MLE_m']):.2f} |"
        )
    lines.extend([
        "", "## Source and protocol", "",
        "- University-1652 through Bearing-UAV + WA: published UAV-view values "
        "from [Bearing-UAV Table 2](https://arxiv.org/abs/2603.22153); "
        "these rows were not rerun here.",
        "- FieldAnchor: [original v13 inference metrics]"
        "(../v39_otherdata/output/original3759_inference_20260925_220434_1615534/original_3759_all_metrics.json), "
        "1,482 held-out frames, MLE 3.7591277 m rounded to 3.76 m.",
        "- FieldAnchor uses a GT-assisted local/path reference during acquisition. "
        "The published rows and FieldAnchor do not share an identical input protocol; "
        "this table is a reported-values comparison, not a same-protocol benchmark.",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
