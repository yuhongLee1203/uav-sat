#!/usr/bin/env python3
"""Report exactly replayed four-city v13 GT-assisted localization ablations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = ('no_gru', 'no_kalman', 'no_ms', 'full', 'frames1', 'frames2',
            'grid4', 'grid5', 'grid7', 'grid8')
CITIES = ('citya', 'cityb', 'cityc', 'cityd')
LABELS = {'no_gru': 'w/o GRU', 'no_kalman': 'w/o Kalman',
          'no_ms': 'w/o final Mean Shift', 'full': 'Full',
          'frames1': '1 frame', 'frames2': '2 frames',
          'grid4': '4x4', 'grid5': '5x5', 'grid7': '7x7', 'grid8': '8x8'}
EXPECTED = {
    'no_gru': (4.275, 65.92), 'no_kalman': (3.763, 73.68),
    'no_ms': (6.785, 33.40), 'full': (3.759, 73.75),
    'frames1': (3.883, 72.47), 'frames2': (3.921, 71.79),
    'grid4': (4.168, 67.14), 'grid5': (3.763, 73.75),
    'grid7': (3.738, 73.95), 'grid8': (3.738, 74.09),
}
EXPECTED_LSR3 = {'frames1': 39.81, 'frames2': 39.14, 'full': 41.90}


def actual_errors(root: Path, variant: str) -> list[float]:
    errors = []
    for city in CITIES:
        folder = root / city / 'variants' / variant
        summaries = json.loads((folder / 'bearing_v39_summary.json').read_text())
        for route in ('test_01', 'test_02'):
            path = folder / Path(summaries[route]['CSV']).name
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise RuntimeError(f'Empty inference output: {path}')
            errors.extend(float(row['error_final_m']) for row in rows)
    return errors


def table_row(label: str, row: dict, *, localization_only: bool = False) -> str:
    fields = [label, f"{row['R@1*_pct']:.2f}%", f"{row['LSR@5_pct']:.2f}%",
              f"{row['LSR@15_pct']:.2f}%", f"{row['HSR@15_pct']:.2f}%",
              f"{row['MLE_m']:.3f}", f"{row['MHE_deg']:.2f}"]
    if not localization_only:
        fields += ['N/A', 'N/A', 'N/A']
    return '| ' + ' | '.join(fields) + ' |'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--no-baseline-gate', action='store_true',
                        help='Report measured values without requiring paper-value reproduction.')
    args = parser.parse_args()
    root = args.suite_root.resolve()
    paper = json.loads((root / 'paper_ablation_results.json').read_text())
    if tuple(paper['cities']) != CITIES:
        raise RuntimeError('This paper replay requires City A/B/C/D, in that order')
    results = {}
    failures = []
    exact_mismatches = []
    for variant in VARIANTS:
        errors = actual_errors(root, variant)
        n = len(errors)
        if n != 1482:
            raise RuntimeError(f'{variant}: expected 1482 frames, found {n}')
        base = paper['rows'][variant]
        mle = sum(errors) / n
        if abs(mle - float(base['MLE_m'])) > 1e-6:
            raise RuntimeError(f'{variant}: frame CSV MLE {mle} disagrees with paper table {base["MLE_m"]}')
        lsr3 = 100.0 * sum(e <= 3.0 for e in errors) / n
        lsr5 = 100.0 * sum(e <= 5.0 for e in errors) / n
        expected_mle, expected_lsr5 = EXPECTED[variant]
        if round(mle, 3) != expected_mle or round(lsr5, 2) != expected_lsr5:
            exact_mismatches.append(f"{variant}: MLE {mle:.3f}/{expected_mle:.3f}; LSR@5 {lsr5:.2f}%/{expected_lsr5:.2f}%")
        if abs(mle - expected_mle) > 0.05 or abs(lsr5 - expected_lsr5) > 0.15:
            failures.append(f'{variant}: MLE {mle:.3f} vs {expected_mle:.3f}; '
                            f'LSR@5 {lsr5:.2f}% vs {expected_lsr5:.2f}%')
        if variant in EXPECTED_LSR3 and abs(lsr3 - EXPECTED_LSR3[variant]) > 0.15:
            failures.append(f'{variant}: LSR@3 {lsr3:.2f}% vs {EXPECTED_LSR3[variant]:.2f}%')
        results[variant] = {**base, 'LSR@3_pct': lsr3, 'LSR@5_pct': lsr5}
    status = 'PASS' if not failures else 'FAIL'
    payload = {
        'source_protocol': 'soft_sequence_v13_multicity_auto_train_fullroute; controlled_gt_jitter; 4 m',
        'cities': CITIES, 'test_frames': 1482,
        'historical_paper_value_check': {'status': status, 'differences': failures, 'exact_printed_match': not exact_mismatches, 'exact_mismatches': exact_mismatches},
        'Recall@1': None, 'Recall@1_reason': 'No official four-tile retrieval ranking in this inference output; R@1* is a same-quadrant XY proxy.',
        'SR@20': None, 'SPL': None, 'NE': None,
        'navigation_reason': 'Independent geotagged images reassembled into synthetic routes; no recorded closed-loop flight trajectory or actions.',
        'rows': results,
    }
    (root / 'original_3759_all_metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Original 3.759 m four-city inference replay', '',
             f'Historical MLE/LSR@5 proximity gate: **{status}**; exact printed-value match: **{"YES" if not exact_mismatches else "NO"}**.', '',
             'Dataset: original v13 4 m City A/B/C/D synthetic routes; 1,482 held-out images.',
             'Inference uses current-frame GT for the search reference and acquisition teacher selection; checkpoints are unchanged.', '',
             '## Main architecture: GT-assisted localization', '',
             '| Method | R@1* ↑ | LSR@5 ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ | SR@20 ↑ | SPL ↑ | NE ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
             table_row('Full', results['full']), '',
             '## Component ablation', '',
             '| Variant | R@1* ↑ | LSR@5 ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|']
    lines += [table_row(LABELS[v], results[v], localization_only=True)
              for v in ('no_gru', 'no_kalman', 'no_ms', 'full')]
    lines += ['', '## Temporal context', '',
              '| Input | R@1* ↑ | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant in ('frames1', 'frames2', 'full'):
        row = results[variant]
        lines.append(f"| {LABELS[variant]} | {row['R@1*_pct']:.2f}% | {row['LSR@3_pct']:.2f}% | "
                     f"{row['LSR@5_pct']:.2f}% | {row['LSR@15_pct']:.2f}% | "
                     f"{row['HSR@15_pct']:.2f}% | {row['MLE_m']:.3f} | {row['MHE_deg']:.2f} |")
    lines += ['', '## Final MeanShift window', '',
              '| Lattice | R@1* ↑ | LSR@5 ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE (m) ↓ | MHE (deg) ↓ | MS time (ms) ↓ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in ('grid4', 'grid5', 'full', 'grid7', 'grid8'):
        row = results[variant]
        lattice = '6x6' if variant == 'full' else LABELS[variant]
        lines.append(f"| {lattice} | {row['R@1*_pct']:.2f}% | {row['LSR@5_pct']:.2f}% | "
                     f"{row['LSR@15_pct']:.2f}% | {row['HSR@15_pct']:.2f}% | "
                     f"{row['MLE_m']:.3f} | {row['MHE_deg']:.2f} | {row['MS_Latency_ms']:.3f} |")
    lines += ['', 'R@1* is a same-quadrant XY proxy, not official retrieval Recall@1.',
              'SR@20/SPL/NE are N/A: the data have no recorded closed-loop navigation rollout.',
              'MeanShift time depends on hardware/load and need not match the paper millisecond value.', '']
    if exact_mismatches:
        lines += ['## Exact printed-value mismatches', '', *('- ' + d for d in exact_mismatches), '']
    if failures:
        lines += ['## Proximity-gate failures', '', *('- ' + d for d in failures), '']
    (root / 'original_3759_all_metrics.md').write_text('\n'.join(lines))
    print('[ORIGINAL 3759 INFERENCE METRICS]', root / 'original_3759_all_metrics.md')
    if failures and not args.no_baseline_gate:
        raise SystemExit('Historical paper numbers did not reproduce; inspect original_3759_all_metrics.md')


if __name__ == '__main__':
    main()
