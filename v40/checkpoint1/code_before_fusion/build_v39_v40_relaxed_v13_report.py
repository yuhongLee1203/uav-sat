#!/usr/bin/env python3
"""Report four-city v13 GT-assisted v39/v40-fix ablations with LSR@5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = ('no_gru', 'no_kalman', 'no_ms', 'full',
            'frames1', 'frames2', 'grid4', 'grid5', 'grid7', 'grid8')
LABELS = {'no_gru':'w/o GRU', 'no_kalman':'w/o Kalman',
          'no_ms':'w/o final Mean Shift', 'full':'Full',
          'frames1':'1 frame', 'frames2':'2 frames',
          'grid4':'4x4', 'grid5':'5x5', 'grid7':'7x7', 'grid8':'8x8'}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--suite-root',type=Path,required=True)
    args=p.parse_args()
    root=args.suite_root
    paper=json.loads((root/'paper_ablation_results.json').read_text())
    result={}
    for variant in VARIANTS:
        values=[]
        by_city={}
        for city in ('citya','cityb','cityc','cityd'):
            folder=root/city/'variants'/variant
            summaries=json.loads((folder/'bearing_v39_summary.json').read_text())
            city_values=[]
            for route in ('test_01','test_02'):
                csv_file=folder/Path(summaries[route]['CSV']).name
                with csv_file.open(newline='') as handle:
                    city_values.extend(float(r['error_final_m']) for r in csv.DictReader(handle))
            if not city_values:
                raise RuntimeError(f'No frames: {city}/{variant}')
            by_city[city]={'frames':len(city_values), 'MLE_m':sum(city_values)/len(city_values),
                           'LSR@5_pct':100*sum(x<=5.0 for x in city_values)/len(city_values)}
            values.extend(city_values)
        mle=sum(values)/len(values)
        reported=float(paper['rows'][variant]['MLE_m'])
        if len(values)!=1482 or abs(mle-reported)>1e-6:
            raise RuntimeError(f'Pooled alignment failed for {variant}: {len(values)} frames, {mle} vs {reported}')
        result[variant]={'frames':len(values), 'MLE_m':mle,
                         'LSR@5_pct':100*sum(x<=5.0 for x in values)/len(values),
                         'cities':by_city}
    payload={'dataset':'soft_sequence_v13_multicity_auto_train_fullroute, 4 m, 4 cities',
             'reference_protocol':'controlled_gt_jitter; current frame GT selects visual reference patch',
             'training':'fresh ABCD shared 1/2/3 frame checkpoints; v40 manual motion cap relaxation',
             'rows':result}
    (root/'v39_v40_relaxed_ablation.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Four-city v13 4 m GT-assisted ablation, fresh v39 training with v40 limit fixes','',
           'This is a new experiment. The archived 3.759 m is a historical comparator, not an expected result.','',
           '## Component removal','',
           '| Variant | Frames | MLE (m) ↓ | LSR@5 ↑ |',
           '|---|---:|---:|---:|']
    for variant in VARIANTS[:4]:
        r=result[variant]
        lines.append(f"| {LABELS[variant]} | {r['frames']} | {r['MLE_m']:.3f} | {r['LSR@5_pct']:.2f}% |")
    lines+=['','## Temporal input','', '| Input | MLE (m) ↓ | LSR@5 ↑ |', '|---|---:|---:|']
    for variant in ('frames1','frames2','full'):
        r=result[variant]
        lines.append(f"| {LABELS[variant]} | {r['MLE_m']:.3f} | {r['LSR@5_pct']:.2f}% |")
    lines+=['','## Final Mean Shift window','', '| Window | MLE (m) ↓ | LSR@5 ↑ |', '|---|---:|---:|']
    for variant in ('grid4','grid5','full','grid7','grid8'):
        r=result[variant]
        label='6x6' if variant=='full' else LABELS[variant]
        lines.append(f"| {label} | {r['MLE_m']:.3f} | {r['LSR@5_pct']:.2f}% |")
    lines+=['','Other pooled R@1*, LSR@15, HSR@15, MLE and MHE metrics are in `paper_ablation_tables.md`.',
            'R@1* is a same-quadrant proxy. These synthetic routes are not recorded flights.',
            'GT is a current-frame search input; these are GT-assisted results.','']
    (root/'v39_v40_relaxed_ablation.md').write_text('\n'.join(lines))
    print('[V39+V40 ABLATION REPORT]',root/'v39_v40_relaxed_ablation.md')


if __name__=='__main__':
    main()
