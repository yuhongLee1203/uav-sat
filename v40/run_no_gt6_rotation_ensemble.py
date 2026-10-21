"""Diagnostic: four causal 6x6 trackers, image rotation ensemble, no GT fusion."""
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
OUT = HERE / 'output/no_gt_6x6_20261008_rotation_ensemble'
CASES = [('rot0', '0', '0'), ('rot1', '5', '1'), ('rot2', '6', '2'), ('rot3', '0', '3')]


def launch_case(base, checkpoint, name, gpu, turns):
    dest = OUT / name
    env = os.environ.copy()
    env.update(base)
    env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': '6',
                'MS_BANDWIDTH_M': '18',
                'V40_FEATURE_CACHE_ROOT': str(dest / 'cache'),
                'V40_GT_FRONT18_ORACLE': '0', 'V40_GT_FRONT18_PATCH_ORACLE': '0',
                'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
                'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
                'V40_GT_JITTER_EVAL_M': '0', 'V40_GT_JITTER_PATCH_SELECT': '0',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2'})
    cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
           '--checkpoint', checkpoint, '--city', 'citya',
           '--tta-rotations', turns, '--output', str(dest)]
    label = '%s GPU%s' % (name, gpu)
    return (label, *launch(cmd, env, label, dest / 'console.log'))


def main():
    base = json.loads((SOURCE / 'run_settings.json').read_text())['environment']
    checkpoint = json.loads((SOURCE / 'report.json').read_text())['checkpoint']
    OUT.mkdir(parents=True, exist_ok=True)
    wait_all([launch_case(base, checkpoint, *row) for row in CASES[:3]])
    wait_all([launch_case(base, checkpoint, *CASES[3])])
    routes = []
    reports = []
    for name, gpu, turns in CASES:
        dest = OUT / name
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        files = list((dest / 'citya').glob('*.csv'))
        if len(files) != 1:
            raise RuntimeError('expected one validation route CSV: ' + str(files))
        with files[0].open(newline='') as stream:
            routes.append(list(csv.DictReader(stream)))
        reports.append({'rotation': int(turns), 'MLE_m': report['MLE_m']})
    if len(set(map(len, routes))) != 1:
        raise RuntimeError('frame count mismatch')
    image_paths = [[row['image_path'] for row in route] for route in routes]
    if any(paths != image_paths[0] for paths in image_paths[1:]):
        raise RuntimeError('frame alignment mismatch')
    gt = np.asarray([[float(row['gt_x']), float(row['gt_y'])]
                     for row in routes[0]], dtype=np.float64)
    predictions = np.asarray([[[float(row['final_x']), float(row['final_y'])]
                               for row in route] for route in routes], dtype=np.float64)
    summary = {'individual': reports, 'ensemble': {}}
    for name, xy in [('coordinate_median', np.median(predictions, axis=0)),
                     ('coordinate_mean', np.mean(predictions, axis=0))]:
        error = np.linalg.norm(xy - gt, axis=1)
        summary['ensemble'][name] = {'MLE_m': float(error.mean()),
                                      'P90_m': float(np.percentile(error, 90)),
                                      'LSR_at_15_pct': float(np.mean(error <= 15) * 100),
                                      'n': len(error)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    say('[RESULT] ' + json.dumps(summary))


if __name__ == '__main__':
    main()
