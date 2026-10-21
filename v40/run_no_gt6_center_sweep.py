"""Evaluate causal final 6x6 lattice centering without changing model weights."""
import json
import os
import sys
from pathlib import Path

from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
OUT = HERE / 'output/no_gt_6x6_20261008_center_sweep'


def main():
    base = json.loads((SOURCE / 'run_settings.json').read_text())['environment']
    checkpoint = json.loads((SOURCE / 'report.json').read_text())['checkpoint']
    cases = [('blend', '0'), ('kalman', '0')]
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, gpu in cases:
        dest = OUT / mode
        env = os.environ.copy()
        env.update(base)
        env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': '6',
                    'MS_BANDWIDTH_M': '18', 'V40_FINAL_MS_LATTICE_CENTER': mode,
                    'V40_FEATURE_CACHE_ROOT': str(dest / 'cache'),
                    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2'})
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', checkpoint, '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest)]
        label = 'center_%s GPU%s' % (mode, gpu)
        wait_all([(label, *launch(cmd, env, label, dest / 'console.log'))])
    rows = []
    for mode, gpu in cases:
        dest = OUT / mode
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'center': mode, 'MLE_m': report['MLE_m'],
                     'P90_m': report['P90_m'], 'report': str(dest / 'report.json')})
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        say('[RESULT] ' + json.dumps(row))


if __name__ == '__main__':
    main()
