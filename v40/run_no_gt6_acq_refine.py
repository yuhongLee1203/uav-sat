"""Warm-start causal acquisition scorer from best 6x6 temporal checkpoint."""
import json
import os
import sys
from pathlib import Path
from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
BASE_DIR = HERE / 'output/no_gt_6x6_20260927_a'
OUT = HERE / 'output/no_gt_6x6_20261008_acq_refine'
TRAIN_GRID = HERE / 'data/official96_fixed_train_grid'


def environment(base, dest, train):
    env = os.environ.copy()
    env.update(base)
    env.update({'CUDA_VISIBLE_DEVICES': '0', 'MS_GRID_SIZE': '6',
                'MS_BANDWIDTH_M': '18',
                'V40_ACQ_TRAIN_LABEL_MODE': 'visual_anchor',
                'V40_FEATURE_CACHE_ROOT': str(dest / ('train_cache' if train else 'val_cache')),
                'V40_GT_FRONT18_ORACLE': '0', 'V40_GT_FRONT18_PATCH_ORACLE': '0',
                'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
                'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
                'V40_GT_JITTER_EVAL_M': '0', 'V40_GT_JITTER_PATCH_SELECT': '0',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2'})
    return env


def main():
    base = json.loads((SOURCE / 'run_settings.json').read_text())['environment']
    checkpoint = json.loads((SOURCE / 'report.json').read_text())['checkpoint']
    audit = json.loads((TRAIN_GRID / 'citya/audit.json').read_text())
    assert audit['no_cross_split_samples'] and audit['no_duplicate_extra_rows']
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'train/best_temporal_interim.pt').is_file():
        cmd = [sys.executable, '-u', str(HERE / 'train_multicity_routes.py'),
               '--train-cities', 'citya', '--eval-cities', 'citya',
               '--supervision', 'gt_train', '--init', 'random_temporal',
               '--epochs', '4', '--max-train-frames', '136', '--lr', '5e-6',
               '--dense-offbank-loss', '--acquisition-loss-weight', '4',
               '--acquisition-target-sigma-m', '4',
               '--extra-train-routes-dir', str(TRAIN_GRID),
               '--validate-every-epochs', '1', '--early-stop-patience', '3',
               '--skip-test', '--seed', '2033',
               '--init-temporal-checkpoint', checkpoint,
               '--output', str(OUT / 'train')]
        label = 'acq_refine GPU0 TRAIN'
        wait_all([(label, *launch(cmd, environment(base, OUT, True),
                                 label, OUT / 'train.log', 4 * 14))])
    ck = OUT / 'train/best_temporal_interim.pt'
    cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
           '--checkpoint', str(ck), '--city', 'citya',
           '--tta-rotations', '0,1,2,3', '--output', str(OUT / 'validation')]
    label = 'acq_refine GPU0 VAL'
    wait_all([(label, *launch(cmd, environment(base, OUT, False),
                             label, OUT / 'validation.log'))])
    report = json.loads((OUT / 'validation/report.json').read_text())
    settings = json.loads((OUT / 'validation/run_settings.json').read_text())
    assert report['valid_no_GT_inference_result'] and not report['current_frame_GT_model_input']
    assert settings['environment']['MS_GRID_SIZE'] == '6'
    result = {'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
              'SelectedCandidateCapture_pct': report['SelectedCandidateCapture_pct'],
              'checkpoint': str(ck), 'report': str(OUT / 'validation/report.json')}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    say('[RESULT] ' + json.dumps(result))


if __name__ == '__main__':
    main()
