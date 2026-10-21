"""Second acquisition-only refinement from the previous train-only checkpoint."""
import json
import os
import sys
from pathlib import Path
from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
SOURCE_SETTINGS = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
SOURCE_TEMPORAL = HERE / 'output/no_gt_6x6_20261008_acq_refine/train/best_temporal_interim.pt'
TRAIN_GRID = HERE / 'data/official96_fixed_train_grid'
OUT = HERE / 'output/no_gt_6x6_20261008_acq_refine_stage2'
CASES = [('acq8_sigma4', '0', '8', '4'),
         ('acq8_sigma2', '5', '8', '2'),
         ('acq16_sigma4', '6', '16', '4')]


def environment(base, gpu, dest, train):
    env = os.environ.copy()
    env.update(base)
    env.update({'CUDA_VISIBLE_DEVICES': gpu,
                'MS_GRID_SIZE': '6', 'MS_BANDWIDTH_M': '18',
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
    base = json.loads((SOURCE_SETTINGS / 'run_settings.json').read_text())['environment']
    audit = json.loads((TRAIN_GRID / 'citya/audit.json').read_text())
    assert audit['no_cross_split_samples'] and audit['no_duplicate_extra_rows']
    if not SOURCE_TEMPORAL.is_file():
        raise FileNotFoundError(SOURCE_TEMPORAL)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, weight, sigma in CASES:
        dest = OUT / name
        cmd = [sys.executable, '-u', str(HERE / 'train_multicity_routes.py'),
               '--train-cities', 'citya', '--eval-cities', 'citya',
               '--supervision', 'gt_train', '--init', 'random_temporal',
               '--epochs', '3', '--max-train-frames', '136', '--lr', '2e-6',
               '--dense-offbank-loss', '--acquisition-loss-weight', weight,
               '--acquisition-target-sigma-m', sigma,
               '--extra-train-routes-dir', str(TRAIN_GRID),
               '--validate-every-epochs', '1', '--early-stop-patience', '3',
               '--skip-test', '--seed', '2033',
               '--init-temporal-checkpoint', str(SOURCE_TEMPORAL),
               '--output', str(dest / 'train')]
        label = '%s GPU%s TRAIN' % (name, gpu)
        jobs.append((label, *launch(cmd, environment(base, gpu, dest, True),
                                   label, dest / 'train.log', 3 * 14)))
    wait_all(jobs)
    jobs = []
    for name, gpu, _, _ in CASES:
        dest = OUT / name
        ck = dest / 'train/best_temporal_interim.pt'
        if not ck.is_file():
            raise FileNotFoundError(ck)
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', str(ck), '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest / 'validation')]
        label = '%s GPU%s VAL' % (name, gpu)
        jobs.append((label, *launch(cmd, environment(base, gpu, dest, False),
                                   label, dest / 'validation.log')))
    wait_all(jobs)
    rows = []
    for name, gpu, weight, sigma in CASES:
        dest = OUT / name / 'validation'
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'name': name, 'gpu': gpu, 'acquisition_loss_weight': float(weight),
                     'sigma_m': float(sigma), 'MLE_m': report['MLE_m'],
                     'P90_m': report['P90_m'],
                     'SelectedCandidateCapture_pct': report['SelectedCandidateCapture_pct'],
                     'report': str(dest / 'report.json')})
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        say('[RESULT] ' + json.dumps(row))


if __name__ == '__main__':
    main()
