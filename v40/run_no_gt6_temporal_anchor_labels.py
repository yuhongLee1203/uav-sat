"""Train three no-GT CityA temporal variants on GPUs 0/5/6; final MS is 6x6."""
import json
import os
import subprocess
import sys
from pathlib import Path

from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
BEST = HERE / 'checkpoint1/best_no_gt_20260926'
OUT = HERE / 'output/no_gt_6x6_20260927_a/temporal_anchor_labels'
EXTRA = HERE / 'data/official96_fixed_train_grid'
VISUAL_ROOT = HERE / 'output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints'
VISUAL_FILE = VISUAL_ROOT / 'citya/visual_retrieval_A_only.pt'
BASE = json.loads((BEST / 'run_settings_original.json').read_text())['environment']
CASES = [
    ('anchor_sigma4_acq1', '0', '1', '4', 2033, '10'),
    ('anchor_sigma8_acq1', '5', '1', '8', 2033, '10'),
    ('anchor_sigma4_acq4', '6', '4', '4', 2033, '10'),
]


def environment(gpu, dest, train, raw_weight):
    env = os.environ.copy()
    env.update(BASE)
    env.update({
        'CUDA_VISIBLE_DEVICES': gpu,
        'MS_GRID_SIZE': '6',
        'V40_TEMPORAL_FUSION_MLP': '1',
        'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS': '1',
        'V40_ACQ_RAW_VISUAL_WEIGHT': raw_weight,
        'V40_ACQ_LEARNED_LOGIT_SCALE': '16',
        'V40_ACQ_TRAIN_LABEL_MODE': 'visual_anchor',
        'V40_VISUAL_CHECKPOINT_ROOT': str(VISUAL_ROOT),
        'V40_VISUAL_CHECKPOINT_FILE': str(VISUAL_FILE),
        'V40_FEATURE_CACHE_ROOT': str(dest / ('train_cache' if train else 'val_cache')),
        'V40_GT_FRONT18_ORACLE': '0',
        'V40_GT_FRONT18_PATCH_ORACLE': '0',
        'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
        'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
        'V40_GT_JITTER_EVAL_M': '0',
        'V40_GT_JITTER_PATCH_SELECT': '0',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    })
    return env


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, acq, sigma, seed, raw_weight in CASES:
        dest = OUT / name
        cmd = [sys.executable, '-u', str(HERE / 'train_multicity_routes.py'),
               '--train-cities', 'citya', '--eval-cities', 'citya',
               '--supervision', 'gt_train', '--init', 'random_temporal',
               '--epochs', '4', '--max-train-frames', '136', '--lr', '1e-5',
               '--dense-offbank-loss', '--acquisition-loss-weight', acq,
               '--acquisition-target-sigma-m', sigma,
               '--extra-train-routes-dir', str(EXTRA),
               '--validate-every-epochs', '1', '--early-stop-patience', '3',
               '--skip-test', '--seed', str(seed), '--output', str(dest / 'train')]
        label = name + ' GPU' + gpu
        proc, thread, log = launch(cmd, environment(gpu, dest, True, raw_weight), label,
                                   dest / 'train.log', 4 * 14)
        jobs.append((label, proc, thread, log))
    wait_all(jobs)
    jobs = []
    for name, gpu, _, _, _, raw_weight in CASES:
        dest = OUT / name
        ckpt = dest / 'train/best_temporal_interim.pt'
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', str(ckpt), '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest / 'validation')]
        label = name + ' GPU' + gpu + ' TTA val'
        proc, thread, log = launch(cmd, environment(gpu, dest, False, raw_weight), label,
                                   dest / 'validation.log')
        jobs.append((label, proc, thread, log))
    wait_all(jobs)
    rows = []
    for name, gpu, acq, sigma, seed, raw_weight in CASES:
        dest = OUT / name
        report = json.loads((dest / 'validation/report.json').read_text())
        settings = json.loads((dest / 'validation/run_settings.json').read_text())
        assert report['valid_no_GT_inference_result'] and not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'name': name, 'gpu': gpu, 'acquisition_loss_weight': float(acq),
                     'sigma_m': float(sigma), 'seed': seed, 'raw_visual_weight': float(raw_weight),
                     'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
                     'LSR@15_pct': report['LSR@15_pct'],
                     'checkpoint': str(dest / 'train/best_temporal_interim.pt'),
                     'report': str(dest / 'validation/report.json')})
        say('[RESULT] %s GPU%s MLE %.3f m' % (name, gpu, report['MLE_m']))
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    say('[DONE] ' + str(OUT / 'summary.json'))


if __name__ == '__main__':
    main()
