"""Train and evaluate hard-negative visual candidate weights with fixed temporal/6x6."""
import json
import os
import sys
from pathlib import Path

from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / 'output/no_gt_6x6_20260927_a'
BASE_REPORT = BASE_DIR / 'cross_prior_sweep/cross_prior_0.00'
TEMPORAL = BASE_DIR / 'temporal_visual2061/raw10_sigma4/train/best_temporal_interim.pt'
SOURCE = BASE_DIR / 'visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'
OUT = HERE / 'output/no_gt_6x6_20261008_visual_candidates_retry16'
CASES = [
    ('hard64', '5', '64', '16', '2067'),
    ('hard64_random32', '6', '64', '32', '2073'),
]


def environment(base, gpu, dest, visual=None):
    env = os.environ.copy()
    env.update(base)
    env.update({
        'CUDA_VISIBLE_DEVICES': gpu,
        'V40_FEATURE_CACHE_ROOT': str(dest / 'cache'),
        'MS_GRID_SIZE': '6', 'MS_BANDWIDTH_M': '18',
        'V40_GT_FRONT18_ORACLE': '0', 'V40_GT_FRONT18_PATCH_ORACLE': '0',
        'V40_GT_FINAL_MS_ORACLE_CENTER': '0', 'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
        'V40_GT_JITTER_EVAL_M': '0', 'V40_GT_JITTER_PATCH_SELECT': '0',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    })
    if visual:
        env['V40_VISUAL_CHECKPOINT_FILE'] = str(visual)
    return env


def main():
    base = json.loads((BASE_REPORT / 'run_settings.json').read_text())['environment']
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, topk, random_count, seed in CASES:
        dest = OUT / name
        cmd = [sys.executable, '-u', str(HERE / 'train_visual_backbone_gtloss.py'),
               '--city', 'citya', '--source-checkpoint', str(SOURCE),
               '--samples', '18000', '--epochs', '1', '--batch-size', '16',
               '--backbone-lr', '5e-6', '--head-lr', '3e-5',
               '--unfreeze-last-blocks', '8', '--target-sigma-m', '4',
               '--distance-loss-weight', '10', '--live-sat-topk', topk,
               '--live-sat-random', random_count, '--uav-quarter-turn-augment',
               '--seed', seed, '--output', str(dest / 'train')]
        label = '%s GPU%s TRAIN' % (name, gpu)
        jobs.append((label, *launch(cmd, environment(base, gpu, dest), label,
                                   dest / 'train.log')))
    wait_all(jobs)
    jobs = []
    for name, gpu, _, _, _ in CASES:
        dest = OUT / name
        visual = dest / 'train/candidate_epoch_last.pt'
        if not visual.is_file():
            raise FileNotFoundError(visual)
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', str(TEMPORAL), '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest / 'validation')]
        label = '%s GPU%s VAL' % (name, gpu)
        jobs.append((label, *launch(cmd, environment(base, gpu, dest, visual),
                                   label, dest / 'validation.log')))
    wait_all(jobs)
    rows = []
    for name, gpu, topk, random_count, seed in CASES:
        dest = OUT / name / 'validation'
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'name': name, 'gpu': gpu, 'topk': int(topk),
                     'random': int(random_count), 'seed': int(seed),
                     'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
                     'report': str(dest / 'report.json')})
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        say('[RESULT] ' + json.dumps(row))


if __name__ == '__main__':
    main()
