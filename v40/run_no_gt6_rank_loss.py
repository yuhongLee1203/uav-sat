"""Train/evaluate GT-loss-only gallery margin variants; no GT at inference."""
import json
import os
import sys
from pathlib import Path
from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / 'output/no_gt_6x6_20260927_a'
SOURCE_REPORT = BASE_DIR / 'cross_prior_sweep/cross_prior_0.00'
VISUAL = BASE_DIR / 'visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'
TEMPORAL = BASE_DIR / 'temporal_visual2061/raw10_sigma4/train/best_temporal_interim.pt'
OUT = HERE / 'output/no_gt_6x6_20261008_rank_loss'
CASES = [('rank01', '0', '0.1', '2101'),
         ('rank03', '5', '0.3', '2103'),
         ('rank06', '6', '0.6', '2107')]


def environment(base, gpu, dest, visual=None):
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
    if visual:
        env['V40_VISUAL_CHECKPOINT_FILE'] = str(visual)
    return env


def main():
    base = json.loads((SOURCE_REPORT / 'run_settings.json').read_text())['environment']
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, weight, seed in CASES:
        dest = OUT / name
        candidate = dest / 'train/candidate_epoch_last.pt'
        if candidate.is_file():
            continue
        cmd = [sys.executable, '-u', str(HERE / 'train_visual_backbone_gtloss.py'),
               '--city', 'citya', '--source-checkpoint', str(VISUAL),
               '--samples', '18000', '--epochs', '1', '--batch-size', '48',
               '--backbone-lr', '5e-6', '--head-lr', '3e-5',
               '--unfreeze-last-blocks', '8', '--target-sigma-m', '4',
               '--distance-loss-weight', '10', '--live-sat-topk', '16',
               '--live-sat-random', '16', '--uav-quarter-turn-augment',
               '--hard-negative-margin-weight', weight,
               '--hard-negative-min-distance-m', '20',
               '--hard-negative-logit-margin', '2',
               '--hard-negative-temperature', '4',
               '--seed', seed, '--output', str(dest / 'train')]
        label = '%s GPU%s TRAIN' % (name, gpu)
        jobs.append((label, *launch(cmd, environment(base, gpu, dest),
                                   label, dest / 'train.log')))
    wait_all(jobs)
    jobs = []
    for name, gpu, weight, seed in CASES:
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
    for name, gpu, weight, seed in CASES:
        dest = OUT / name / 'validation'
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'name': name, 'gpu': gpu, 'margin_weight': float(weight),
                     'seed': int(seed), 'MLE_m': report['MLE_m'],
                     'P90_m': report['P90_m'],
                     'SelectedCandidateCapture_pct': report['SelectedCandidateCapture_pct'],
                     'checkpoint': str(OUT / name / 'train/candidate_epoch_last.pt'),
                     'report': str(dest / 'report.json')})
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        say('[RESULT] ' + json.dumps(row))


if __name__ == '__main__':
    main()
