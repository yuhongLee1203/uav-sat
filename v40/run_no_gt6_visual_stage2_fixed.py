"""Continue CityA visual GT-loss training on GPUs 0/5/6; validate with fixed 6x6 MS."""
import json
import os
import sys
from pathlib import Path

from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
BEST = HERE / 'checkpoint1/best_no_gt_20260926'
OUT = HERE / 'output/no_gt_6x6_20260927_a/visual_stage2_fixed'
BASE = json.loads((BEST / 'run_settings_original.json').read_text())['environment']
SOURCE = HERE / 'output/no_gt_6x6_20260927_a/visual_search/continue_seed2061/train/checkpoints/citya/visual_retrieval_A_only.pt'
TEMPORAL_PRIOR = HERE / 'output/no_gt_6x6_20260927_a/temporal_visual2061/raw10_sigma4/train/best_temporal_interim.pt'
CASES = [
    ('continue_sigma4', '0', '2081', '4', '10'),
    ('soft_sigma6', '5', '2081', '6', '10'),
    ('distance5', '6', '2081', '4', '5'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu, seed, sigma, distance in CASES:
        dest = OUT / name
        env = os.environ.copy()
        env.update(BASE)
        env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': '6',
                    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2'})
        cmd = [sys.executable, '-u', str(HERE / 'train_visual_backbone_gtloss.py'),
               '--city', 'citya', '--source-checkpoint', str(SOURCE),
               '--samples', '18000', '--epochs', '1', '--batch-size', '48',
               '--backbone-lr', '5e-6', '--head-lr', '3e-5',
               '--unfreeze-last-blocks', '8', '--target-sigma-m', sigma,
               '--distance-loss-weight', distance, '--live-sat-topk', '16',
               '--live-sat-random', '16', '--uav-quarter-turn-augment',
               '--seed', seed, '--output', str(dest / 'train')]
        label = name + ' GPU' + gpu
        proc, thread, log = launch(cmd, env, label, dest / 'train.log')
        jobs.append((label, proc, thread, log))
    wait_all(jobs)
    temporal_rows_path = HERE / 'output/no_gt_6x6_20260927_a/temporal_visual2061/summary.json'
    temporal = TEMPORAL_PRIOR
    if temporal_rows_path.is_file():
        rows = json.loads(temporal_rows_path.read_text())
        if rows[0]['MLE_m'] < 4.814497651596793:
            temporal = Path(rows[0]['checkpoint'])
    jobs = []
    for name, gpu, _, _, _ in CASES:
        dest = OUT / name
        visual = dest / 'train/checkpoints/citya/visual_retrieval_A_only.pt'
        if not visual.is_file():
            raise FileNotFoundError(visual)
        env = os.environ.copy()
        env.update(BASE)
        env.update({
            'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': '6',
            'V40_TEMPORAL_FUSION_MLP': '1',
            'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS': '1',
            'V40_ACQ_RAW_VISUAL_WEIGHT': '10',
            'V40_ACQ_LEARNED_LOGIT_SCALE': '16',
            'V40_VISUAL_CHECKPOINT_FILE': str(visual),
            'V40_FEATURE_CACHE_ROOT': str(dest / 'val_cache'),
            'V40_GT_FRONT18_ORACLE': '0',
            'V40_GT_FRONT18_PATCH_ORACLE': '0',
            'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
            'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
            'V40_GT_JITTER_EVAL_M': '0',
            'V40_GT_JITTER_PATCH_SELECT': '0',
            'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2',
        })
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', str(temporal), '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest / 'validation')]
        label = name + ' GPU' + gpu + ' TTA val'
        proc, thread, log = launch(cmd, env, label, dest / 'validation.log')
        jobs.append((label, proc, thread, log))
    wait_all(jobs)
    rows = []
    for name, gpu, seed, sigma, distance in CASES:
        dest = OUT / name
        report = json.loads((dest / 'validation/report.json').read_text())
        settings = json.loads((dest / 'validation/run_settings.json').read_text())
        assert report['valid_no_GT_inference_result'] and not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'name': name, 'gpu': gpu, 'seed': int(seed),
                     'target_sigma_m': float(sigma), 'distance_loss_weight': float(distance),
                     'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
                     'visual_checkpoint': str(dest / 'train/checkpoints/citya/visual_retrieval_A_only.pt'),
                     'temporal_checkpoint': str(temporal),
                     'report': str(dest / 'validation/report.json')})
        say('[RESULT] %s GPU%s MLE %.3f m' % (name, gpu, report['MLE_m']))
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    say('[DONE] ' + str(OUT / 'summary.json'))


if __name__ == '__main__':
    main()
