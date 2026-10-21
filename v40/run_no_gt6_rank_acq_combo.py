"""Evaluate trained rank-loss visual candidates with trained acquisition refinement."""
import json
import os
import sys
from pathlib import Path
from run_fused_temporal_citya import launch, wait_all, say

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep/cross_prior_0.00'
VIS_ROOT = HERE / 'output/no_gt_6x6_20261008_rank_loss'
TEMPORAL = HERE / 'output/no_gt_6x6_20261008_acq_refine/train/best_temporal_interim.pt'
OUT = HERE / 'output/no_gt_6x6_20261008_rank_acq_combo'
CASES = [('rank01', '0'), ('rank03', '5'), ('rank06', '6')]


def main():
    base = json.loads((SOURCE / 'run_settings.json').read_text())['environment']
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, gpu in CASES:
        visual = VIS_ROOT / name / 'train/candidate_epoch_last.pt'
        if not visual.is_file():
            raise FileNotFoundError(visual)
        dest = OUT / name
        env = os.environ.copy()
        env.update(base)
        env.update({'CUDA_VISIBLE_DEVICES': gpu,
                    'V40_VISUAL_CHECKPOINT_FILE': str(visual),
                    'V40_FEATURE_CACHE_ROOT': str(dest / 'cache'),
                    'MS_GRID_SIZE': '6', 'MS_BANDWIDTH_M': '18',
                    'V40_GT_FRONT18_ORACLE': '0', 'V40_GT_FRONT18_PATCH_ORACLE': '0',
                    'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
                    'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
                    'V40_GT_JITTER_EVAL_M': '0', 'V40_GT_JITTER_PATCH_SELECT': '0',
                    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2'})
        cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
               '--checkpoint', str(TEMPORAL), '--city', 'citya',
               '--tta-rotations', '0,1,2,3', '--output', str(dest)]
        label = '%s + acq_refine GPU%s' % (name, gpu)
        jobs.append((label, *launch(cmd, env, label, dest / 'console.log')))
    wait_all(jobs)
    rows = []
    for name, gpu in CASES:
        dest = OUT / name
        report = json.loads((dest / 'report.json').read_text())
        settings = json.loads((dest / 'run_settings.json').read_text())
        assert report['valid_no_GT_inference_result']
        assert not report['current_frame_GT_model_input']
        assert settings['environment']['MS_GRID_SIZE'] == '6'
        rows.append({'visual': name, 'temporal': str(TEMPORAL),
                     'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
                     'SelectedCandidateCapture_pct': report['SelectedCandidateCapture_pct'],
                     'report': str(dest / 'report.json')})
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        say('[RESULT] ' + json.dumps(row))


if __name__ == '__main__':
    main()
