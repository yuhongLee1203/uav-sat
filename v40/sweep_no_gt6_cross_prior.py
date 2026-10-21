"""Compare soft causal cross-track priors during front18 acquisition with GT-free inference and fixed 6x6 final MS."""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEST = HERE / 'checkpoint1/best_no_gt_20260926'
OUT = HERE / 'output/no_gt_6x6_20260927_a/cross_prior_sweep'
BASE = json.loads((BEST / 'run_settings_original.json').read_text())['environment']
TEMP = HERE / 'output/fused_temporal_citya_grid_parallel_20260926/grid_06x06/train/best_temporal_interim.pt'
VISUAL = BEST / 'visual_retrieval_A_only.pt'
WEIGHTS = (0, 1, 2, 4, 8, 16, 32)


def run(gpu, weight, temporal, visual, raw_visual_weight):
    dest = OUT / ('cross_prior_%04.2f' % weight)
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(BASE)
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'MS_GRID_SIZE': '6',
                'V40_TEMPORAL_FUSION_MLP': '1',
                'V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS': '1',
                'V40_ACQ_RAW_VISUAL_WEIGHT': str(raw_visual_weight),
                'V40_ACQ_LEARNED_LOGIT_SCALE': '16',
                'V40_ACQ_TEMPORAL_QUERY_BLEND': '0',
                'V40_ACQ_MOTION_PRIOR_WEIGHT': '0',
                'V40_ACQ_CROSS_PRIOR_WEIGHT': str(weight),
                'V40_VISUAL_CHECKPOINT_FILE': str(visual),
                'V40_FEATURE_CACHE_ROOT': str(dest / 'cache'),
                'V40_GT_FRONT18_ORACLE': '0', 'V40_GT_FRONT18_PATCH_ORACLE': '0',
                'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
                'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
                'V40_GT_JITTER_EVAL_M': '0', 'V40_GT_JITTER_PATCH_SELECT': '0',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2'})
    cmd = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
           '--checkpoint', str(temporal), '--city', 'citya',
           '--tta-rotations', '0,1,2,3', '--output', str(dest)]
    with (dest / 'console.log').open('w') as log:
        code = subprocess.run(cmd, cwd=HERE.parent, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('%s failed: %s' % (weight, dest / 'console.log'))
    report = json.loads((dest / 'report.json').read_text())
    settings = json.loads((dest / 'run_settings.json').read_text())
    assert report['valid_no_GT_inference_result'] and not report['current_frame_GT_model_input']
    assert settings['environment']['MS_GRID_SIZE'] == '6'
    row = {'weight': weight, 'gpu': gpu, 'MLE_m': report['MLE_m'],
           'P90_m': report['P90_m'],
           'SelectedCandidateCapture_pct': report['SelectedCandidateCapture_pct'],
           'report': str(dest / 'report.json')}
    print('[RESULT]', row, flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    visual_summary = HERE / 'output/no_gt_6x6_20260927_a/visual_search/summary.json'
    temporal_summary = HERE / 'output/no_gt_6x6_20260927_a/temporal_visual2061/summary.json'
    temporal, visual, raw_visual_weight = TEMP, VISUAL, 5
    if visual_summary.is_file():
        row = json.loads(visual_summary.read_text())[0]
        if row['MLE_m'] < 4.993845688700482:
            temporal = Path(row['temporal_checkpoint'])
            visual = Path(row['visual_checkpoint'])
    if temporal_summary.is_file():
        row = json.loads(temporal_summary.read_text())[0]
        if row['MLE_m'] < 4.814497651596793:
            temporal = Path(row['checkpoint'])
            raw_visual_weight = row['raw_visual_weight']
    stage2 = HERE / 'output/no_gt_6x6_20260927_a/visual_stage2/summary.json'
    if stage2.is_file():
        row = json.loads(stage2.read_text())[0]
        if row['MLE_m'] < 4.667706912743488:
            temporal = Path(row['temporal_checkpoint'])
            visual = Path(row['visual_checkpoint'])
    print('[PLAN] temporal=%s visual=%s' % (temporal, visual), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, (0, 5, 6)[i % 3], weight, temporal, visual, raw_visual_weight)
                   for i, weight in enumerate(WEIGHTS)]
        rows = [future.result() for future in futures]
    rows.sort(key=lambda row: row['MLE_m'])
    (OUT / 'summary.json').write_text(json.dumps({
        'temporal_checkpoint': str(temporal),
        'visual_checkpoint': str(visual),
        'raw_visual_weight': raw_visual_weight, 'variants': rows,
    }, indent=2) + '\n')
    print('[DONE]', OUT / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
