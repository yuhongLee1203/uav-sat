"""Same-seed CityA fused-temporal runs: GPU 5 final MS 18x18, GPU 6 final MS 6x6.

Prints live frame, route, epoch, loss, validation, progress and approximate ETA.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEST = HERE / 'checkpoint1' / 'best_no_gt_20260926'
BASELINE_SETTINGS = BEST / 'run_settings_original.json'
TRAIN_SETTINGS = HERE / 'checkpoint1' / 'code_before_fusion' / 'output_temporal_training_settings.json'
EXTRA = HERE / 'data' / 'official96_fixed_train_grid'
PRINT_LOCK = threading.Lock()


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def format_time(seconds):
    if seconds is None:
        return 'estimating'
    minutes, sec = divmod(max(0, round(seconds)), 60)
    return '%02d:%02d' % (minutes, sec)


def stream_process(process, label, log_path, total_units=None, epochs=None, route_count=None):
    start = time.monotonic()
    completed = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', buffering=1) as log:
        for line in process.stdout:
            log.write(line)
            message = line.rstrip()
            if not message:
                continue
            important = message.startswith(('[V40 ROUTE START]', '[V40 FRAME]',
                                            '[V40 EPISODE]', '[V40 VALIDATION]',
                                            '[V40 EARLY STOP]', '[V40 DONE]',
                                            '[V40 VALIDATION ONLY]', '[V40 FIXED TEST]'))
            if important:
                say('[%s] %s' % (label, message))
            if message.startswith(('[V40 EPISODE]', '[V40 VALIDATION]')) and total_units:
                completed += 1
                elapsed = time.monotonic() - start
                fraction = min(completed / total_units, 1.0)
                eta = (elapsed / completed * (total_units - completed)
                       if completed else None)
                filled = min(20, round(20 * fraction))
                bar = '#' * filled + '.' * (20 - filled)
                say('[%s] [%s] %d/%d steps (%.1f%%) elapsed %s ETA %s' % (
                    label, bar, completed, total_units, fraction * 100,
                    format_time(elapsed), format_time(eta)))
    code = process.wait()
    elapsed = time.monotonic() - start
    say('[%s] finished exit=%d elapsed=%s log=%s' % (
        label, code, format_time(elapsed), log_path))


def launch(command, env, label, log_path, total_units=None):
    say('[%s] starting GPU=%s final_MS=%sx%s; log=%s' % (
        label, env['CUDA_VISIBLE_DEVICES'], env.get('MS_GRID_SIZE', '?'),
        env.get('MS_GRID_SIZE', '?'), log_path))
    process = subprocess.Popen(command, cwd=HERE.parent, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    thread = threading.Thread(target=stream_process,
                              args=(process, label, log_path, total_units), daemon=True)
    thread.start()
    return process, thread, log_path


def wait_all(jobs):
    failures = []
    for label, process, thread, log_path in jobs:
        thread.join()
        if process.returncode:
            failures.append('%s exit=%d log=%s' % (label, process.returncode, log_path))
    if failures:
        raise RuntimeError('; '.join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=HERE / 'output' / 'fused_temporal_citya_grid_parallel_20260926')
    parser.add_argument('--seed', type=int, default=2033)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--skip-heldout-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_base = json.loads(TRAIN_SETTINGS.read_text())['environment']
    eval_base = json.loads(BASELINE_SETTINGS.read_text())['environment']
    common = {'V40_TEMPORAL_FUSION_MLP': '1',
              'V40_VISUAL_CHECKPOINT_ROOT': str(BEST / 'checkpoints'),
              'V40_PREPARED_ROOT': str(HERE / 'data' / 'official96_fixed_routes'),
              'V40_GT_FRONT18_ORACLE': '0',
              'V40_GT_FRONT18_PATCH_ORACLE': '0',
              'V40_GT_FINAL_MS_ORACLE_CENTER': '0',
              'V40_GT_FINAL_MS_PATCH_ORACLE': '0',
              'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
              'MKL_NUM_THREADS': '2'}
    route_count = 1 + len(list((EXTRA / 'citya').glob('train_*')))
    total_units = args.epochs * (route_count + 1)
    say('[PLAN] one seed=%d; %d epochs; %d train routes/epoch + 1 validation/epoch' % (
        args.seed, args.epochs, route_count))
    say('[PLAN] GPU 5 trains/evaluates 18x18; GPU 6 trains/evaluates 6x6 concurrently.')
    say('[PLAN] ETA is estimated from completed route/validation steps and may change.')
    variants = [(18, '5'), (6, '6')]
    train_jobs = []
    for grid, gpu in variants:
        label = 'GPU%s %dx%d TRAIN' % (gpu, grid, grid)
        jobdir = output / ('grid_%02dx%02d' % (grid, grid))
        env = os.environ.copy()
        env.update(train_base)
        env.update(common)
        env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': str(grid),
                    'V40_FEATURE_CACHE_ROOT': str(jobdir / 'train_cache')})
        command = [sys.executable, '-u', str(HERE / 'train_multicity_routes.py'),
                   '--train-cities', 'citya', '--eval-cities', 'citya',
                   '--supervision', 'gt_train', '--init', 'random_temporal',
                   '--epochs', str(args.epochs), '--max-train-frames', '136',
                   '--lr', '1e-5', '--dense-offbank-loss',
                   '--acquisition-target-sigma-m', '4',
                   '--extra-train-routes-dir', str(EXTRA),
                   '--validate-every-epochs', '1', '--early-stop-patience', '3',
                   '--skip-test', '--seed', str(args.seed),
                   '--output', str(jobdir / 'train')]
        log_path = jobdir / 'train.log'
        process, thread, _ = launch(command, env, label, log_path, total_units)
        train_jobs.append((label, process, thread, log_path))
    wait_all(train_jobs)

    validation_jobs = []
    for grid, gpu in variants:
        name = 'grid_%02dx%02d' % (grid, grid)
        jobdir = output / name
        checkpoint = jobdir / 'train' / 'best_temporal_interim.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        env = os.environ.copy()
        env.update(eval_base)
        env.update(common)
        env.update({'CUDA_VISIBLE_DEVICES': gpu, 'MS_GRID_SIZE': str(grid),
                    'V40_FEATURE_CACHE_ROOT': str(jobdir / 'val_cache')})
        command = [sys.executable, '-u', str(HERE / 'evaluate_official96_validation.py'),
                   '--checkpoint', str(checkpoint), '--city', 'citya',
                   '--tta-rotations', '0,1,2,3',
                   '--output', str(jobdir / 'validation')]
        label = 'GPU%s %dx%d TTA VAL' % (gpu, grid, grid)
        log_path = jobdir / 'validation.log'
        process, thread, _ = launch(command, env, label, log_path)
        validation_jobs.append((label, process, thread, log_path))
    wait_all(validation_jobs)

    results = []
    for grid, gpu in variants:
        jobdir = output / ('grid_%02dx%02d' % (grid, grid))
        report_path = jobdir / 'validation' / 'report.json'
        report = json.loads(report_path.read_text())
        results.append({'grid_size': grid, 'candidate_count': grid * grid,
                        'gpu': gpu, 'seed': args.seed,
                        'checkpoint': str(jobdir / 'train' / 'best_temporal_interim.pt'),
                        'MLE_m': report['MLE_m'], 'P90_m': report['P90_m'],
                        'LSR@15_pct': report['LSR@15_pct'],
                        'report': str(report_path)})
        say('[RESULT] GPU%s %dx%d MLE=%.3fm P90=%.3fm LSR@15=%.2f%%' % (
            gpu, grid, grid, report['MLE_m'], report['P90_m'], report['LSR@15_pct']))
    selected = min(results, key=lambda item: item['MLE_m'])
    summary = {'architecture': 'MeanDiffConcatSingleMLP', 'seed': args.seed,
               'epochs_requested': args.epochs, 'comparison_route': 'citya/val_01',
               'baseline_previous_architecture_val_MLE_m':
               json.loads((BEST / 'report.json').read_text())['MLE_m'],
               'variants': results, 'selected_on_validation': selected,
               'heldout_test_evaluated': not args.skip_heldout_test,
               'comparison_note': 'Same seed and training schedule; each grid has its own training run and checkpoint.'}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    say('[SELECTED] validation MLE chooses %dx%d' %
        (selected['grid_size'], selected['grid_size']))
    if not args.skip_heldout_test:
        env = os.environ.copy()
        env.update(eval_base)
        env.update(common)
        env.update({'CUDA_VISIBLE_DEVICES': selected['gpu'],
                    'MS_GRID_SIZE': str(selected['grid_size']),
                    'V40_FEATURE_CACHE_ROOT': str(output / 'selected_test_cache')})
        command = [sys.executable, '-u', str(HERE / 'evaluate_official96_fixed_test.py'),
                   '--checkpoint', selected['checkpoint'], '--city', 'citya',
                   '--tta-rotations', '0,1,2,3',
                   '--output', str(output / 'selected_heldout_test')]
        label = 'GPU%s HELDOUT TEST' % selected['gpu']
        log_path = output / 'selected_heldout_test.log'
        process, thread, _ = launch(command, env, label, log_path)
        wait_all([(label, process, thread, log_path)])
    say('[DONE] comparison: %s' % (output / 'comparison.json'))


if __name__ == '__main__':
    main()
