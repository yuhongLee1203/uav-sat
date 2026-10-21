"""Live terminal dashboard for the two running CityA fused-temporal jobs."""
import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'output' / 'fused_temporal_citya_grid_parallel_20260926'


def stamp(seconds):
    if seconds is None:
        return 'estimating'
    minutes, sec = divmod(max(0, int(seconds)), 60)
    return '%02d:%02d' % (minutes, sec)


def render(root):
    rows = []
    for grid, gpu in ((18, 5), (6, 6)):
        work = root / ('grid_%02dx%02d' % (grid, grid))
        log = work / 'train.log'
        settings = work / 'train' / 'run_settings.json'
        if not log.is_file():
            rows.append('GPU %d %dx%d: waiting for log' % (gpu, grid, grid))
            continue
        content = log.read_text(errors='replace')
        episodes = content.count('[V40 EPISODE]')
        validations = content.count('[V40 VALIDATION]')
        epochs = 4
        if settings.is_file():
            try:
                epochs = int(json.loads(settings.read_text())['arguments']['epochs'])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
        total = epochs * 14
        done = episodes + validations
        frac = min(done / total, 1)
        filled = min(24, round(frac * 24))
        elapsed = time.time() - settings.stat().st_mtime if settings.is_file() else None
        eta = (elapsed * (total - done) / done
               if elapsed is not None and done else None)
        last_frame = re.findall(r'\[V40 FRAME\] ([^\n]+)', content)
        last_val = re.findall(r'\[V40 VALIDATION\] ([^\n]+)', content)
        finished = '[V40 DONE]' in content
        rows.append('GPU %d  %2dx%2d  [%s] %2d/%d  %5.1f%%  elapsed %s  ETA %s%s' % (
            gpu, grid, grid, '#' * filled + '.' * (24 - filled), done, total,
            frac * 100, stamp(elapsed), 'done' if finished else stamp(eta),
            '  TRAIN DONE' if finished else ''))
        if last_frame:
            rows.append('  latest frame: ' + last_frame[-1])
        if last_val:
            rows.append('  latest validation: ' + last_val[-1])
    comparison = root / 'comparison.json'
    if comparison.is_file():
        report = json.loads(comparison.read_text())
        rows.append('VALIDATION RESULTS: ' + ', '.join(
            '%dx%d MLE %.3f m' % (r['grid_size'], r['grid_size'], r['MLE_m'])
            for r in report['variants']))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        print('\n'.join(render(args.output.resolve())), flush=True)
        if args.once or (args.output / 'comparison.json').is_file():
            break
        time.sleep(5)
        print('', flush=True)


if __name__ == '__main__':
    main()
