"""Concise, read-only progress for local supervised Past2Next experiments."""
from pathlib import Path
import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def last_rows(path, limit=1000):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 256000))
            lines = f.read().decode(errors='replace').splitlines()
        result = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except ValueError:
                pass
        return result
    except OSError:
        return []


def live(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-inactive', action='store_true', help='Also print historical training runs')
    args = parser.parse_args()
    active_training = set()
    out = ROOT / 'output'
    for spec in sorted((out / 'jobs').glob('*/*.running.json')):
        name = spec.name.removesuffix('.running.json')
        status = read_json(spec.parent / name / 'status.json')
        running = live(status.get('pid'))
        print(f"RUN {spec.parent.name}/{name}: recorded={status.get('state')} pid={status.get('pid')} live={running}")
        if running:
            for argument in status.get('spec', {}).get('command', []):
                if argument.startswith('hydra.run.dir='):
                    active_training.add(Path(argument.split('=', 1)[1]).resolve())
    for spec in sorted((out / 'jobs').glob('*/*.pending.json')):
        print(f"QUEUE {spec.parent.name}/{spec.name.removesuffix('.pending.json')}")
    for folder in sorted((out / 'evaluations').glob('*')):
        if not folder.is_dir() or read_json(folder / 'metadata.json').get('status') in ('complete', 'completed'):
            continue
        episodes = last_rows(folder / 'episodes.jsonl')
        if episodes:
            print(f"EVAL {folder.name}: {sum(bool(d.get('success')) for d in episodes)}/{len(episodes)} observed episodes; incomplete")
    for folder in sorted((out / 'training').glob('*')):
        if not args.include_inactive and folder.resolve() not in active_training:
            continue
        rows = last_rows(folder / 'logs.json')
        if rows:
            row = rows[-1]
            print(f"TRAIN {folder.name}: last recorded epoch={row.get('epoch')} batch={row.get('global_step')}; liveness requires its service/PID check")


if __name__ == '__main__':
    main()
