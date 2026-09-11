#!/usr/bin/env python3
"""Queue a finite set of checkpoint evaluations, then exit (stdlib, POSIX).

Usage: python scripts/experiments/watch_checkpoints.py --manifest output/checkpoint_watch.json
Paths in the manifest are relative to the repository, except checkpoint paths,
which are relative to their run_dir. Commands are literal argv lists for worker.py.
The manifest is read once. Disabled targets do not prevent completion. --once
performs one poll and exits 2 if checkpoints are still missing, otherwise 0.

Only exact, positive-size regular .ckpt files are eligible: training must publish
them atomically. This watcher neither inspects weights nor waits for evaluation
completion. Job IDs are global across queues. Existing queue records/artifacts and its own queued receipts are
terminal, even for failed jobs; it never retries, replaces, or deletes jobs.
"""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r'[A-Za-z0-9_-]{1,160}\Z')
TERMINAL = {'queued', 'existing_pending', 'existing_running', 'existing_done',
            'existing_artifact', 'existing_job'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def exact_path(value, base, label):
    if not isinstance(value, str) or not value or any(c in value for c in '*?[]\0'):
        raise ValueError(f'{label} must be an exact path without wildcards')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_manifest(path):
    source = json.loads(path.read_text())
    if source.get('version') != 1 or not isinstance(source.get('targets'), list):
        raise ValueError('manifest requires version=1 and a finite targets list')
    interval = source.get('poll_seconds', 30)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('poll_seconds must be finite and positive')
    targets, identities = [], set()
    for item in source['targets']:
        if not isinstance(item, dict):
            raise ValueError('each target must be an object')
        target = dict(item)
        for key in ('queue', 'job_id'):
            if not isinstance(target.get(key), str) or not NAME.fullmatch(target[key]):
                raise ValueError(f'{key} must be a simple alphanumeric, underscore or hyphen name')
        identity = target['job_id']
        if identity in identities:
            raise ValueError(f'duplicate job_id target: {identity}')
        identities.add(identity)
        if not isinstance(target.get('enabled', True), bool):
            raise ValueError('enabled must be boolean')
        target['enabled'] = target.get('enabled', True)
        run_dir = exact_path(target.get('run_dir'), ROOT, 'run_dir')
        checkpoint = exact_path(target.get('checkpoint'), run_dir, 'checkpoint')
        if checkpoint.suffix != '.ckpt' or not checkpoint.is_relative_to(run_dir):
            raise ValueError('checkpoint must be an exact .ckpt path inside run_dir')
        target.update(run_dir=str(run_dir), checkpoint=str(checkpoint))
        command = target.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg and '\0' not in arg for arg in command):
            raise ValueError('command must be a nonempty literal argv list')
        if not isinstance(target.get('purpose'), str) or not target['purpose'].strip():
            raise ValueError('purpose must be a nonempty string')
        env = target.get('env', {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError('env must map strings to strings')
        targets.append(target)
    return {'version': 1, 'poll_seconds': interval, 'targets': targets,
            'sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()}


def write_json(path, payload, exclusive=False):
    """Publish complete JSON; exclusive mode cannot overwrite an existing job."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def producer_lock(jobs_root, stop):
    """Serialize watcher producers across manifests, including status updates."""
    jobs_root.mkdir(parents=True, exist_ok=True)
    with (jobs_root / '.checkpoint_watch.lock').open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if stop.wait(0.1):
                    yield False
                    return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def existing_job(queue, job_id):
    # Match the worker's monotonic rename order, so an in-flight transition
    # cannot disappear between checking its old name and its new name.
    for state in ('pending', 'running', 'done'):
        path = queue / f'{job_id}.{state}.json'
        if path.exists():
            return f'existing_{state}', str(path)
    artifact = queue / job_id
    if artifact.exists():
        return 'existing_artifact', str(artifact)
    return None


def find_existing_job(jobs_root, queue_name, job_id):
    # Job IDs are global: a queue reassignment must not duplicate an evaluation.
    queues = [jobs_root / queue_name]
    queues.extend(path for path in sorted(jobs_root.iterdir())
                  if path.is_dir() and path.name != queue_name and NAME.fullmatch(path.name))
    for queue in queues:
        result = existing_job(queue, job_id)
        if result:
            return result
    return None


def poll_once(manifest, manifest_path, status_path, jobs_root, stop=None):
    stop = stop or threading.Event()
    with producer_lock(jobs_root, stop) as acquired:
        if not acquired:
            return None
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        receipts = dict(previous.get('queued_receipts', {}))
        receipts.update({row['job_id']: row for row in previous.get('targets', [])
                         if row.get('state') in TERMINAL})
        rows = []
        for target in manifest['targets']:
            identity = target['job_id']
            row = {key: target[key] for key in ('queue', 'job_id', 'run_dir', 'checkpoint', 'purpose', 'enabled')}
            row['checked_utc'] = now()
            if not target['enabled']:
                row['state'] = 'disabled'
            elif receipts.get(identity, {}).get('state') in TERMINAL:
                # A durable queued receipt also prevents retries if an operator
                # later moves queue records elsewhere. Preserve original facts.
                row = dict(receipts[identity], checked_utc=row['checked_utc'])
            else:
                queue = jobs_root / target['queue']
                existing = find_existing_job(jobs_root, target['queue'], target['job_id'])
                if existing:
                    row.update(state=existing[0], queue_path=existing[1])
                elif stop.is_set():
                    row['state'] = 'stopped'
                else:
                    checkpoint = Path(target['checkpoint'])
                    try:
                        info = checkpoint.stat()
                    except FileNotFoundError:
                        row['state'] = 'missing_checkpoint'
                    except OSError as exc:
                        row.update(state='checkpoint_unavailable', error=str(exc))
                    else:
                        row['checkpoint_bytes'] = info.st_size
                        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                            row['state'] = 'invalid_checkpoint'
                        else:
                            queued_utc = now()
                            pending = queue / f"{target['job_id']}.pending.json"
                            spec = {'command': target['command'], 'purpose': target['purpose'],
                                    'checkpoint_watch': {'manifest': str(manifest_path),
                                                         'run_dir': target['run_dir'],
                                                         'checkpoint': str(checkpoint),
                                                         'queued_utc': queued_utc}}
                            if target.get('env'):
                                spec['env'] = target['env']
                            try:
                                write_json(pending, spec, exclusive=True)
                            except FileExistsError:
                                existing = find_existing_job(jobs_root, target['queue'], target['job_id'])
                                row.update(state=existing[0] if existing else 'existing_job',
                                           queue_path=existing[1] if existing else str(pending))
                            else:
                                row.update(state='queued', queued_utc=queued_utc, queue_path=str(pending))
                                print(f"{queued_utc} queued {target['queue']}/{target['job_id']}", flush=True)
            if row['state'] in TERMINAL:
                receipts[row['job_id']] = row
            rows.append(row)
        complete = all(row['state'] in TERMINAL | {'disabled'} for row in rows)
        report = {'version': 1, 'manifest': str(manifest_path), 'manifest_sha256': manifest['sha256'],
                  'updated_utc': now(), 'state': 'complete' if complete else ('stopped' if stop.is_set() else 'watching'),
                  'targets': rows, 'queued_receipts': receipts}
        write_json(status_path, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--status', type=Path, help='default: manifest basename + .status.json')
    parser.add_argument('--jobs-root', type=Path, default=ROOT / 'output/jobs')
    parser.add_argument('--poll-seconds', type=float)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    interval = manifest['poll_seconds'] if args.poll_seconds is None else args.poll_seconds
    if not math.isfinite(interval) or interval <= 0:
        parser.error('--poll-seconds must be finite and positive')
    status_path = args.status.resolve() if args.status else manifest_path.with_suffix('.status.json')
    if status_path.is_relative_to(args.jobs_root.resolve()):
        parser.error('--status must be outside --jobs-root to protect immutable job records')
    if status_path == manifest_path:
        parser.error('--status must differ from --manifest')
    stop = threading.Event()
    def terminate(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while True:
        report = poll_once(manifest, manifest_path, status_path, args.jobs_root.resolve(), stop)
        if report is not None:
            counts = {state: sum(row['state'] == state for row in report['targets'])
                      for state in sorted({row['state'] for row in report['targets']})}
            print(f"{now()} {report['state']}: {counts}", flush=True)
            if report['state'] == 'complete':
                return 0
        if stop.is_set():
            return 0
        if args.once:
            return 2
        stop.wait(interval)


if __name__ == '__main__':
    raise SystemExit(main())
