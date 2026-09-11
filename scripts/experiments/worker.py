"""Run queued GPU experiments under supervisor, keeping durable job records."""
import argparse
import datetime as dt
import json
import hashlib
import tarfile
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def snapshot_sources():
    paths = sorted(p for folder in ('oat', 'scripts', 'tests')
                   for p in (ROOT / folder).rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = ROOT / 'output' / 'source_snapshots'
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'{identity}.tar.gz'
    if not archive.exists():
        temporary = destination / f'{identity}.{os.getpid()}.tmp'
        with tarfile.open(temporary, 'w:gz') as tar:
            for p in paths:
                tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        temporary.replace(archive)
    return {'sha256': identity, 'archive': str(archive), 'files': hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--queue', help='Optional queue name for an independent service on a shared GPU')
    args = parser.parse_args()
    queue_name = args.queue or f'gpu{args.gpu}'
    if not queue_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in queue_name):
        parser.error('queue must be a simple alphanumeric name')
    queue = ROOT / 'output' / 'jobs' / queue_name
    queue.mkdir(parents=True, exist_ok=True)
    child = None
    stopped = False

    def terminate(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(f'{now()} worker pid={os.getpid()} gpu={args.gpu}', flush=True)
    while not stopped:
        jobs = sorted(queue.glob('*.pending.json'))
        if not jobs:
            time.sleep(2)
            continue
        pending = jobs[0]
        claimed = pending.with_name(pending.name.replace('.pending.', '.running.'))
        try:
            pending.rename(claimed)
        except FileNotFoundError:
            continue
        spec = json.loads(claimed.read_text())
        job_id = claimed.name.removesuffix('.running.json')
        artifact = queue / job_id
        artifact.mkdir(exist_ok=True)
        status_path = artifact / 'status.json'
        status = {'id': job_id, 'gpu': args.gpu, 'worker_pid': os.getpid(),
                  'state': 'starting', 'start_time': now(), 'spec': spec,
                  'source_snapshot': snapshot_sources()}
        atomic_json(status_path, status)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': str(args.gpu), 'MUJOCO_GL': 'egl',
                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                    'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                    'WANDB_MODE': 'offline', 'PYTHONPATH': str(ROOT)})
        env.update({str(k): str(v) for k, v in spec.get('env', {}).items()})
        try:
            with (artifact / 'output.log').open('a', buffering=1) as logfile:
                child = subprocess.Popen(spec['command'], cwd=ROOT, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
                status.update(state='running', pid=child.pid)
                atomic_json(status_path, status)
                print(f'{now()} start {job_id} pid={child.pid}', flush=True)
                for line in child.stdout:
                    logfile.write(line)
                    print(line, end='', flush=True)
                returncode = child.wait()
            status.update(state='completed' if returncode == 0 else 'failed',
                          returncode=returncode, end_time=now())
        except Exception as exc:
            status.update(state='failed', error=repr(exc), end_time=now())
        finally:
            child = None
            atomic_json(status_path, status)
            claimed.rename(claimed.with_name(claimed.name.replace('.running.', '.done.')))
            print(f'{now()} finish {job_id}: {status["state"]}', flush=True)


if __name__ == '__main__':
    main()
