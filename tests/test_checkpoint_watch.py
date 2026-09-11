"""Filesystem-only regressions for immutable, restartable checkpoint scheduling."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/experiments/watch_checkpoints.py'
spec = importlib.util.spec_from_file_location('watch_checkpoints', SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def fixture(tmp_path, *, enabled=True, queue='gpu0', job_id='test_job'):
    run_dir = tmp_path / 'training'
    run_dir.mkdir(exist_ok=True)
    checkpoint = run_dir / 'ep-0009.ckpt'
    manifest_path = tmp_path / 'watch.json'
    source = {'version': 1, 'poll_seconds': 0.1, 'targets': [
        {'enabled': enabled, 'run_dir': str(run_dir), 'checkpoint': str(checkpoint),
         'queue': queue, 'job_id': job_id, 'command': ['/venv/oat/bin/python', 'evaluate.py', str(checkpoint)],
         'purpose': 'Temporary regression evaluation'}]}
    manifest_path.write_text(json.dumps(source))
    return manifest_path, checkpoint


def poll(manifest_path, jobs_root=None, status_path=None):
    return watch.poll_once(watch.load_manifest(manifest_path), manifest_path,
                           status_path or manifest_path.with_suffix('.status.json'),
                           jobs_root or manifest_path.parent / 'jobs')


def test_waits_for_exact_nonempty_checkpoint_and_disabled_targets_do_not_block(tmp_path):
    manifest_path, checkpoint = fixture(tmp_path)
    checkpoint.with_suffix('.ckpt.tmp').write_bytes(b'not atomically published yet')
    report = poll(manifest_path)
    assert report['targets'][0]['state'] == 'missing_checkpoint'
    checkpoint.touch()
    assert poll(manifest_path)['targets'][0]['state'] == 'invalid_checkpoint'
    assert not list((tmp_path / 'jobs').rglob('*.pending.json'))
    checkpoint.write_bytes(b'atomic checkpoint substitute')
    report = poll(manifest_path)
    assert report['state'] == 'complete'
    assert report['targets'][0]['state'] == 'queued'
    pending = tmp_path / 'jobs/gpu0/test_job.pending.json'
    contents = pending.read_bytes()
    assert json.loads(contents)['command'][-1] == str(checkpoint)
    assert poll(manifest_path)['targets'][0]['queued_utc'] == report['targets'][0]['queued_utc']
    assert pending.read_bytes() == contents
    # Even manually archived queue records do not trigger a retry from this receipt.
    pending.rename(tmp_path / 'archived.json')
    assert poll(manifest_path)['state'] == 'complete'
    assert not pending.exists()
    source = json.loads(manifest_path.read_text())
    source['targets'][0].update(enabled=False, checkpoint=str(checkpoint.with_name('missing.ckpt')))
    manifest_path.write_text(json.dumps(source))
    assert poll(manifest_path)['targets'][0]['state'] == 'disabled'
    source['targets'][0]['enabled'] = True
    manifest_path.write_text(json.dumps(source))
    assert poll(manifest_path)['targets'][0]['state'] == 'queued'
    assert not pending.exists()  # Disabling/re-enabling cannot erase an old receipt.


@pytest.mark.parametrize('state', ['pending', 'running', 'done', 'artifact'])
def test_existing_jobs_are_never_replaced_or_retried_across_queues(tmp_path, state):
    manifest_path, checkpoint = fixture(tmp_path)
    checkpoint.write_bytes(b'checkpoint')
    queue = tmp_path / 'jobs/other_queue'
    queue.mkdir(parents=True)
    existing = queue / ('test_job' if state == 'artifact' else f'test_job.{state}.json')
    if state == 'artifact':
        existing.mkdir()
        existing = existing / 'status.json'
    original = b'{"state": "failed", "immutable": true}\n'
    existing.write_bytes(original)
    report = poll(manifest_path)
    assert report['state'] == 'complete'
    assert report['targets'][0]['state'] == f'existing_{state}'
    assert existing.read_bytes() == original
    assert not (tmp_path / 'jobs/gpu0/test_job.pending.json').exists()


def test_exclusive_json_publication_never_overwrites(tmp_path):
    path = tmp_path / 'job.pending.json'
    watch.write_json(path, {'command': ['original']}, exclusive=True)
    with pytest.raises(FileExistsError):
        watch.write_json(path, {'command': ['replacement']}, exclusive=True)
    assert json.loads(path.read_text()) == {'command': ['original']}
    assert not list(tmp_path.glob('*.tmp'))


def _concurrent_poll(manifest_path, status_path, jobs_root, start):
    start.wait()
    poll(Path(manifest_path), Path(jobs_root), Path(status_path))


def test_concurrent_watchers_and_worker_rename_publish_once(tmp_path):
    manifest_path, checkpoint = fixture(tmp_path)
    checkpoint.write_bytes(b'checkpoint')
    context = multiprocessing.get_context('fork')
    start = context.Event()
    workers = [context.Process(target=_concurrent_poll,
               args=(str(manifest_path), str(tmp_path / f'status-{i}.json'), str(tmp_path / 'jobs'), start))
               for i in range(4)]
    for worker in workers:
        worker.start()
    start.set()
    pending = tmp_path / 'jobs/gpu0/test_job.pending.json'
    running = pending.with_name('test_job.running.json')
    deadline = time.monotonic() + 5
    while not pending.exists() and time.monotonic() < deadline:
        time.sleep(.005)
    assert pending.exists()
    # Simulate worker claiming immediately, while the other producers are active.
    assert json.loads(pending.read_text())['command']
    pending.rename(running)
    done = running.with_name('test_job.done.json')
    running.rename(done)
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    states = [json.loads((tmp_path / f'status-{i}.json').read_text())['targets'][0]['state'] for i in range(4)]
    assert states.count('queued') == 1
    assert len(list((tmp_path / 'jobs').rglob('test_job.*.json'))) == 1
    assert done.exists()


@pytest.mark.parametrize('field,value', [('queue', '../escape'), ('job_id', 'x.pending.json'),
                                         ('checkpoint', '*.ckpt'), ('command', 'shell command')])
def test_manifest_rejects_unsafe_names_wildcards_and_shell_strings(tmp_path, field, value):
    manifest_path, _ = fixture(tmp_path)
    source = json.loads(manifest_path.read_text())
    source['targets'][0][field] = value
    manifest_path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        watch.load_manifest(manifest_path)


def test_sigterm_records_clean_stop_without_enqueuing(tmp_path):
    manifest_path, _ = fixture(tmp_path)
    process = subprocess.Popen([sys.executable, str(SCRIPT), '--manifest', str(manifest_path),
                                '--jobs-root', str(tmp_path / 'jobs'), '--poll-seconds', '30'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    status = manifest_path.with_suffix('.status.json')
    deadline = time.monotonic() + 5
    try:
        while not status.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert status.exists()
        process.send_signal(signal.SIGTERM)
        output, error = process.communicate(timeout=5)
        assert process.returncode == 0, (output, error)
        assert json.loads(status.read_text())['state'] == 'stopped'
        assert not list((tmp_path / 'jobs').rglob('*.pending.json'))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
