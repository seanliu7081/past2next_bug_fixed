"""Preserve checkpoint selection across a completed-epoch training resume."""
import json
from pathlib import Path

import pytest

from oat.common.checkpoint_util import TopKCheckpointManager


FORMAT = 'ep-{epoch:04d}_mse-{test_reconst_mse:.6f}.ckpt'


def setup_manager(tmp_path, rows, *, mode='min', k=2, retained=None):
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    manager = TopKCheckpointManager(str(directory), 'test_reconst_mse',
                                    mode=mode, k=k, format_str=FORMAT)
    log = tmp_path / 'logs.json'
    log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    for row in rows if retained is None else retained:
        if 'test_reconst_mse' in row:
            (directory / FORMAT.format(**row)).write_bytes(b'checkpoint')
    return manager, log


@pytest.mark.parametrize('mode,expected,rejected,new_value', [
    ('min', [1, 2], 0.5, 0.05),
    ('max', [0, 3], 0.05, 0.5),
])
def test_restore_ranking_and_continued_selection_without_deleting_extras(
        tmp_path, mode, expected, rejected, new_value):
    rows = [{'epoch': i, 'test_reconst_mse': v}
            for i, v in enumerate([0.4, 0.1, 0.2, 0.3])]
    manager, log = setup_manager(tmp_path, rows, mode=mode)
    unrelated = Path(manager.save_dir) / 'latest.ckpt'
    unrelated.write_bytes(b'latest')
    before = set(Path(manager.save_dir).iterdir())
    assert manager.restore_from_logs(log, next_epoch=4) == 2
    assert set(manager.path_value_map) == {
        str(Path(manager.save_dir) / FORMAT.format(**rows[i])) for i in expected}
    assert set(Path(manager.save_dir).iterdir()) == before
    assert manager.get_ckpt_path({'epoch': 4, 'test_reconst_mse': rejected}) is None
    next_path = manager.get_ckpt_path({'epoch': 4, 'test_reconst_mse': new_value})
    assert next_path is not None
    assert len(manager.path_value_map) == 2
    assert unrelated.read_bytes() == b'latest'


def test_restore_uses_full_precision_and_only_retained_completed_epochs(tmp_path):
    rows = [
        {'epoch': 0, 'test_reconst_mse': 0.00010049},
        {'epoch': 1, 'test_reconst_mse': 0.00010041},
        {'epoch': 2, 'test_reconst_mse': 0.00000001},  # already pruned
        {'epoch': 3, 'test_reconst_mse': 0.00000002},  # uncompleted at resume point
        {'epoch': 4, 'train_loss': 1.0},
    ]
    manager, log = setup_manager(tmp_path, rows, k=1, retained=[rows[0], rows[1], rows[3]])
    assert manager.restore_from_logs(log, next_epoch=3) == 1
    selected = str(Path(manager.save_dir) / FORMAT.format(**rows[1]))
    assert manager.path_value_map == {selected: rows[1]['test_reconst_mse']}


def test_restore_ignores_bad_metrics_records_and_incomplete_tail(tmp_path):
    valid = {'epoch': 0, 'test_reconst_mse': 0.5}
    rows = [valid,
            {'epoch': 1, 'test_reconst_mse': float('nan')},
            {'epoch': 2, 'test_reconst_mse': float('inf')},
            {'epoch': 3, 'test_reconst_mse': True},
            {'epoch': 4, 'test_reconst_mse': None}]
    manager, log = setup_manager(tmp_path, rows, retained=[valid])
    tail = {'epoch': 5, 'test_reconst_mse': 0.01}
    (Path(manager.save_dir) / FORMAT.format(**tail)).touch()
    with log.open('a') as stream:
        stream.write('not-json\n[]\n')
        stream.write(json.dumps(tail))  # A record is complete only with its newline.
    assert manager.restore_from_logs(log, next_epoch=6) == 1
    assert list(manager.path_value_map.values()) == [0.5]


def test_restore_disabled_or_missing_log_leaves_files_untouched(tmp_path):
    row = {'epoch': 0, 'test_reconst_mse': 0.5}
    manager, log = setup_manager(tmp_path, [row], k=0)
    before = set(Path(manager.save_dir).iterdir())
    assert manager.restore_from_logs(log, next_epoch=1) == 0
    manager.k = 2
    assert manager.restore_from_logs(tmp_path / 'missing.json', next_epoch=1) == 0
    assert set(Path(manager.save_dir).iterdir()) == before
