from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scripts import evaluate_p2n_new as evaluation


def arguments(tmp_path, *extra):
    checkpoint = tmp_path / 'policy.ckpt'
    checkpoint.touch()
    return evaluation.parser().parse_args([
        '--checkpoint', str(checkpoint), '--output-dir', str(tmp_path / 'evaluation'), *extra,
    ])


def test_requested_command_has_500_episodes_ema_and_online_logging(tmp_path):
    args = arguments(tmp_path, '--gpu', '2', '--n-test', '500', '--wandb-mode', 'online')
    evaluation.validate_args(args)
    assert (args.n_test, args.n_parallel_envs, args.weights, args.protocol) == (500, 10, 'ema', 'corrected')
    assert args.wandb_mode == 'online'


def test_existing_results_are_never_replaced(tmp_path):
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        evaluation.validate_args(args)


@pytest.mark.parametrize('variant,runner_name', [
    ('p2n_new', 'P2NNewLiberoRunner'),
    ('p2n_state_gate_new', 'P2NStateGateNewLiberoRunner'),
])
def test_runner_preserves_history_and_uses_checkpoint_action_stride(tmp_path, variant, runner_name):
    args = arguments(tmp_path, '--n-test', '500', '--n-parallel-envs', '10')
    cfg = OmegaConf.create({'task': {'policy': {'env_runner': {
        '_target_': 'oat.env_runner.p2n_new_runner.' + runner_name,
        'task_name': 'libero10', 'n_test': 20, 'n_test_vis': 3,
        'n_action_steps': 99, 'n_obs_steps': 9,
        'state_history_keys': ['robot0_eef_pos'], 'state_history_steps': 8,
    }}}})
    policy = SimpleNamespace(variant=variant, n_action_steps=8, n_obs_steps=1)
    runner = evaluation.runner_config_for(args, cfg, policy)
    assert runner['_target_'].endswith('.' + runner_name)
    assert runner['n_action_steps'] == 8 and runner['n_obs_steps'] == 1
    assert runner['state_history_keys'] == ['robot0_eef_pos']
    assert runner['state_history_steps'] == 8
    assert runner['n_test'] == 500 and runner['n_test_vis'] == 0
    assert runner['episode_records_path'].endswith('/episodes.jsonl')
    assert cfg.task.policy.env_runner.n_test == 20


def test_dispatch_uses_embedded_modern_loader_without_fresh_asset_construction(tmp_path, monkeypatch):
    from oat.policy.p2n_new_common import P2NNewCommonPolicy
    import hydra.utils

    calls = []

    class Policy(P2NNewCommonPolicy):
        @classmethod
        def from_checkpoint(cls, path, **kwargs):
            calls.append((path, kwargs))
            return 'restored_policy', 'restored_configuration'

    def get_class(target):
        assert target == 'embedded.ModernPolicy'
        return Policy

    monkeypatch.setattr(hydra.utils, 'get_class', get_class)
    args = arguments(tmp_path)
    payload = {'policy_config': {'_target_': 'embedded.ModernPolicy'},
               'cfg': {'policy': {'_target_': 'obsolete.FreshPolicy'}}}
    assert evaluation.load_policy(args, payload) == ('restored_policy', 'restored_configuration')
    assert calls == [(str(args.checkpoint), {'return_configuration': True, 'weights': 'ema'})]


def test_500_episode_schedule_covers_every_task_equally():
    from oat.env.libero.factory import get_subtasks
    from oat.env_runner.libero_runner import build_episode_schedule
    from collections import Counter

    tasks = get_subtasks('libero10')
    for protocol in ('corrected', 'official'):
        schedule = build_episode_schedule(tasks, 500, 10, 1000, protocol, 0)
        assert len(schedule) == 500
        assert Counter(item['task_name'] for item in schedule) == {task: 50 for task in tasks}
