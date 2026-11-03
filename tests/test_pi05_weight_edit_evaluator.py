"""Policy boundary and evaluator-only contact checks, without a model or simulator."""
from collections import deque
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_pi05_weight_edit import (
    ObstacleMonitor, array_pack, array_unpack, policy_observation, validate_reset,
)


def test_array_transport_preserves_bytes_and_rejects_objects():
    x = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[::-1]
    np.testing.assert_array_equal(array_unpack(array_pack(x)), x)
    with pytest.raises(ValueError, match='Object'):
        array_unpack({'dtype': '|O', 'shape': [1], 'base64': ''})


def test_policy_boundary_vertical_flip_byte_rgb_and_no_object_truth():
    shape = {'agentview_rgb': {'type': 'rgb'}, 'robot0_eef_pos': {'type': 'state'},
             'task_uid': {'type': 'state'}}
    policy = SimpleNamespace(shape_meta={'obs': shape}, n_obs_steps=1,
                             dtype=torch.float32, device='cpu', get_observation_ports=lambda: list(shape))
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw = {'agentview_image': image, 'robot0_eef_pos': np.array([1., 2., 3.]),
           'hidden_obstacle_pose': np.array([99., 99., 99.])}
    result = policy_observation(raw, deque(maxlen=1), policy)
    assert set(result) == set(shape)
    assert result['agentview_rgb'].dtype == torch.uint8
    np.testing.assert_array_equal(result['agentview_rgb'][0, 0], image[::-1])
    assert result['robot0_eef_pos'].shape == (1, 1, 3)
    assert result['task_uid'].item() == 38


class FakeSim:
    def __init__(self):
        self.names = ['ketchup_obstacle_1_g0', 'gripper0_finger1_collision',
                      'moka_pot_1_g3', 'table_collision']
        self.model = SimpleNamespace(geom_id2name=lambda i: self.names[i])
        self.data = SimpleNamespace(ncon=0, contact=[])
        self.calls = 0

    def step(self):
        self.calls += 1

    def contacts(self, others, distance=-0.001):
        self.data.contact = [SimpleNamespace(geom1=0, geom2=i, dist=distance, pos=np.zeros(3)) for i in others]
        self.data.ncon = len(others)


def test_monitor_counts_substeps_and_endpoint_excludes_table_and_proximity(tmp_path):
    sim = FakeSim()
    original = sim.step
    monitor = ObstacleMonitor(SimpleNamespace(sim=sim), tmp_path / 'contacts.jsonl')
    monitor.action_index = 2
    sim.contacts([1, 2, 3]); sim.step()
    sim.contacts([1], distance=0.001); sim.step()
    sim.contacts([2]); monitor.inspect('control_endpoint')
    assert sim.calls == 2
    record = monitor.snapshot()
    assert record['counts'] == {'robot_obstacle': 1, 'pot_obstacle': 2}
    assert record['contacted_action_indices'] == [2]
    assert record['physics_substeps'] == 2
    assert record['inspection_calls'] == 3
    assert record['first_event']['physics_substep'] == 1
    monitor.close()
    assert sim.step == original
    events = [json.loads(line) for line in (tmp_path / 'contacts.jsonl').read_text().splitlines()]
    assert len(events) == 2


def test_reset_validation_detects_joint_mapping_and_physical_drift():
    model = SimpleNamespace(get_joint_qpos_addr=lambda name: (0, 7),
                            get_joint_qvel_addr=lambda name: (0, 6))
    env = SimpleNamespace(sim=SimpleNamespace(model=model))
    manifest = {'mapped_joints': [{'joint': 'pot', 'target_qpos_address': [0,7], 'target_qvel_address': [0,6]}],
                'custom_initial_settled_object_poses': {'pot': {'qpos': [1.,2.,3.]}},
                'custom_initial_fixture': {'body_xyz_m': [0.,0.,0.], 'button_qpos': 1.4}}
    state = {'objects': {'pot': {'qpos': [1.,2.,3.]}}, 'stove_body_xyz_m': [0.,0.,0.], 'stove_button_qpos': 1.4}
    assert validate_reset(env, state, manifest)['passed']
    state['objects']['pot']['qpos'][0] += 0.001
    assert not validate_reset(env, state, manifest)['passed']
