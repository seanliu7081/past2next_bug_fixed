#!/usr/bin/env python
"""Paired Pi05-past weight-edit evaluation on the frozen LIBERO-10 t08 scene.

The policy uses its training evaluator's RGB/state/action-history interface. A
separate simulator uses the exact Python environment and LIBERO checkout that
authored the frozen scene. Only RGB, robot state and task UID reach the policy;
object poses and obstacle contacts are evaluator records. No optimizer is used.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import selectors
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_LAYOUT = Path('/workspace/gpt_field/output/layouts/libero_10_t8_bottle_obstacle_20261103_001')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def array_pack(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return {'dtype': value.dtype.str, 'shape': list(value.shape),
            'base64': base64.b64encode(value.tobytes()).decode('ascii')}


def array_unpack(value):
    import numpy as np
    dtype = np.dtype(value['dtype'])
    if dtype.hasobject:
        raise ValueError('Object arrays are forbidden on the simulator interface')
    return np.frombuffer(base64.b64decode(value['base64'], validate=True), dtype=dtype).reshape(value['shape']).copy()


def policy_observation(raw, history, policy):
    """Same byte RGB, vertical flip, float state, temporal padding as LiberoRunner."""
    import numpy as np
    import torch
    from oat.gymnasium_util.multistep_wrapper import stack_last_n_obs
    current = {}
    for key, spec in policy.shape_meta['obs'].items():
        if spec.get('type') == 'rgb':
            current[key] = np.flip(raw[key.removesuffix('_rgb') + '_image'], axis=0).astype(np.uint8)
        elif key == 'task_uid':
            current[key] = np.array([38], dtype=np.uint8)
        else:
            current[key] = np.asarray(raw[key], dtype=np.float32)
    history.append(current)
    result = {}
    for key in policy.get_observation_ports():
        value = stack_last_n_obs([item[key] for item in history], policy.n_obs_steps)[None]
        dtype = torch.uint8 if policy.shape_meta['obs'][key].get('type') == 'rgb' else policy.dtype
        result[key] = torch.from_numpy(value.copy()).to(device=policy.device, dtype=dtype)
    return result


class ObstacleMonitor:
    """Record obstacle/robot or obstacle/pot penetration at each physics step."""
    def __init__(self, env, path):
        self.sim = env.sim
        self.original_step = self.sim.step
        self.stream = Path(path).open('w', buffering=1)
        self.action_index = -1
        self.substep = 0
        self.physics_substeps = 0
        self.inspection_calls = 0
        self.counts = Counter()
        self.contacted_actions = set()
        self.first = None
        self.sim.step = self._step

    def inspect(self, phase):
        self.inspection_calls += 1
        contacts = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.dist > 0:
                continue
            names = [self.sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
            obstacle = [name.startswith('ketchup_obstacle_1_') for name in names]
            if sum(obstacle) != 1:
                continue
            other = names[1 if obstacle[0] else 0]
            category = ('robot_obstacle' if other.startswith(('robot0_', 'gripper0_')) else
                        'pot_obstacle' if other.startswith(('moka_pot_1_', 'moka_pot_2_')) else None)
            if category is None:
                continue
            item = {'category': category, 'geoms': names, 'distance_m': float(contact.dist),
                    'position_m': contact.pos.tolist()}
            contacts.append(item)
            self.counts[category] += 1
        if contacts:
            event = {'action_index': self.action_index, 'physics_substep': self.substep,
                     'phase': phase, 'contacts': contacts}
            self.stream.write(json.dumps(event) + '\n')
            self.contacted_actions.add(self.action_index)
            if self.first is None:
                self.first = event

    def _step(self, *args, **kwargs):
        result = self.original_step(*args, **kwargs)
        self.substep += 1
        self.physics_substeps += 1
        self.inspect('physics_substep')
        return result

    def snapshot(self):
        return {'obstacle_collision': self.first is not None, 'contact_point_samples': sum(self.counts.values()),
                'counts': dict(self.counts), 'contacted_action_indices': sorted(self.contacted_actions),
                'physics_substeps': self.physics_substeps, 'inspection_calls': self.inspection_calls,
                'first_event': self.first,
                'scope': 'ketchup obstacle versus robot/gripper or either moka pot; table support excluded',
                'count_unit': 'contact-point samples, not independent impacts', 'policy': 'record_only'}

    def close(self):
        self.sim.step = self.original_step
        self.stream.close()


def evaluator_state(env, raw):
    import numpy as np
    model, data = env.sim.model, env.sim.data
    objects = {}
    for name, obj in env.env.objects_dict.items():
        body_id = model.body_name2id(obj.root_body)
        item = {'body_xyz_m': data.body_xpos[body_id].tolist(),
                'body_quat_wxyz': data.body_xquat[body_id].tolist()}
        if obj.joints:
            joint = obj.joints[0]
            item.update(joint=joint, qpos=np.atleast_1d(data.get_joint_qpos(joint)).tolist(),
                        qvel=np.atleast_1d(data.get_joint_qvel(joint)).tolist())
        objects[name] = item
    stove_id = model.body_name2id(env.env.fixtures_dict['flat_stove_1'].root_body)
    return {'eef_pos': np.asarray(raw['robot0_eef_pos']).tolist(),
            'eef_quat_xyzw': np.asarray(raw['robot0_eef_quat']).tolist(),
            'gripper_qpos': np.asarray(raw['robot0_gripper_qpos']).tolist(),
            'objects': objects, 'stove_body_xyz_m': data.body_xpos[stove_id].tolist(),
            'stove_button_qpos': float(data.get_joint_qpos('flat_stove_1_button'))}


def validate_reset(env, state, manifest):
    import numpy as np
    checks = {}
    for mapping in manifest['mapped_joints']:
        for kind in ('qpos', 'qvel'):
            address = getattr(env.sim.model, 'get_joint_' + kind + '_addr')(mapping['joint'])
            actual = list(address) if isinstance(address, tuple) else [int(address), int(address) + 1]
            checks[mapping['joint'] + '_' + kind + '_address'] = actual == mapping['target_' + kind + '_address']
    errors = {}
    for name, reference in manifest['custom_initial_settled_object_poses'].items():
        delta = float(np.max(np.abs(np.asarray(state['objects'][name]['qpos']) - reference['qpos'])))
        errors[name + '_qpos_max_abs'] = delta
        checks[name + '_settled_qpos'] = delta <= 1e-7
    errors['stove_position_m'] = float(np.linalg.norm(np.asarray(state['stove_body_xyz_m']) -
                                                       manifest['custom_initial_fixture']['body_xyz_m']))
    checks['stove_position'] = errors['stove_position_m'] <= 1e-10
    checks['stove_button'] = abs(state['stove_button_qpos'] - manifest['custom_initial_fixture']['button_qpos']) <= 1e-7
    return {'passed': all(checks.values()), 'checks': checks, 'errors': errors,
            'image_hash_note': 'Pi05 uses 128px vertical-flipped RGB; the 512px UserCheck observation hash is not comparable.'}


def simulator_worker(config):
    """JSON transport runs in the frozen scene author's simulator environment."""
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import numpy as np
    import torch
    root = Path(config['libero_root'])
    benchmark_root = root / 'libero' / 'libero'
    config_dir = Path(config['output_dir']) / 'simulator_config'
    config_dir.mkdir()
    paths = {'benchmark_root': str(benchmark_root), 'bddl_files': str(benchmark_root / 'bddl_files'),
             'init_states': str(benchmark_root / 'init_files'), 'assets': str(benchmark_root / 'assets'),
             'datasets': str(root / 'libero' / 'datasets')}
    write_json(config_dir / 'config.yaml', paths)
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    sys.path.insert(0, str(root))
    from libero.libero.envs import OffScreenRenderEnv
    env = monitor = None
    video = None
    try:
        random.seed(config['env_seed']); np.random.seed(config['env_seed']); torch.manual_seed(config['env_seed'])
        env = OffScreenRenderEnv(bddl_file_name=config['bddl'], camera_heights=128, camera_widths=128,
                                 control_freq=20, horizon=float('inf'), ignore_done=True,
                                 controller='OSC_POSE', robots=['Panda'])
        env.seed(config['env_seed'])
        env.reset()
        initial = np.load(config['initial_state'], allow_pickle=False)
        if initial.shape != np.asarray(env.sim.get_state().flatten()).shape:
            raise ValueError('Frozen state shape does not match simulator model')
        raw = env.set_init_state(initial)
        for _ in range(config['settle_steps']):
            raw, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        state = evaluator_state(env, raw)
        manifest = json.loads(Path(config['layout_manifest']).read_text())
        validation = validate_reset(env, state, manifest)
        if not validation['passed']:
            raise RuntimeError('Frozen scene reset differs from authoring manifest: ' + json.dumps(validation))
        output = Path(config['output_dir'])
        np.save(output / 'post_reset_sim_state.npy', env.sim.get_state().flatten())
        np.savez_compressed(output / 'initial_raw_observation.npz', **raw)
        from PIL import Image
        for camera, label in [('agentview', 'external'), ('robot0_eye_in_hand', 'wrist')]:
            Image.fromarray(np.flip(raw[camera + '_image'], axis=0)).save(output / ('initial_' + label + '.png'))
        if not config['no_video']:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(output / 'episode.mp4'), fps=20, codec='libx264',
                                       quality=7, macro_block_size=1)
        monitor = ObstacleMonitor(env, output / 'obstacle_contacts.jsonl')
        monitor.inspect('initial_state')
        action_index = 0
        video_seconds = 0.

        def response():
            sensors = {key: array_pack(raw[key]) for key in
                       ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
                        'robot0_eef_quat', 'robot0_gripper_qpos')}
            return {'observation': sensors, 'evaluator_state': evaluator_state(env, raw),
                    'raw_success': bool(env.check_success()), 'collision': monitor.snapshot(),
                    'video_render_seconds': video_seconds}

        def record_frame():
            nonlocal video_seconds
            if video is None:
                return
            started = time.monotonic()
            views = [np.flip(env.sim.render(256, 256, camera_name=name), axis=0)
                     for name in ('agentview', 'robot0_eye_in_hand')]
            video.append_data(np.concatenate(views, axis=1))
            video_seconds += time.monotonic() - started

        record_frame()
        hello = response()
        hello['reset_validation'] = validation
        hello['simulator_metadata'] = {
            'python': sys.executable, 'libero_root': str(root),
            'packages': {name: importlib.metadata.version(name) for name in ['numpy','robosuite','mujoco','torch']},
            'joint_names': list(env.sim.model.joint_names),
            'controller_output_max': env.robots[0].controller.output_max.tolist(),
            'environment_source': str(sys.modules['libero.libero.envs.env_wrapper'].__file__),
            'model_xml_sha256': hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()}
        protocol.write(json.dumps({'ok': True, 'result': hello}) + '\n')
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            if request['op'] != 'step':
                raise ValueError('Unknown simulator operation')
            monitor.action_index = action_index; monitor.substep = 0
            action = np.asarray(request['action'], dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Action must contain seven finite values')
            raw, _, _, _ = env.step(action)
            env.sim.forward()
            monitor.inspect('control_endpoint')
            action_index += 1
            record_frame()
            protocol.write(json.dumps({'ok': True, 'result': response()}) + '\n')
    except Exception:
        protocol.write(json.dumps({'ok': False, 'error': traceback.format_exc()}) + '\n')
        raise
    finally:
        if video is not None:
            video.close()
        if monitor is not None:
            monitor.close()
        if env is not None:
            env.close()


class Simulator:
    def __init__(self, config, renderer):
        self.log = (Path(config['output_dir']) / 'simulator.log').open('w')
        env = dict(os.environ, MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES=str(renderer['egl_device_id']),
                   MUJOCO_EGL_DEVICE_ID=str(renderer['egl_device_id']))
        self.process = subprocess.Popen([config['sim_python'], '-u', str(Path(__file__).resolve()),
                                         '--sim-worker', json.dumps(config)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)
        try:
            self.initial = self.read()
        except Exception:
            self.close()
            raise

    def read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=180):
                raise TimeoutError('Simulator response timed out; inspect simulator.log')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Simulator exited; inspect simulator.log')
        result = json.loads(line)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result['result']

    def step(self, action):
        self.process.stdin.write(json.dumps({'op': 'step', 'action': action.tolist()}) + '\n')
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n'); self.process.stdin.flush()
                self.process.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        self.log.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--layout-dir', type=Path, default=DEFAULT_LAYOUT)
    p.add_argument('--weights', choices=['model','ema'], default='model')
    p.add_argument('--policy-seed', type=int, default=42)
    p.add_argument('--env-seed', type=int, default=7)
    p.add_argument('--settle-steps', type=int, default=10)
    p.add_argument('--max-episode-steps', type=int, default=550)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--patch-json', type=Path)
    p.add_argument('--capture-head', action='store_true')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Load checkpoint and write provenance, without simulator')
    p.add_argument('--sim-python', default='/workspace/venvs/libero-plus/bin/python')
    p.add_argument('--libero-root', default='/workspace/heading_flow_pub/third_party/LIBERO')
    p.add_argument('--pi05', type=Path)
    p.add_argument('--spm', type=Path)
    return p


def main(args):
    import numpy as np
    import torch
    from scripts.evaluate_p2n_vla import read_payload, load_policy
    from oat.env_runner.p2n_vla_rollout import resolve_renderer
    from oat.model.vla.weight_space_edit import OutputHeadCapture, apply_output_head_patch, load_patch_spec
    if args.max_episode_steps < 1 or args.settle_steps != 10 or args.env_seed != 7:
        raise ValueError('Frozen t08 experiment requires env seed 7, 10 settling steps and positive step cap')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'; inputs.mkdir()
    for name in ['custom_task.bddl','initial_state.npy','layout_manifest.json']:
        shutil.copy2(args.layout_dir / name, inputs / name)
    shutil.copy2(__file__, output / 'evaluator_source.py')
    torch.set_num_threads(args.threads)
    random.seed(args.policy_seed); np.random.seed(args.policy_seed); torch.manual_seed(args.policy_seed)
    payload = read_payload(args.snapshot)
    if payload['policy_config']['_target_'] != 'oat.policy.pi05_past.Pi05PastPolicy':
        raise ValueError('This experiment requires Pi05PastPolicy')
    policy = load_policy(args, payload)
    if policy.n_obs_steps != 1:
        raise ValueError('This standalone evaluator currently requires the saved one-observation Pi05-past contract')
    patch = None
    if args.patch_json is not None:
        spec = load_patch_spec(args.patch_json)
        patch = apply_output_head_patch(policy, spec)
        write_json(output / 'weight_edit.json', patch.provenance)
        shutil.copy2(args.patch_json, inputs / 'patch.json')
    renderer = resolve_renderer(args.device)
    if renderer is None and not args.dry_run:
        raise ValueError('A CUDA policy and EGL renderer are required')
    manifest = json.loads((inputs / 'layout_manifest.json').read_text())
    metadata = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'snapshot': str(args.snapshot.resolve()),
                'snapshot_sha256': sha256(args.snapshot), 'checkpoint_metadata': payload.get('metadata'),
                'weights': args.weights, 'policy_seed': args.policy_seed, 'env_seed': args.env_seed,
                'task_index': 8, 'task_uid': 38, 'task_language': manifest['prompt'],
                'layout': 'frozen custom bottle obstacle, single layout', 'settle_steps': args.settle_steps,
                'max_policy_steps': args.max_episode_steps, 'renderer': renderer,
                'policy_python': sys.executable, 'policy_image_size': 128,
                'image_transform': 'raw LIBERO image vertical flip, uint8; policy native preprocessing thereafter',
                'controller': 'stock OSC_POSE, XYZ 0.05m, rotation0.5rad; no user-check 0.10m override',
                'execution_history': 'record_executed_actions after each actually executed chunk prefix',
                'input_files': {name: sha256(inputs / name) for name in ['custom_task.bddl','initial_state.npy','layout_manifest.json']},
                'evaluator_sha256': sha256(__file__), 'patch': None if patch is None else patch.provenance,
                'metrics': 'raw LIBERO success and obstacle-collision-free success reported separately',
                'no_training': True, 'no_runtime_action_correction': True}
    write_json(output / 'metadata.json', metadata)
    if args.dry_run:
        write_json(output / 'summary.json', {'status': 'dry_run', 'no_rollout': True})
        return
    config = dict(output_dir=str(output), libero_root=args.libero_root, sim_python=args.sim_python,
                  bddl=str(inputs / 'custom_task.bddl'), initial_state=str(inputs / 'initial_state.npy'),
                  layout_manifest=str(inputs / 'layout_manifest.json'), env_seed=args.env_seed,
                  settle_steps=args.settle_steps, no_video=args.no_video)
    simulator = capture = None
    trace = None
    started = time.monotonic()
    all_actions = []
    chunk_starts = []
    completed_steps = 0
    summary = {'status': 'error', 'raw_success': False, 'steps': 0}
    try:
        simulator = Simulator(config, renderer)
        current = simulator.initial
        write_json(output / 'initial_evaluator_state.json', current['evaluator_state'])
        write_json(output / 'reset_validation.json', current['reset_validation'])
        write_json(output / 'simulator_metadata.json', current['simulator_metadata'])
        if current['raw_success']:
            raise RuntimeError('Frozen task must be incomplete at reset')
        # Policy RNG starts identically in every comparison, independent of setup.
        random.seed(args.policy_seed); np.random.seed(args.policy_seed); torch.manual_seed(args.policy_seed)
        policy.reset()
        history = deque(maxlen=policy.n_obs_steps)
        if args.capture_head:
            capture = OutputHeadCapture(policy)
            (output / 'head_captures').mkdir()
        trace = (output / 'trajectory.jsonl').open('w', buffering=1)
        policy_seconds = 0.
        chunk_index = 0
        while completed_steps < args.max_episode_steps and not current['raw_success']:
            raw = {key: array_unpack(value) for key,value in current['observation'].items()}
            observation = policy_observation(raw, history, policy)
            state_before = current['evaluator_state']
            if capture is not None:
                capture.clear()
                # Save the policy-visible inputs and inference RNG before sampling,
                # so offline candidate comparisons can replay exactly this context.
                replay = {'obs__' + key: value.detach().cpu().numpy().copy()
                          for key, value in observation.items()}
                past = policy._past_buffer
                valid = policy._past_valid_buffer
                replay.update(history_present=np.array(past is not None),
                              history_actions=(np.zeros((1, policy.past_n, policy.action_dim), np.float32)
                                               if past is None else past.detach().cpu().numpy().copy()),
                              history_valid=(np.zeros((1, policy.past_n), bool)
                                             if valid is None else valid.detach().cpu().numpy().copy()),
                              torch_rng_cpu=torch.get_rng_state().numpy().copy(),
                              torch_rng_cuda=torch.cuda.get_rng_state(policy.device).cpu().numpy().copy())
            tick = time.monotonic()
            with torch.inference_mode():
                prediction = policy.predict_action(observation)
            torch.cuda.synchronize(policy.device)
            policy_seconds += time.monotonic() - tick
            actions = prediction['action'].detach().cpu().numpy()
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError('Policy returned invalid actions')
            chunk_starts.append(completed_steps)
            if capture is not None:
                data = capture.pop()
                np.savez_compressed(output / 'head_captures' / f'chunk_{chunk_index:04d}.npz',
                                    **data, **replay, actions=actions,
                                    action_pred=prediction['action_pred'].detach().cpu().numpy(),
                                    policy_step=np.array(completed_steps), eef_pos=np.asarray(state_before['eef_pos']))
            executed = 0
            for action in actions[0]:
                if completed_steps >= args.max_episode_steps or current['raw_success']:
                    break
                before = current['evaluator_state']
                current = simulator.step(action)
                record = {'step': completed_steps, 'chunk': chunk_index, 'index_in_chunk': executed,
                          'action': action.tolist(), 'before': before, 'after': current['evaluator_state'],
                          'raw_success': current['raw_success'], 'collision': current['collision']}
                trace.write(json.dumps(record) + '\n')
                all_actions.append(action.copy()); completed_steps += 1; executed += 1
            policy.record_executed_actions(actions, executed_lengths=np.array([executed], dtype=np.int64))
            chunk_index += 1
            print(f'chunk={chunk_index} steps={completed_steps} success={current["raw_success"]} '
                  f'obstacle_contacts={current["collision"]["contact_point_samples"]}', flush=True)
            write_json(output / 'progress.json', {'steps': completed_steps, 'chunks': chunk_index,
                                                 'raw_success': current['raw_success'],
                                                 'collision': current['collision']})
        collision = current['collision']
        summary = {'status': 'complete', 'termination': 'success' if current['raw_success'] else 'timeout',
                   'raw_success': current['raw_success'], 'obstacle_collision_free_success':
                   current['raw_success'] and not collision['obstacle_collision'],
                   'steps': completed_steps, 'chunks': chunk_index, 'policy_seed': args.policy_seed,
                   'max_policy_steps': args.max_episode_steps, 'collision': collision,
                   'policy_inference_seconds': policy_seconds, 'video_render_seconds': current['video_render_seconds'],
                   'final_evaluator_state': current['evaluator_state'],
                   'claim_scope': 'one policy RNG seed on one frozen custom layout; not the official LIBERO benchmark'}
    except Exception:
        summary.update(error=traceback.format_exc(), steps=completed_steps)
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'summary.json', summary)
        if trace is not None:
            trace.close()
        np.savez_compressed(output / 'commands.npz', actions=np.asarray(all_actions), chunk_starts=np.asarray(chunk_starts))
        if capture is not None:
            capture.close()
        if patch is not None:
            patch.restore()
        if simulator is not None:
            simulator.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--sim-worker':
        simulator_worker(json.loads(sys.argv[2]))
    else:
        main(parser().parse_args())
