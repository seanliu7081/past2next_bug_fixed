"""Exercise policy handoff without GPUs or actual optimizer steps."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recovery_run(tmp_path):
    dataset = tmp_path / "saved robot data.zarr"
    data = zarr.open_group(str(dataset), mode="w")
    data.create_dataset("meta/episode_ends", data=np.array([2, 4, 6, 8], dtype=np.int64))
    shapes = {
        "action": (7,), "agentview_rgb": (128, 128, 3),
        "robot0_eye_in_hand_rgb": (128, 128, 3), "robot0_eef_pos": (3,),
        "robot0_eef_rot6d": (6,), "robot0_gripper_qpos": (1,), "task_uid": (1,),
    }
    for key, shape in shapes.items():
        data.create_dataset(f"data/{key}", shape=(8, *shape),
                            dtype="uint8" if key.endswith("_rgb") else "float32")
    run = tmp_path / "existing run"
    (run / "tokenizer/.hydra").mkdir(parents=True)
    (run / "tokenizer/checkpoints").mkdir()
    cfg = {
        "training": {"num_epochs": 3, "val_every": 1, "sample_every": 1},
        "task": {"tokenizer": {"dataset": {"zarr_path": str(dataset)}}},
    }
    (run / "tokenizer/.hydra/config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "tokenizer/.hydra/hydra.yaml").write_text(
        yaml.safe_dump({"hydra": {"job": {"config_name": "oattok"}}}))
    records = [
        {"epoch": 0, "val_loss": .2, "test_reconst_mse": .0000121},
        {"epoch": 1, "val_loss": .1, "test_reconst_mse": .0000119},
        {"epoch": 2, "val_loss": .3, "test_reconst_mse": .0000140},
    ]
    (run / "tokenizer/logs.json").write_text("".join(json.dumps(r) + "\n" for r in records))
    for record in records:
        checkpoint = run / "tokenizer/checkpoints" / (
            "ep-{epoch:04d}_mse-{test_reconst_mse:.3f}.ckpt".format(**record))
        checkpoint.write_bytes(f"tokenizer epoch {record['epoch']}".encode())
    (run / "launcher.sh").write_text("original launcher\n")
    (run / "launch_commands.txt").write_text("original commands\n")
    return run, dataset


@pytest.fixture
def fake_python(tmp_path):
    """Run real dataset/config/selection code, mock only distributed train/check."""
    executable = tmp_path / "fake_python"
    executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
        import json
        import os
        from pathlib import Path
        import shutil
        import sys

        args = sys.argv[1:]
        if args[0] == '-':
            source = sys.stdin.read()
            if 'from scripts.check_real_robot_checkpoint import check_checkpoint' in source:
                run = Path(args[1])
                assert (run / 'policy').is_dir()
                (run / 'checkpoint_check.json').write_text('{"mock_check_called": true}')
            else:
                sys.argv = args
                exec(compile(source, '<launcher stdin>', 'exec'), {'__name__': '__main__'})
        elif args[:2] == ['-m', 'torch.distributed.run']:
            output = Path(json.loads(next(x.split('=', 1)[1] for x in args
                                          if x.startswith('hydra.run.dir='))))
            with Path(os.environ['MOCK_TRAIN_CALLS']).open('a') as stream:
                stream.write(json.dumps(args) + '\\n')
            if output.name == 'tokenizer':
                shutil.copytree(Path(os.environ['MOCK_SOURCE_RUN']) / 'tokenizer', output)
                # Reproduce editing the same inode while Bash waits for tokenizer.
                Path(os.environ['MOCK_MUTATE_LAUNCHER']).write_text(') invalid Bash after edit\\n')
            else:
                assert (output.parent / 'frozen_tokenizer.ckpt').read_bytes() == b'tokenizer epoch 1'
                output.mkdir()
        else:
            # Real Hydra resolution must still validate the requested recipe.
            os.execv(sys.executable, [sys.executable, *args])
    '''))
    executable.chmod(0o755)
    return executable


def launcher_env(tmp_path, fake_python):
    env = os.environ.copy()
    for key in ("TOKENIZER_CONFIG", "DATASET_PATH", "TOKENIZER_EPOCHS", "POLICY_EPOCHS", "RUN_DIR"):
        env.pop(key, None)
    env.update(TRAIN_PY=str(fake_python), WANDB_MODE="offline",
               MOCK_TRAIN_CALLS=str(tmp_path / "train_calls.jsonl"))
    return env


def invoke(args, env, *, script=ROOT / "train_pen_cabinet.sh"):
    return subprocess.run(["bash", str(script), *args], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=90)


def test_policy_recovery_selects_full_precision_best_and_preserves_records(
        tmp_path, recovery_run, fake_python):
    run, _ = recovery_run
    env = launcher_env(tmp_path, fake_python)
    result = invoke(["--policy-only", "--output-dir", str(run), "--gpus", "4,5"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "frozen_tokenizer.ckpt").read_bytes() == b"tokenizer epoch 1"
    selection = json.loads((run / "tokenizer_selection.json").read_text())
    assert selection["epoch"] == 1
    assert selection["metric"] == .0000119
    assert selection["sha256"] == hashlib.sha256(b"tokenizer epoch 1").hexdigest()
    assert (run / "launcher.sh").read_text() == "original launcher\n"
    assert (run / "launch_commands.txt").read_text() == "original commands\n"
    recovery = list(run.glob("policy_recovery.*"))
    assert len(recovery) == 1
    assert (recovery[0] / "launcher.sh").is_file()
    assert "--config-name=oattok" not in (recovery[0] / "launch_commands.txt").read_text()
    calls = [json.loads(line) for line in Path(env["MOCK_TRAIN_CALLS"]).read_text().splitlines()]
    assert len(calls) == 1
    assert "--nproc_per_node=2" in calls[0]
    assert "--config-name=train_past2next_scratch_all500" in calls[0]
    assert "training.num_epochs=1001" in calls[0]
    assert json.loads((run / "checkpoint_check.json").read_text())["mock_check_called"]
    result = invoke(["--policy-only", "--output-dir", str(run)], env)
    assert result.returncode != 0
    assert "Refusing existing policy output" in result.stderr
    assert len(Path(env["MOCK_TRAIN_CALLS"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("failure", ["incomplete", "unfinished_final_epoch", "dataset", "config"])
def test_recovery_rejects_incomplete_or_mismatched_tokenizer(
        tmp_path, recovery_run, fake_python, failure):
    run, dataset = recovery_run
    args = ["--policy-only", "--output-dir", str(run)]
    if failure == "incomplete":
        (run / "tokenizer/logs.json").write_text('{"epoch": 1, "val_loss": 0.1}\n')
    elif failure == "unfinished_final_epoch":
        (run / "tokenizer/logs.json").write_text('{"epoch": 2, "train_loss": 0.1}\n')
    elif failure == "dataset":
        args += ["--dataset", str(dataset.parent / "other.zarr")]
    else:
        args += ["--tokenizer-config", "train_oattok_so3aug"]
    env = launcher_env(tmp_path, fake_python)
    result = invoke(args, env)
    assert result.returncode != 0
    assert not Path(env["MOCK_TRAIN_CALLS"]).exists()
    assert not (run / "frozen_tokenizer.ckpt").exists()
    assert not list(run.glob("policy_recovery.*"))
    expected = {"incomplete": "Tokenizer is incomplete", "unfinished_final_epoch": "missing finite",
                "dataset": "Dataset does not match", "config": "Config does not match"}
    assert expected[failure] in result.stderr


def test_handoff_survives_in_place_launcher_edit_during_tokenizer(
        tmp_path, recovery_run, fake_python):
    source_run, dataset = recovery_run
    checkout = tmp_path / "isolated_checkout"
    checkout.mkdir()
    for directory in ("oat", "scripts"):
        (checkout / directory).symlink_to(ROOT / directory, target_is_directory=True)
    script = checkout / "train_pen_cabinet.sh"
    shutil.copy2(ROOT / script.name, script)
    run = tmp_path / "new two stage run"
    env = launcher_env(tmp_path, fake_python)
    env.update(TOKENIZER_EPOCHS="3", MOCK_MUTATE_LAUNCHER=str(script),
               MOCK_SOURCE_RUN=str(source_run))
    result = invoke(["--output-dir", str(run), "--dataset", str(dataset),
                     "--tokenizer-config", "oattok", "--gpus", "4,5"], env, script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert script.read_text().startswith(") invalid Bash")
    calls = [json.loads(line) for line in Path(env["MOCK_TRAIN_CALLS"]).read_text().splitlines()]
    assert len(calls) == 2
    assert "--config-name=oattok" in calls[0]
    assert "--config-name=train_past2next_scratch_all500" in calls[1]
    assert (run / "frozen_tokenizer.ckpt").read_bytes() == b"tokenizer epoch 1"
    assert json.loads((run / "tokenizer_completed.json").read_text()) == {"num_epochs": 3}
    assert json.loads((run / "checkpoint_check.json").read_text())["mock_check_called"]
