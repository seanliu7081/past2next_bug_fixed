"""Run the sparse PRT tokenizer or policy using the current Python environment."""
import argparse
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"tokenizer": "PRT_Sparse_Tok", "policy": "PRT_Sparse_Policy"}


def choose_checkpoint(parser):
    if not sys.stdin.isatty():
        parser.error("policy training requires --ckpt FILE in a noninteractive shell")
    root = ROOT / "output"
    paths = sorted(p for pattern in ("*/*_PRT_Tok_*/checkpoints/*.ckpt",
                                     "*/*_PRT_Sparse_Tok_*/checkpoints/*.ckpt")
                   for p in root.glob(pattern))
    if not paths:
        parser.error("no tokenizer checkpoints found in output; pass --ckpt FILE")
    for i, path in enumerate(paths, 1):
        print(f"{i:3d}  {path.relative_to(ROOT)}")
    while True:
        try:
            number = input("Choose a tokenizer checkpoint by number: ").strip()
        except EOFError:
            parser.error("no tokenizer checkpoint selected")
        if number.isdecimal() and 1 <= int(number) <= len(paths):
            return paths[int(number) - 1]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Put Hydra overrides after --.")
    parser.add_argument("stage", choices=CONFIGS)
    parser.add_argument("--gpu_ids", default="0", help="comma-separated GPU IDs")
    parser.add_argument("--ckpt", type=Path, help="fitted PRT or sparse PRT tokenizer checkpoint")
    divider = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:divider])
    overrides = argv[divider + 1:]
    ids = args.gpu_ids.split(",")
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", args.gpu_ids) or len(set(map(int, ids))) != len(ids):
        parser.error("--gpu_ids requires distinct comma-separated GPU numbers")
    launch = [sys.executable, "-m", "accelerate.commands.launch", "--gpu_ids", args.gpu_ids,
              "--num_processes", str(len(ids)), "--num_machines", "1"]
    if len(ids) > 1:
        launch.extend(["--multi_gpu", "--main_process_port", "0"])
    launch.extend(["scripts/run_workspace.py", f"--config-name={CONFIGS[args.stage]}"])
    if args.stage == "policy":
        ckpt = (args.ckpt or choose_checkpoint(parser)).expanduser().resolve()
        if not ckpt.is_file():
            parser.error(f"tokenizer checkpoint does not exist: {ckpt}")
        os.environ["PRT_SPARSE_TOKENIZER_CKPT"] = str(ckpt)
        launch.append("policy.action_tokenizer.checkpoint=${oc.env:PRT_SPARSE_TOKENIZER_CKPT}")
        os.environ.setdefault("MUJOCO_GL", "egl")
        print(f"Stage 1 checkpoint: {ckpt}", flush=True)
    elif args.ckpt is not None:
        parser.error("--ckpt selects the frozen tokenizer for the policy stage")
    os.chdir(ROOT)
    os.execv(sys.executable, launch + overrides)


if __name__ == "__main__":
    main(sys.argv[1:])
