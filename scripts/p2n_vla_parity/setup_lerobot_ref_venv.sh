#!/usr/bin/env bash
# Recreate the throwaway LeRobot PI0.5 reference venv used by dump_pi05_reference.py.
#
# It never touches /venv/oat. LeRobot main needs Python >= 3.12 and transformers >= 5.4, < 5.6,
# while /venv/oat stays on transformers 5.2.
#
# Usage, from the repository root:
#   bash scripts/p2n_vla_parity/setup_lerobot_ref_venv.sh [VENV_DIR]     # default /venv/lerobot_ref
#
# Recorded environment of the reference dump (full list: output/parity/lerobot_ref_freeze.txt):
# - torch 2.11.0+cu128 and torchvision 0.26.0+cu128;
# - lerobot 0.6.2 at commit 2577da0e;
# - transformers 5.5.4, sentencepiece 0.2.2, safetensors 0.8.0, numpy 2.2.6.
set -euo pipefail

VENV="${1:-/venv/lerobot_ref}"
LEROBOT_SHA="2577da0ef39b47f870592d62d81edc8bce922cc4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$(readlink -f "$VENV")" in
  /venv/oat|/venv/oat/*) echo "refusing to touch /venv/oat" >&2; exit 2 ;;
esac
if [[ -e "$VENV" ]]; then
  echo "$VENV already exists; remove it first or pass another directory" >&2
  exit 2
fi

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "torch==2.11.0" "torchvision==0.26.0" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" \
  "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}" \
  "transformers==5.5.4" "sentencepiece==0.2.2" "safetensors==0.8.0"
"$VENV/bin/python" "$REPO_ROOT/scripts/p2n_vla_parity/dump_pi05_reference.py" --check-env
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md, json, torch, transformers
commit = json.loads(md.distribution("lerobot").read_text("direct_url.json"))["vcs_info"]["commit_id"]
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}, lerobot {commit}")
assert commit == "2577da0ef39b47f870592d62d81edc8bce922cc4", commit
assert torch.__version__.startswith("2.11.0") and transformers.__version__ == "5.5.4"
EOF
