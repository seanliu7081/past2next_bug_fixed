#!/usr/bin/env bash
# Add MimicGen to an isolated environment that inherits the PRT training stack.
set -euo pipefail

base_python=${PRT_BASE_PYTHON:-/venv/oat/bin/python}
venv_dir=${MIMICGEN_VENV:-/workspace/venvs/mimicgen-prt}
python="$venv_dir/bin/python"

if [[ ! -f $venv_dir/pyvenv.cfg ]]; then
    "$base_python" -m venv --system-site-packages "$venv_dir"
fi
"$python" -c 'import sys, torch; assert sys.prefix != sys.base_prefix, "Expected an isolated venv"'

# MIMICGEN_SOURCE_DIR can point to an existing pair of Git clones (e.g. fm_dno/third_party).
# Each checkout below is private to this environment, including the simulator assets.
checkout() {
    local name=$1 url=$2 revision=$3 path="$venv_dir/src/$1"
    local source=$url
    if [[ -n ${MIMICGEN_SOURCE_DIR:-} && -d $MIMICGEN_SOURCE_DIR/$name/.git ]]; then
        source="$MIMICGEN_SOURCE_DIR/$name"
    fi
    if [[ ! -d $path/.git ]]; then
        git clone --no-checkout "$source" "$path"
        git -C "$path" checkout --detach "$revision"
    fi
    if [[ $(git -C "$path" rev-parse HEAD) != "$revision" ]]; then
        echo "Unexpected revision in $path (expected $revision)" >&2
        return 1
    fi
}

mkdir -p "$venv_dir/src"
checkout mimicgen https://github.com/NVlabs/mimicgen.git 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
checkout robosuite-task-zoo https://github.com/ARISE-Initiative/robosuite-task-zoo.git 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed

"$python" -m pip install --no-deps robosuite==1.4.1 mujoco==3.2.6 chardet==5.2.0
# Expose the full pinned source trees, including their XML, meshes and textures.
"$python" - "$venv_dir/src" <<'PY'
from pathlib import Path
import sys
import sysconfig

src = Path(sys.argv[1]).resolve()
package_paths = (src / 'mimicgen', src / 'robosuite-task-zoo')
Path(sysconfig.get_path('purelib'), 'mimicgen_sources.pth').write_text(
    ''.join(f'{path}\n' for path in package_paths)
)
PY
printf 'MimicGen training Python: %s\n' "$python"
