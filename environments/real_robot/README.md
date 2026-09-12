# real_robot environment

Created at `/venv/real_robot` from the working `/venv/oat` environment on this
instance, then rebound to this checkout:

```bash
/opt/miniforge3/bin/conda create -y -n real_robot --clone /venv/oat
git submodule update --init --recursive third_party/LIBERO
uv pip install --python /venv/real_robot/bin/python --no-deps \
  -e ./third_party/LIBERO -e .
/venv/real_robot/bin/python -m pip check
```

Activation:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate real_robot
```

Use `/venv/real_robot/bin/python` in managed jobs. The resolved stack includes
Python 3.10.21, PyTorch 2.10.0 (CUDA 12.9), torchvision 0.25.0, NumPy 2.2.6,
Zarr 2.18.3 and numcodecs 0.13.1. CUDA operations were checked on all eight
RTX 4090 GPUs. Both OAT and LIBERO editable imports point inside
`/workspace/ysk/past2next_bug_fixed`.

`conda-explicit.txt` records the Conda package builds; `requirements-freeze.txt`
records Python distributions, including packages supplied by Conda and local
editable paths. They are provenance records for this environment; installing
the entire pip freeze over the Conda export can replace Conda-managed binaries.

The repository's original `environment.yml` was left intact. Its W&B pin differs
from the project dependency requirement, and an unrestricted dependency resolve
could select Zarr 3, which the current data code does not support.
