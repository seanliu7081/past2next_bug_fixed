"""
Usage:
python scripts/run_workspace.py --config-name=train_oattok_so3aug
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pathlib
from oat.workspace.base_workspace import BaseWorkspace
from oat.common.hydra_util import register_new_resolvers
from oat.common.policy_handoff import maybe_wait_for_policy_handoff

register_new_resolvers()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'oat','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    maybe_wait_for_policy_handoff(cfg, HydraConfig.get().runtime.output_dir)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
