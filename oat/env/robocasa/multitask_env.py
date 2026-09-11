"""RoboCasa env that injects a ``task_uid`` observation for multi-task eval.

This is the RoboCasa analogue of how ``oat.env.libero.env.LiberoEnv`` exposes
``task_uid``: a single subclass that adds the canonical integer task id (from
``oat.env.robocasa.factory``) to both the observation space and every emitted
observation, so a multi-task policy can condition on which task it is solving.

Kept as a separate file from the single-task ``RoboCasaEnv`` so the base env is
untouched.
"""

import numpy as np
import gymnasium

from oat.env.robocasa.env import RoboCasaEnv
from oat.env.robocasa.factory import get_task_uid, num_robocasa_tasks

from typing import Optional, Dict


class MultiTaskRoboCasaEnv(RoboCasaEnv):
    """``RoboCasaEnv`` + a constant ``task_uid`` observation."""

    def __init__(self, env_name: str = "TurnOnSinkFaucet", **kwargs):
        super().__init__(env_name=env_name, **kwargs)

        # Canonical integer id for this task (must match the id used when the
        # multi-task zarr was built — both come from robocasa.factory).
        self.task_uid = get_task_uid(env_name)

        # Extend the observation space with task_uid (shape [1]), like LiberoEnv.
        self.observation_space.spaces["task_uid"] = gymnasium.spaces.Box(
            low=0, high=max(num_robocasa_tasks - 1, 0),
            shape=(1,), dtype=np.uint8,
        )

    def _extract_obs(
        self, raw_obs: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        obs_dict = super()._extract_obs(raw_obs)
        obs_dict["task_uid"] = np.array([self.task_uid], dtype=np.uint8)
        return obs_dict
