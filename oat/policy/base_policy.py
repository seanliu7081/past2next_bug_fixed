from typing import Dict, List, Union, Optional, Tuple
import torch
import dill
import hydra
from omegaconf import OmegaConf
from oat.common.hydra_util import register_new_resolvers
from oat.model.common.module_attr_mixin import ModuleAttrMixin
from oat.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    n_obs_steps: int
    n_action_steps: int

    @classmethod
    def from_checkpoint(cls, 
        checkpoint: str,
        output_dir: Optional[str] = None,
        return_configuration: bool = False,
        weights: Optional[str] = None,
        policy_overrides: Optional[Dict] = None,
    ):
        # Trusted local checkpoints load on CPU without workspace/optimizer allocation.
        with open(checkpoint, 'rb') as stream:
            payload = torch.load(stream, pickle_module=dill, map_location='cpu')
        # Rebuild a mutable copy: Hydra checkpoint configs can be struct-locked.
        cfg = OmegaConf.create(OmegaConf.to_container(payload['cfg'], resolve=False))
        if policy_overrides:
            cfg = OmegaConf.merge(cfg, {'policy': policy_overrides})
        register_new_resolvers()
        if weights is None:
            weights = 'ema' if getattr(cfg.training, 'use_ema', False) else 'model'
        if weights not in ('ema', 'model'):
            raise ValueError("weights must be 'ema' or 'model'")
        state_key = 'ema_model' if weights == 'ema' else 'model'
        if state_key not in payload['state_dicts']:
            raise ValueError(f'Checkpoint does not contain {state_key} weights')
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(payload['state_dicts'][state_key])
        policy.eval()

        if return_configuration:
            return policy, cfg
        else:
            return policy
    
    def get_optimizer(self, *args, **kwargs):
        return torch.optim.AdamW(self.parameters(), *args, **kwargs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: Union[LinearNormalizer, List[LinearNormalizer]]):
        raise NotImplementedError()
    
    def get_observation_encoder(self):
        raise NotImplementedError()
    
    def get_observation_modalities(self) -> List[str]:
        raise NotImplementedError()
    
    def get_observation_ports(self) -> List[str]:
        raise NotImplementedError()
    
    def get_policy_name(self) -> str:
        raise NotImplementedError()
    
    def create_dummy_observation(self,
        batch_size: int,
        horizon: int,
        obs_key_shapes: Dict[str, Tuple[int]],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        for obs_port, obs_shape in obs_key_shapes.items():
            obs_dict[obs_port] = torch.randn(
                size=(batch_size, horizon, *obs_shape),
            ).to(device)
        return obs_dict
