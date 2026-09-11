import torch
import dill
import hydra
from oat.common.hydra_util import register_new_resolvers
from oat.model.common.module_attr_mixin import ModuleAttrMixin
from typing import Optional


class BaseTokenizer(ModuleAttrMixin):

    @classmethod
    def from_checkpoint(cls, 
        checkpoint: str, 
        output_dir: Optional[str] = None,
        return_configuration: bool = False,
    ):
        with open(checkpoint, 'rb') as stream:
            payload = torch.load(stream, pickle_module=dill, map_location='cpu')
        cfg = payload['cfg']
        register_new_resolvers()
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)
        state_key = 'ema_model' if getattr(cfg.training, 'use_ema', False) else 'model'
        tokenizer.load_state_dict(payload['state_dicts'][state_key])
        tokenizer.eval()

        if return_configuration:
            return tokenizer, cfg
        else:
            return tokenizer
    
    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError
    
    def set_normalizer(self, *args, **kwargs):
        raise NotImplementedError
    
    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    def decode(self, *args, **kwargs):
        raise NotImplementedError
    
    def autoencode(self, *args, **kwargs):
        raise NotImplementedError
    
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError
    
    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
    