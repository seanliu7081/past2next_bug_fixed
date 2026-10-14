"""Small random OAT fixture for vision/action tests; never a training asset."""
from pathlib import Path

import dill
import hydra
from omegaconf import OmegaConf
import torch

from oat.model.common.normalizer import SingleFieldLinearNormalizer


def make_test_tokenizer_checkpoint(path):
    """Create a real 5000-code, 8-token/16-action OAT with small transformer widths."""
    path = Path(path)
    common = dict(sample_dim=7, sample_horizon=16, emb_dim=32, head_dim=16,
                  depth=1, pdropout=0., latent_dim=5)
    config = {
        '_target_': 'oat.tokenizer.oat.tokenizer.OATTok',
        'encoder': dict(_target_='oat.tokenizer.oat.encoder.register_encoder.RegisterEncoder',
                        **common, num_registers=8),
        'decoder': dict(_target_='oat.tokenizer.oat.decoder.single_pass_decoder.SinglePassDecoder',
                        **common, token_dropout_mode='pow2', use_causal_decoder=True, latent_horizon=8),
        'quantizer': {'_target_': 'oat.tokenizer.oat.quantizer.fsq.FSQ', 'levels': [8, 5, 5, 5, 5]},
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(941)
        tokenizer = hydra.utils.instantiate(config)
    ones, zeros = torch.ones(7), torch.zeros(7)
    tokenizer.normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
        ones, zeros, {'min': -ones, 'max': ones, 'mean': zeros, 'std': ones})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'cfg': OmegaConf.create({'tokenizer': config}),
                'state_dicts': {'ema_model': tokenizer.state_dict()}}, path, pickle_module=dill)
    return path
