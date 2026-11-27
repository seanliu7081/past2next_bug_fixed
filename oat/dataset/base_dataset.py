import hydra
import numpy as np
import os
import tqdm
from PIL import Image
import time
import torch
from typing import Dict, Optional
from oat.model.common.normalizer import LinearNormalizer


def make_train_validation_datasets(task):
    """Use the usual episode split or reserve an entire separate Zarr for validation."""
    config = task.dataset
    validation_path = task.get("validation_zarr_path")
    if validation_path is None:
        train = hydra.utils.instantiate(config)
        return train, train.get_validation_dataset()

    if not validation_path or os.path.realpath(validation_path) == os.path.realpath(config.zarr_path):
        raise ValueError("validation_zarr_path must identify a separate validation store")
    train = hydra.utils.instantiate(config, val_ratio=0.0)
    validation = hydra.utils.instantiate(
        config, zarr_path=validation_path, val_ratio=0.0, max_train_episodes=None,
    )
    if not len(validation):
        raise ValueError(f"No validation windows in {validation_path}")
    return train, validation


class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()

    def replay(self, 
        obs_key: str, type: str = 'rgb', 
        dt: float = 0.1, sample_every: int = 1,
        export_dir: Optional[str] = None,
    ):
        if export_dir is not None:
            os.makedirs(export_dir, exist_ok=True)
        try:
            for idx in tqdm.tqdm(range(len(self)), desc=f'Replay {obs_key}'):
                sample = self.seq_sampler.sample_sequence(idx)
                obs = self._sample_to_data(sample)['obs'][obs_key][0]
                if type == 'rgb' and idx % sample_every == 0:
                    img = Image.fromarray(obs.astype(np.uint8))
                    if export_dir is None:
                        img.save('rgb.png')
                    else:
                        img.save(os.path.join(export_dir, f'rgb_{idx}.png'))
                    time.sleep(dt)
        except KeyboardInterrupt as e:
            print(f"An error occurred: {e}")
        finally:
            if os.path.exists('rgb.png'):
                os.remove('rgb.png')
            if os.path.exists('depth.png'):
                os.remove('depth.png')
