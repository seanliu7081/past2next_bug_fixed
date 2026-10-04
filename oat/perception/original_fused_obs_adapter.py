"""Original trainable Past2Next observation recipe with a modern-width bridge.

RGB and state normalization, crop selection, both camera networks, SpatialSoftmax,
ReLU, and concatenation run in the unchanged FusedObservationEncoder. Only a
linear bridge and learned frame positions are added after its output.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.crop_randomizer import CropRandomizer
from oat.perception.fused_obs_encoder import FusedObservationEncoder
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
from oat.perception.state_encoder import ProjectionStateEncoder


ENCODER_TYPE = "original_fused"
ENCODER_SCHEMA_VERSION = 2
_DEFAULT_CONFIG = dict(
    crop_shape=[112, 112],
    eval_fixed_crop=True,
    use_group_norm=True,
    share_rgb_model=False,
    pretrained=False,
    state_out_dim=None,
    feature_dimension=64,
    spatial_softmax_num_kp=32,
    spatial_softmax_temperature=1.0,
    spatial_softmax_noise=0.0,
)


def _plain_dict(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError("Original observation configuration must be a mapping.")
    return copy.deepcopy(dict(value))


def normalize_original_obs_config(config=None):
    """Resolve the original recipe, rejecting unsupported implicit extensions."""
    supplied = {} if config is None else _plain_dict(config)
    unknown = sorted(set(supplied) - set(_DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unsupported original observation options: {unknown}")
    resolved = copy.deepcopy(_DEFAULT_CONFIG)
    resolved.update(supplied)
    # These values are implicit in the original implementation. Pretending to
    # pass them to its constructors would silently change or discard a recipe.
    for key in ("pretrained", "state_out_dim", "feature_dimension",
                "spatial_softmax_num_kp", "spatial_softmax_temperature",
                "spatial_softmax_noise"):
        if resolved[key] != _DEFAULT_CONFIG[key]:
            raise ValueError(f"Original encoder requires {key}={_DEFAULT_CONFIG[key]!r}.")
    for key in ("eval_fixed_crop", "use_group_norm", "share_rgb_model"):
        if not isinstance(resolved[key], bool):
            raise TypeError(f"{key} must be boolean.")
    if not resolved["eval_fixed_crop"]:
        raise ValueError("Original fused policy requires deterministic evaluation crops.")
    crop = resolved["crop_shape"]
    if crop is not None:
        if not isinstance(crop, (tuple, list)) or len(crop) != 2:
            raise ValueError("crop_shape must be a two-element size or null.")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in crop):
            raise ValueError("crop_shape dimensions must be positive integers.")
        resolved["crop_shape"] = list(crop)
    return resolved


resolve_original_obs_config = normalize_original_obs_config


def validate_normalizer_fields(normalizer, field_widths, *, check_values=True):
    """Validate dynamic normalizer state that the original loader cannot police.

    The old DictOfTensorMixin reconstructs ParameterDict entries itself, so
    strict=True alone cannot notice a missing scale, offset, or input statistic.
    Scalar normalization and per-feature normalization are both original APIs.
    """
    missing = [key for key in field_widths if key not in normalizer.params_dict]
    if missing:
        raise ValueError(f"Original encoder requires dataset normalizer statistics for {missing}.")
    for key, width in field_widths.items():
        params = normalizer.params_dict[key]
        if any(name not in params for name in ("scale", "offset", "input_stats")):
            raise ValueError(f"Incomplete dataset normalizer for {key}: scale/offset/input_stats required.")
        scale, offset, stats = params["scale"], params["offset"], params["input_stats"]
        if any(name not in stats for name in ("min", "max", "mean", "std")):
            raise ValueError(f"Incomplete input statistics for {key}: min/max/mean/std required.")
        values = [scale, offset, *[stats[name] for name in ("min", "max", "mean", "std")]]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1
               or not value.is_floating_point() for value in values):
            raise ValueError(f"Dataset normalizer for {key} requires floating point vectors.")
        if scale.numel() not in (1, width) or any(value.shape != scale.shape for value in values):
            raise ValueError(f"Dataset normalizer for {key} requires one or {width} matching statistics.")
        if check_values:
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError(f"Nonfinite dataset normalizer for {key}.")
            if (scale == 0).any():
                raise ValueError(f"Dataset normalizer scale for {key} must be nonzero.")
            if (stats["std"] < 0).any() or (stats["min"] > stats["max"]).any():
                raise ValueError(f"Invalid dataset input statistics for {key}.")


class OriginalFusedObservationAdapter(BaseObservationEncoder):
    def __init__(self, shape_meta, n_obs_steps=2, embed_dim=768,
                 original_obs_config=None):
        super().__init__()
        self.shape_meta = _plain_dict(shape_meta)
        self.original_obs_config = normalize_original_obs_config(original_obs_config)
        self.n_obs_steps = int(n_obs_steps)
        self.embed_dim = int(embed_dim)
        if self.n_obs_steps < 1 or self.embed_dim < 1:
            raise ValueError("n_obs_steps and embed_dim must be positive.")
        self.rgb_ports = []
        self.state_ports = []
        for name, attr in self.shape_meta["obs"].items():
            kind, shape = attr.get("type"), list(attr["shape"])
            if kind == "rgb":
                if len(shape) != 3 or shape[-1] != 3:
                    raise ValueError(f"RGB {name} requires HWC shape with three channels.")
                crop = self.original_obs_config["crop_shape"]
                if crop is not None and any(c >= size for c, size in zip(crop, shape[:2])):
                    raise ValueError(f"Crop must be strictly smaller than RGB {name}: {shape}.")
                self.rgb_ports.append(name)
            elif kind == "state":
                if len(shape) != 1 or shape[0] < 1:
                    raise ValueError(f"State {name} requires a nonempty vector shape.")
                self.state_ports.append(name)
            else:
                raise ValueError(f"Original fused adapter supports RGB+state only: {name}={kind!r}.")
        if not self.rgb_ports or not self.state_ports:
            raise ValueError("Original fused adapter requires both RGB and state observations.")
        shapes = {tuple(self.shape_meta["obs"][key]["shape"]) for key in self.rgb_ports}
        if len(shapes) != 1:
            raise ValueError("Original RGB encoder requires cameras with identical image shapes.")
        self._verify_dependency_defaults()
        cfg = self.original_obs_config
        # Preserve the original construction and RNG order; initialize the new
        # bridge only after the original encoder has finished initializing.
        self.fused_encoder = FusedObservationEncoder(
            shape_meta=self.shape_meta,
            vision_encoder={
                "_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
                **{key: copy.deepcopy(cfg[key]) for key in (
                    "crop_shape", "eval_fixed_crop", "use_group_norm", "share_rgb_model")},
            },
            state_encoder={
                "_target_": "oat.perception.state_encoder.ProjectionStateEncoder",
                "out_dim": None,
            },
        )
        self.fused_feature_dim = self.fused_encoder.output_feature_dim()
        self.obs_projection = nn.Linear(self.fused_feature_dim, self.embed_dim)
        self.frame_embedding = nn.Parameter(torch.empty(self.n_obs_steps, self.embed_dim))
        nn.init.normal_(self.frame_embedding, std=0.02)
        self._effective_recipe = self._inspect_effective_recipe()
        self.register_load_state_dict_post_hook(self._freeze_after_restore)

    @staticmethod
    def _verify_dependency_defaults():
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        from robomimic.models.obs_nets import ObservationEncoder
        checks = (
            (ResNet18Conv, "pretrained", False),
            (ObservationEncoder, "feature_activation", nn.ReLU),
            (SpatialSoftmax, "noise_std", 0.0),
            (SpatialSoftmax, "learnable_temperature", False),
            (SpatialSoftmax, "output_variance", False),
        )
        for cls, key, required in checks:
            parameter = inspect.signature(cls.__init__).parameters.get(key)
            if parameter is None or parameter.default != required:
                raise RuntimeError(
                    f"Installed {cls.__name__}.{key} default differs from the original recipe; "
                    f"expected {required!r}. Check the saved dependency versions."
                )

    def _inspect_effective_recipe(self):
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        encoder = self.fused_encoder.vision_encoder.encoder
        if not isinstance(encoder.activation, nn.ReLU):
            raise RuntimeError("Original ObservationEncoder must apply ReLU.")
        if not isinstance(self.fused_encoder.state_encoder.state_proj, nn.Identity):
            raise RuntimeError("Original state features must remain an identity projection.")
        cameras = []
        networks = []
        for key in self.rgb_ports:
            # Shared-camera mode is an explicit alternative resolved recipe;
            # the default and all shipped configs construct independent CNNs.
            source = encoder.obs_share_mods.get(key) or key
            network = encoder.obs_nets[source]
            networks.append(network)
            pool = network.pool
            if not isinstance(network.backbone, ResNet18Conv) or not isinstance(pool, SpatialSoftmax):
                raise RuntimeError("Original recipe requires ResNet18Conv and SpatialSoftmax.")
            if (network.feature_dimension != 64 or pool._num_kp != 32
                    or float(pool.temperature.detach().item()) != 1.0
                    or pool.noise_std != 0.0 or pool.learnable_temperature or pool.output_variance):
                raise RuntimeError("Installed VisualCore/SpatialSoftmax changes the original recipe.")
            crop = encoder.obs_randomizers[key]
            if self.original_obs_config["crop_shape"] is not None and not isinstance(crop, CropRandomizer):
                raise RuntimeError("Original evaluation requires the center-crop randomizer.")
            group_norms = [module for module in network.modules() if isinstance(module, nn.GroupNorm)]
            batch_norms = [module for module in network.modules() if isinstance(module, nn.BatchNorm2d)]
            if self.original_obs_config["use_group_norm"]:
                if batch_norms or not group_norms or any(
                        module.num_groups != module.num_channels // 16 for module in group_norms):
                    raise RuntimeError("Original GroupNorm replacement rule changed.")
            cameras.append(dict(
                port=key, input_shape=list(self.shape_meta["obs"][key]["shape"]),
                feature_dim=network.feature_dimension,
                backbone=type(network.backbone).__name__, pretrained=False,
                spatial_softmax=dict(num_kp=pool._num_kp, temperature=1.0,
                                     noise_std=pool.noise_std, learnable_temperature=False),
                group_norm_groups=[module.num_groups for module in group_norms],
                crop_shape=None if crop is None else [crop.crop_height, crop.crop_width],
                training_crop="random" if crop is not None else "none",
                evaluation_crop="center" if crop is not None else "none",
            ))
        shared = len({id(network) for network in networks}) == 1 and len(networks) > 1
        if len(networks) > 1 and shared != self.original_obs_config["share_rgb_model"]:
            raise RuntimeError("Camera sharing differs from the resolved original recipe.")
        return dict(cameras=cameras, visual_activation="ReLU", state_projection="Identity",
                    normalization="original_encoder_internal_dataset_statistics",
                    pretrained=False, share_rgb_model=self.original_obs_config["share_rgb_model"],
                    initialization="original_random_initialization_then_bridge",
                    backbone_trainable=True)

    def modalities(self):
        return ["rgb", "state"]

    def output_feature_dim(self):
        return self.embed_dim

    def _normalizer_field_widths(self, ports=None):
        return {key: self.shape_meta["obs"][key]["shape"][-1]
                for key in (self.rgb_ports + self.state_ports if ports is None else ports)}

    def validate_normalizer(self, normalizer):
        validate_normalizer_fields(normalizer, self._normalizer_field_widths())

    def freeze_normalizers(self):
        for module in self.fused_encoder.modules():
            if isinstance(module, LinearNormalizer):
                module.requires_grad_(False)

    def _freeze_after_restore(self, module, incompatible_keys):
        self.freeze_normalizers()
        for encoder, ports in ((self.fused_encoder.vision_encoder, self.rgb_ports),
                               (self.fused_encoder.state_encoder, self.state_ports)):
            validate_normalizer_fields(encoder.normalizer, self._normalizer_field_widths(ports))

    def set_normalizer(self, normalizer):
        self.validate_normalizer(normalizer)
        self.fused_encoder.set_normalizer(normalizer)
        self.freeze_normalizers()

    def _validate_input(self, obs_dict):
        batch = None
        for key in self.rgb_ports + self.state_ports:
            if key not in obs_dict:
                raise ValueError(f"Missing original observation {key}.")
            value = obs_dict[key]
            shape = tuple(self.shape_meta["obs"][key]["shape"])
            if not isinstance(value, torch.Tensor) or value.ndim != len(shape) + 2:
                raise ValueError(f"Observation {key} must be a tensor [B, {self.n_obs_steps}, {shape}].")
            if batch is None:
                batch = value.shape[0]
            if value.shape != (batch, self.n_obs_steps, *shape):
                raise ValueError(f"Observation {key} has incorrect frame, batch, or feature dimensions.")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Observation {key} contains nonfinite values.")
        # No fallback to unnormalized state or RGB is allowed, including after
        # restoring an artifact without having called set_normalizer().
        for encoder, ports in ((self.fused_encoder.vision_encoder, self.rgb_ports),
                               (self.fused_encoder.state_encoder, self.state_ports)):
            validate_normalizer_fields(
                encoder.normalizer, self._normalizer_field_widths(ports), check_values=False)

    def encode_fused(self, obs_dict):
        """Return the unchanged original feature, before bridge and positions."""
        self._validate_input(obs_dict)
        features = self.fused_encoder(obs_dict)
        if features.shape[1:] != (self.n_obs_steps, self.fused_feature_dim):
            raise RuntimeError("Original encoder violated its declared fused output shape.")
        return features

    def forward(self, obs_dict):
        features = self.obs_projection(self.encode_fused(obs_dict))
        return features + self.frame_embedding[None].to(dtype=features.dtype)

    def export_config(self):
        """Constructor-compatible, self-contained Hydra config for strict restore."""
        return dict(
            _target_="oat.perception.original_fused_obs_adapter.OriginalFusedObservationAdapter",
            shape_meta=copy.deepcopy(self.shape_meta), n_obs_steps=self.n_obs_steps,
            embed_dim=self.embed_dim, original_obs_config=copy.deepcopy(self.original_obs_config),
        )

    def export_metadata(self):
        """Resolved structure, ordering, dependency versions and source hashes."""
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        from robomimic.models.obs_nets import ObservationEncoder
        versions = {}
        for package in ("torch", "torchvision", "robomimic", "hydra-core", "omegaconf"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        source_hashes = {}
        for cls in (type(self), FusedObservationEncoder, RobomimicRgbEncoder,
                    ProjectionStateEncoder, CropRandomizer, ResNet18Conv,
                    SpatialSoftmax, ObservationEncoder):
            path = inspect.getsourcefile(cls)
            if path is not None:
                source_hashes[f"{cls.__module__}.{cls.__name__}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return dict(
            encoder_type=ENCODER_TYPE, schema_version=ENCODER_SCHEMA_VERSION,
            fused_feature_dim=self.fused_feature_dim, observation_tokens=self.n_obs_steps,
            rgb_ports=list(self.rgb_ports), state_ports=list(self.state_ports),
            fusion_order=list(self.rgb_ports) + list(self.state_ports),
            original_obs_config=copy.deepcopy(self.original_obs_config),
            effective_recipe=copy.deepcopy(self._effective_recipe),
            dependency_versions=versions, source_hashes=source_hashes,
            frame_embedding_shape=list(self.frame_embedding.shape),
        )
