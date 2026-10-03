"""Explicit observation encoder dispatch for additive modern-policy backends.

Legacy DINO artifacts have no encoder type: the presence of their ``dino``
configuration identifies that exact schema, without renaming any state keys.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

from omegaconf import OmegaConf

from oat.perception.convnext_token_obs_encoder import (
    ConvNeXtTokenObservationEncoder, EMBEDDING_LAYOUT, fusion_config,
)
from oat.perception.token_obs_encoder import TokenObservationEncoder


DINO_TYPE = "dinov3_tokens"
CONVNEXT_TYPE = "convnextv2_tokens"
ENCODER_TARGETS = {
    "oat.perception.token_obs_encoder.TokenObservationEncoder": DINO_TYPE,
    "oat.perception.convnext_token_obs_encoder.ConvNeXtTokenObservationEncoder": CONVNEXT_TYPE,
}


def normalize_observation_encoder_config(config):
    """Return a complete dispatch schema without loading weights or models.

This preserves provenance. Callers comparing architecture may separately remove
source paths/digests; shape, preprocessing, fusion and token layout remain part
of the contract. Unknown targets are rejected even when a type is supplied.
"""
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("Observation encoder config must be a mapping")
    cfg = copy.deepcopy(dict(config))
    target = cfg.pop("_target_", None)
    if target is not None and target not in ENCODER_TARGETS:
        raise ValueError(f"Unsupported observation encoder target: {target!r}")
    if cfg.pop("_recursive_", False):
        raise ValueError("Observation encoder construction must not recursively instantiate arbitrary targets")
    kind = cfg.get("encoder_type") or (ENCODER_TARGETS.get(target) if target else None)
    if kind is None:
        if "dino" in cfg and "convnext" not in cfg:
            kind = DINO_TYPE
        else:
            raise ValueError("Observation encoder requires an explicit encoder_type; only legacy dino schema is inferred")
    if kind not in (DINO_TYPE, CONVNEXT_TYPE):
        raise ValueError(f"Unsupported observation encoder type: {kind!r}")
    if target and ENCODER_TARGETS[target] != kind:
        raise ValueError("Observation encoder target and encoder_type disagree")
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("Unsupported observation encoder schema_version")
    cfg.update(encoder_type=kind, schema_version=1)
    defaults = dict(n_obs_steps=2, n_emb=768, num_queries=64,
                    resampler_depth=2, activation_checkpointing=False)
    if kind == DINO_TYPE:
        if cfg.get("convnext") is not None or cfg.get("convnext_encoder") is not None:
            raise ValueError("DINO encoder cannot contain ConvNeXt configuration")
        cfg.pop("convnext", None)
        defaults.update(n_head=12, ffn_dim=2048, dropout=0.0)
        cfg.setdefault("dino", {})
        feature_defaults = dict(load_mode="fresh", model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                                rgb_range="uint8", image_size=224, geometry="square_resize",
                                interpolation="bicubic", antialias=True,
                                brightness=0.0, contrast=0.0)
        feature = cfg["dino"]
    else:
        if cfg.get("dino") is not None or cfg.get("dino_encoder") is not None:
            raise ValueError("ConvNeXt encoder cannot contain DINO configuration")
        cfg.pop("dino", None)
        defaults.update(visual_resampler_dim=256, visual_resampler_heads=4,
                        visual_resampler_ffn_dim=768, dropout=0.1,
                        embedding_layout=EMBEDDING_LAYOUT)
        cfg.setdefault("convnext", {})
        feature_defaults = dict(
            model_name="convnextv2_nano.fcmae_ft_in22k_in1k", construction_mode="fresh",
            frozen=True, image_size=224, feature_stages=[2, 3],
            input_range="uint8_0_255", mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], brightness=0.1, contrast=0.1,
            interpolation="bicubic", antialias=True,
            backbone_config=dict(features_only=True, out_indices=[2, 3], in_chans=3,
                                 output_stride=32, drop_path_rate=0.0))
        feature = cfg["convnext"]
    if not isinstance(feature, Mapping):
        raise TypeError("Backbone configuration must be a mapping")
    # Never expand dynamic targets through Hydra. Only the whitelisted outer
    # encoder constructor may instantiate a backbone using its fixed model ID.
    if "_target_" in feature or "backbone" in feature:
        raise ValueError("Serialized backbone config cannot inject a target or Python module")
    for key, value in defaults.items():
        cfg.setdefault(key, copy.deepcopy(value))
    for key, value in feature_defaults.items():
        feature.setdefault(key, copy.deepcopy(value))
    if kind == CONVNEXT_TYPE:
        cfg.setdefault("fusion", fusion_config(cfg["visual_resampler_dim"]))
        feature["feature_stages"] = list(feature["feature_stages"])
        feature["mean"], feature["std"] = list(feature["mean"]), list(feature["std"])
    return cfg


def build_observation_encoder(config, construction_mode="fresh"):
    if construction_mode not in ("fresh", "restore"):
        raise ValueError("construction_mode must be fresh or restore")
    cfg = normalize_observation_encoder_config(config)
    kind = cfg.pop("encoder_type")
    cfg.pop("schema_version")
    if kind == DINO_TYPE:
        cfg["dino"]["load_mode"] = construction_mode
        if construction_mode == "restore":
            cfg["dino"].pop("pretrained_path", None)
        return TokenObservationEncoder(**cfg)
    cfg["convnext"]["construction_mode"] = construction_mode
    if construction_mode == "restore":
        cfg["convnext"].pop("model_path", None)
    return ConvNeXtTokenObservationEncoder(**cfg)


def maintain_frozen_backbone_mode(encoder):
    """Maintain shared frozen eval semantics without changing old DINO files."""
    if isinstance(encoder, ConvNeXtTokenObservationEncoder):
        encoder.maintain_frozen_backbone_mode()
    elif isinstance(encoder, TokenObservationEncoder):
        encoder.dino_encoder.backbone.requires_grad_(False).eval()
    else:
        raise TypeError(f"Unsupported observation encoder class: {type(encoder).__name__}")
