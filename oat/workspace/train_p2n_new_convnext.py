"""Additive encoder-aware workspace; the existing training loop is inherited."""
from __future__ import annotations

import os
import platform

import torch

from oat.policy.p2n_new_convnext import (
    ENCODER_POLICY_FIELDS,
    first_config_difference,
    observation_encoder_contract,
    policy_observation_encoder_config,
)
from oat.workspace.train_p2n_new import TrainP2NNewWorkspace as _ModernWorkspace


class TrainP2NNewWorkspace(_ModernWorkspace):
    @staticmethod
    def validate_resume_payload(payload, cfg):
        _ModernWorkspace.validate_resume_payload(payload, cfg)
        # Reject conflicting duplicate public fields even when an embedded
        # encoder configuration would otherwise hide/override those fields.
        source_fields = {'obs_encoder_config', 'convnext_path', 'convnext_revision', 'dino_path', 'dino_revision'}
        for key in sorted(ENCODER_POLICY_FIELDS - source_fields):
            if payload['cfg'].policy.get(key) != cfg.policy.get(key):
                raise ValueError(f'Resume encoder architecture/preprocessing mismatch: policy.{key}')
        # Compare effective normalized encoder fields, including settings nested
        # in a deployment-style encoder config. Paths are deliberately excluded:
        # the artifact supplies both student/EMA and frozen-backbone tensors.
        saved = observation_encoder_contract(policy_observation_encoder_config(payload['cfg'].policy))
        requested = observation_encoder_contract(policy_observation_encoder_config(cfg.policy))
        difference = first_config_difference(saved, requested)
        if difference:
            raise ValueError(f'Resume encoder architecture/preprocessing mismatch: {difference}')
        exported = payload['policy_config'].get('obs_encoder_config')
        if exported is None:
            raise ValueError('Resume requires the embedded observation encoder configuration')
        artifact = observation_encoder_contract(exported)
        if artifact['encoder_type'] != requested['encoder_type']:
            raise ValueError('Resume encoder architecture/preprocessing mismatch: obs_encoder.encoder_type')
        if artifact['encoder_type'] == 'convnextv2_tokens':
            difference = first_config_difference(artifact, requested)
            if difference:
                raise ValueError(f'Resume encoder architecture/preprocessing mismatch: {difference}')
        if payload['cfg'].policy.get('expected_action_tokens') != cfg.policy.get('expected_action_tokens'):
            raise ValueError('Resume architecture/schema mismatch: policy.expected_action_tokens')
        metadata_contract = payload['metadata'].get('observation_encoder_contract')
        if metadata_contract is not None:
            difference = first_config_difference(metadata_contract, artifact)
            if difference:
                raise ValueError(f'Artifact encoder metadata disagrees with policy configuration: {difference}')
        return payload

    def training_report(self, model):
        report = super().training_report(model)
        report.update(
            encoder_type=model.obs_encoder_type,
            policy_name=model.get_policy_name(),
            parameters=model.parameter_counts(),
            environment={
                'python': platform.python_version(),
                'torch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'world_size': int(os.environ.get('WORLD_SIZE', '1')),
            },
            performance_contract={
                'measurements': 'Performance is unmeasured; parameter counts do not imply speedup.',
                'self_past_chunk_size': model.self_past_chunk_size,
                'self_past_probability': model.self_past_probability(),
                'self_past_temperature': model.self_past_temperature,
                'self_past_topk': model.self_past_topk,
                'activation_checkpointing': model._construction['activation_checkpointing'],
                'visual_tokens': model.obs_encoder.num_visual_tokens,
                'observation_frames': model.n_obs_steps,
                'visual_queries_per_image': model.obs_encoder.num_queries,
                'batch_per_rank': self.cfg.dataloader.batch_size,
                'gradient_accumulation': self.cfg.training.gradient_accumulate_every,
            },
        )
        return report
