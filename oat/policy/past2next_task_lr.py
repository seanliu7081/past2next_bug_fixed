"""Optional independent learning rate for the categorical task residual only."""
import math
from numbers import Real

from oat.perception.task_residual_fused_obs_encoder import TaskResidualFusedObservationEncoder
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy


class Past2NextSelfPastTaskLRPolicy(Past2NextSelfPastPolicy):
    """Preserve policy state/inference and optionally split one optimizer tensor.

    With task_residual_lr=None the parent optimizer is returned verbatim. With
    an explicit LR, the existing trainable 10x138 table moves from its encoder
    group to a fifth group. Its weight decay and every other parameter setting
    stay unchanged. A new five-group optimizer requires fresh initialization or
    continuation from a checkpoint made with this same optimizer recipe.
    """

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas,
                      task_residual_lr=None):
        kwargs = dict(policy_lr=policy_lr, obs_enc_lr=obs_enc_lr,
                      weight_decay=weight_decay, betas=betas)
        if task_residual_lr is None:
            return super().get_optimizer(**kwargs)
        if (isinstance(task_residual_lr, bool) or not isinstance(task_residual_lr, Real)
                or not math.isfinite(task_residual_lr) or task_residual_lr <= 0):
            raise ValueError('task_residual_lr must be a finite positive number or None')
        if not isinstance(self.obs_encoder, TaskResidualFusedObservationEncoder):
            raise ValueError('task_residual_lr requires TaskResidualFusedObservationEncoder')
        table = self.obs_encoder.task_residual.weight
        if tuple(table.shape) != (10, 138) or not table.requires_grad:
            raise ValueError('The categorical task residual must be a trainable 10x138 table')
        optimizer = super().get_optimizer(**kwargs)
        parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
        if len(parameters) != len({id(parameter) for parameter in parameters}):
            raise ValueError('Parent optimizer contains duplicate parameter references')
        matches = [group for group in optimizer.param_groups
                   if any(parameter is table for parameter in group['params'])]
        if len(matches) != 1:
            raise ValueError('Task residual must appear exactly once in the parent optimizer')
        source_group = matches[0]
        source_group['params'] = [parameter for parameter in source_group['params'] if parameter is not table]
        optimizer.add_param_group({'params': [table], 'lr': float(task_residual_lr),
                                   'weight_decay': source_group['weight_decay']})
        return optimizer
