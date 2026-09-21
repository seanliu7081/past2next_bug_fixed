"""Offline self-past training with execution-confirmed history at inference."""

from oat.policy.past2next_executed_past import Past2NextExecutedPastPolicy
from oat.policy.past2next_self_past import Past2NextSelfPastPolicy


class Past2NextSelfPastExecutedPolicy(
    Past2NextSelfPastPolicy, Past2NextExecutedPastPolicy,
):
    """Keep the self-past loss and acknowledge commands during rollout.

    Training uses the original offline self-past curriculum: generated previous
    commands are synthetic conditions paired with demonstration observations
    and targets. Training does not execute these commands in an environment.

    Stateful inference uses only commands acknowledged through
    ``record_executed_actions`` for both past-action and dynamic conditions.
    Use an execution-aware runner, or acknowledge each executed prefix before
    calling ``predict_action`` again. Explicit ``past_actions`` remains the
    stateless offline-validation interface.

    The cooperative inheritance order is deliberate: SelfPast supplies the
    constructor, forward loss, and persistent curriculum; ExecutedPast supplies
    inference, reset, and acknowledgement. SelfPast's constructor calls through
    ExecutedPast to initialize the shared base exactly once. No learned
    parameters or checkpoint keys are added relative to SelfPast.
    """

    def get_policy_name(self):
        return "past2next_selfpast_executed_" + "|".join(
            modality for modality in self.modalities if modality != "state"
        )
