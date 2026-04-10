"""
Auto-registration entry point for IsoGraph SDPO policy loss.

This module registers the 'isograph' policy loss with verl's loss registry
when imported. It can be loaded via verl's external_lib mechanism:

    actor_rollout_ref.model.external_lib=verl.trainer.ppo.register_isograph_loss
"""

from verl.trainer.ppo.isograph_sdpo import compute_policy_loss_isograph
from verl.trainer.ppo.core_algos import register_policy_loss, POLICY_LOSS_REGISTRY

if "isograph" not in POLICY_LOSS_REGISTRY:
    register_policy_loss("isograph")(compute_policy_loss_isograph)
    print("[IsoGraph] Policy loss 'isograph' registered via external_lib.")
