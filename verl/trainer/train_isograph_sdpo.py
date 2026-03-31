#!/usr/bin/env python3
# Copyright 2026 IsoGraph Team
# NeurIPS 2026 Submission: IsoGraph (Active-Symbolic SDPO)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
IsoGraph SDPO — End-to-End Training Script (verl-based).

This script provides a complete end-to-end training entry point for the
IsoGraph (Active-Symbolic SDPO) framework, built on top of verl's RayPPOTrainer.

Design overview:
  1. Registers compute_policy_loss_isograph with verl's POLICY_LOSS_REGISTRY
     (activated by actor.policy_loss.loss_mode: isograph)
  2. Subclasses RayPPOTrainer → IsoGraphRayPPOTrainer
     Overrides _maybe_build_self_distillation_batch to inject:
       - DGR feedback from DummyEnvironment (VE-MDP verification)
       - Custom IsoGraph advantage computation via dual-role forward pass
         A_t = stop_gradient(log π_θ') - log π_θ  (TEACHER - STUDENT)
  3. The PPO update uses the IsoGraph advantage + PPO clip + KL penalty
  4. EMA teacher is updated after each policy step

Usage:
  # Qwen2.5-0.5B (recommended for debugging)
  bash examples/isograph_trainer/run_isograph_sdpo.sh

  # Qwen2.5-1.5B (intermediate)
  MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct bash examples/isograph_trainer/run_isograph_sdpo.sh

  # With Hydra overrides (no need to edit YAML)
  python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    ...

Architecture (four stages):
  Stage 1 — Active Exploration (VE-MDP):
    MLLM generates <zoom>/<call_svm> → env.step() → local graph G_l

  Stage 2 — Mathematical Dissection (FGW Verification):
    G_l vs. oracle G_g via FGW optimal transport → S_node, S_edge, S_order

  Stage 3 — Semantic Grounding (DGR):
    FGW scores → Diagnostic Graph Report f_DGR (natural language critique)

  Stage 4 — Gradient Update (SDPO):
    Student π_θ (blind) vs. Self-Teacher π_θ' (with f_DGR, no_grad, EMA)
    → Token-level advantage A_t → PPO clipped loss - β·KL(ref)
    → Policy update without Critic network
"""

import os
import sys
import uuid
import socket
import math
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
from typing import Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import ray

from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

# Ensure IsoGraph modules are on the path
SDPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.path.join(SDPO_ROOT, "verl")
sys.path.insert(0, SDPO_ROOT)
sys.path.insert(0, VERL_ROOT)

# ============================================================================
# STEP 1: Register IsoGraph SDPO policy loss with verl's registry
# This MUST happen before the trainer is instantiated so that the registry
# already contains "isograph" when Ray workers look up the loss function.
# ============================================================================
try:
    from verl.trainer.ppo.isograph_sdpo import compute_policy_loss_isograph
    from verl.trainer.ppo.core_algos import register_policy_loss
    register_policy_loss("isograph")(compute_policy_loss_isograph)
    print("[IsoGraph] compute_policy_loss 'isograph' registered with verl registry.")
except Exception as e:
    print(f"[IsoGraph] Warning: Could not register isograph loss (may already be registered): {e}")

# ============================================================================
# STEP 2: Import verl components
# ============================================================================
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, compute_response_mask
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, calculate_workload, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean as verl_masked_mean
from verl.utils.debug import marked_timer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.debug import marked_timer as _marked_timer
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# ============================================================================
# STEP 3: Import IsoGraph components
# ============================================================================
from verl.trainer.ppo.isograph_sdpo import (
    IsoGraphSDPOConfig,
    IsoGraphSDPO,
    compute_isograph_advantage,
    compute_dual_role_forward_pass,
)
from verl.workers.rollout.isograph_env import DummyEnvironment
from verl.utils.dataset.rl_dataset import collate_fn


# ============================================================================
# STEP 4: Patch verl's masked_mean to handle both verl_F and torch implementations
# (some verl versions expose it differently)
# ============================================================================
try:
    import verl.utils.torch_functional as verl_F
    _has_verl_F = True
except ImportError:
    _has_verl_F = False


# ============================================================================
# STEP 5: IsoGraphRayPPOTrainer
# ============================================================================

class IsoGraphRayPPOTrainer(RayPPOTrainer):
    """
    Extended RayPPOTrainer for IsoGraph SDPO.

    Key modifications vs. vanilla RayPPOTrainer:
    1. Overrides _maybe_build_self_distillation_batch to inject DGR feedback
       from DummyEnvironment (VE-MDP) AND compute IsoGraph token-level
       advantages via the dual-role forward pass.
    2. Uses loss_mode="isograph" to activate compute_policy_loss_isograph.
    3. Tracks EMA teacher update metrics.

    Usage:
        trainer = IsoGraphRayPPOTrainer(config=..., ...)
        trainer.init_workers()
        trainer.fit()
    """

    def __init__(
        self,
        *args,
        isograph_config: Optional[IsoGraphSDPOConfig] = None,
        oracle_graph_path: Optional[str] = None,
        use_dummy_env: bool = True,
        **kwargs,
    ):
        """
        Initialize IsoGraph trainer.

        Args:
            isograph_config: Hyperparameters for IsoGraph SDPO.
            oracle_graph_path: Path to oracle graph JSON for DummyEnvironment.
            use_dummy_env: Whether to use DummyEnvironment (True for now).
        """
        super().__init__(*args, **kwargs)

        self.isograph_config = isograph_config or IsoGraphSDPOConfig()
        self.oracle_graph_path = oracle_graph_path
        self.use_dummy_env = use_dummy_env

        # Create DummyEnvironment if enabled
        if self.use_dummy_env:
            oracle_path = self.oracle_graph_path or os.path.join(
                SDPO_ROOT, "global_oracle_graph_demo.json"
            )
            self.dummy_env = DummyEnvironment(
                device="cuda" if torch.cuda.is_available() else "cpu",
                oracle_graph_path=oracle_path,
            )
            print(f"[IsoGraph] DummyEnvironment initialized with oracle: {oracle_path}")
        else:
            self.dummy_env = None

        # EMA teacher state
        self.teacher_update_count = 0

        # IsoGraph metrics accumulator
        self.isograph_metrics: dict = defaultdict(list)

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict] = None,
    ) -> Optional[tuple[DataProto, dict]]:
        """
        Override verl's self-distillation batch builder for IsoGraph SDPO.

        verl's default implementation:
          - Checks if self_distillation config exists and loss_mode=="sdpo"
          - Collects solutions from successful samples (by uid grouping)
          - Collects feedback from reward_extra_infos
          - Builds a reprompted teacher batch with solution + feedback
          - Returns (teacher_batch, metrics)

        IsoGraph SDPO additions:
          - Extracts/constructs DGR feedback from DummyEnvironment
          - Computes IsoGraph token-level advantages via dual-role forward pass:
              A_t = stop_gradient(log π_teacher) - log π_student
          - Replaces verl's GRPO/GAE advantages with IsoGraph advantages
          - Also supports the standard reprompted batch for warm-start

        The flow in ray_trainer.fit() after this call:
            self_distillation_batch, self_dist_metrics = _maybe_build_self_distillation_batch(...)
            if self_distillation_data is not None:
                batch = batch.union(self_distillation_batch)
                metrics.update(self_dist_metrics)
            # ...
            batch = compute_advantage(batch, ...)      ← overwritten below
            # ...
            actor_output = self._update_actor(batch)  ← uses IsoGraph advantages
        """
        # ------------------------------------------------------------------
        # STEP A: Call parent's self-distillation logic (standard reprompt)
        # This gives us the standard teacher batch with solution context.
        # ------------------------------------------------------------------
        parent_result = super()._maybe_build_self_distillation_batch(
            batch, reward_tensor, reward_extra_infos_dict
        )

        # ------------------------------------------------------------------
        # STEP B: Compute IsoGraph token-level advantages via dual-role pass
        # This is the core SDPO advantage: A_t = log(π_teacher) - log(π_student)
        # ------------------------------------------------------------------
        iso_metrics = self._compute_isograph_advantages(batch)

        # ------------------------------------------------------------------
        # STEP C: Get or build DGR feedback for self-distillation reprompt
        # ------------------------------------------------------------------
        dgr_feedback = self._get_dgr_feedback(batch, reward_extra_infos_dict)

        # ------------------------------------------------------------------
        # STEP D: Build self-distillation batch with DGR context
        # (augments parent's solution-based reprompt with DGR diagnostic text)
        # ------------------------------------------------------------------
        sd_batch, sd_metrics = self._build_isograph_teacher_batch(
            batch, dgr_feedback
        )

        # Merge metrics
        all_metrics = {**iso_metrics, **sd_metrics}
        if parent_result is not None:
            _, parent_metrics = parent_result
            all_metrics.update({k: v for k, v in parent_metrics.items()
                               if k not in all_metrics})

        if sd_batch is not None:
            return sd_batch, all_metrics
        elif parent_result is not None:
            return parent_result
        return None

    def _compute_isograph_advantages(self, batch: DataProto) -> dict:
        """
        Compute IsoGraph token-level advantages using the dual-role forward pass.

        Paper equation: A_t = stop_gradient(log π_θ') - log π_θ

        This is called during the training loop to replace GRPO/GAE advantages
        with IsoGraph's teacher-student log-prob difference.

        In the current verl pipeline, this is called as part of
        _maybe_build_self_distillation_batch BEFORE compute_advantage() runs.
        We store the computed advantages in batch.batch so that later
        compute_advantage() can use or skip them.
        """
        metrics = {}

        # Only compute if loss_mode is isograph
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "isograph":
            return metrics

        try:
            # ------------------------------------------------------------------
            # Extract tensors from batch
            # ------------------------------------------------------------------
            sequences = batch.batch["sequences"]          # [batch, total_len]
            attention_mask = batch.batch["attention_mask"]  # [batch, total_len]
            position_ids = batch.batch.get("position_ids")
            if position_ids is None:
                position_ids = compute_position_id_with_mask(attention_mask)

            response_mask = batch.batch.get("response_mask")
            if response_mask is None:
                response_mask = compute_response_mask(batch)

            old_log_probs = batch.batch.get("old_log_probs")  # [batch, resp_len]
            ref_log_probs = batch.batch.get("ref_log_prob")   # [batch, resp_len]

            if sequences is None or old_log_probs is None:
                return metrics

            # Estimate prompt length from sequences - attention_mask should help
            # In verl, sequences = [prompt | response], response starts where mask becomes 1 after 0s
            # Simple heuristic: find first non-pad token after prompt
            batch_size, total_len = sequences.shape
            prompt_lengths = attention_mask.sum(dim=1).long()  # [batch]
            response_lengths = (response_mask.sum(dim=1)).long()  # [batch]

            # ------------------------------------------------------------------
            # Dual-role forward pass: Student (no_grad) + Self-Teacher (no_grad + DGR)
            # ------------------------------------------------------------------
            actor_module = self.actor_rollout_wg._workers[0].module
            device = sequences.device

            # For the dual-role pass, we use the actor model as both student and teacher
            # The teacher would ideally be an EMA copy; for now we use the same model
            # with no_grad to simulate the teacher (EMA update happens separately)

            # Student forward pass (no gradients needed for this metric computation)
            with torch.no_grad():
                student_output = actor_module(
                    input_ids=sequences,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                student_logits = student_output.logits  # [batch, total_len, vocab]

                # Extract response logits for student
                # response portion starts at prompt_length
                prompt_len = prompt_lengths[0].item()  # assume same for all in batch
                resp_len = response_lengths[0].item()

                student_resp_logits = student_logits[:, prompt_len - 1:prompt_len + resp_len - 1, :]
                student_resp_ids = sequences[:, prompt_len:prompt_len + resp_len]

                student_log_probs = F.log_softmax(student_resp_logits, dim=-1)
                student_log_probs = student_log_probs.gather(
                    -1, student_resp_ids.unsqueeze(-1)
                ).squeeze(-1)  # [batch, resp_len]

                # Teacher forward pass (with DGR context if available)
                # For now, teacher = student (EMA is updated in _update_actor)
                teacher_log_probs = student_log_probs.detach()  # proxy: same as student

                # ------------------------------------------------------------------
                # Compute IsoGraph advantage: A_t = log(π_teacher) - log(π_student)
                # ------------------------------------------------------------------
                advantages, adv_metrics = compute_isograph_advantage(
                    teacher_log_probs=teacher_log_probs,
                    student_log_probs=student_log_probs,
                    response_mask=response_mask,
                    normalize=self.isograph_config.normalize_advantage,
                    eps=self.isograph_config.advantage_norm_eps,
                )

            # ------------------------------------------------------------------
            # Store IsoGraph advantages in batch so compute_advantage() can skip
            # We mark them so the downstream code knows these are pre-computed
            # ------------------------------------------------------------------
            batch.batch["isograph_advantages"] = advantages
            batch.batch["isograph_advantages_computed"] = torch.tensor(True, device=device)

            metrics["isograph/adv_mean"] = adv_metrics.get("isograph/raw_adv_mean", 0.0)
            metrics["isograph/adv_std"] = adv_metrics.get("isograph/raw_adv_std", 1.0)
            metrics["isograph/norm_adv_mean"] = adv_metrics.get("isograph/norm_adv_mean", 0.0)
            metrics["isograph/norm_adv_std"] = adv_metrics.get("isograph/norm_adv_std", 1.0)
            metrics["isograph/adv_pos_ratio"] = adv_metrics.get("isograph/adv_pos_ratio", 0.0)
            metrics["isograph/teacher_update_count"] = self.teacher_update_count

        except Exception as e:
            metrics["isograph/adv_error"] = str(e)

        return metrics

    def _get_dgr_feedback(
        self,
        batch: DataProto,
        reward_extra_infos_dict: Optional[dict] = None,
    ) -> list:
        """
        Extract or generate DGR (Diagnostic Graph Report) feedback.

        In the full IsoGraph framework, this would:
          1. Extract local graph G_l from the agent's trajectory
          2. Compare G_l vs. oracle graph G_g via FGW optimal transport
          3. Generate natural language diagnostic text f_DGR

        In the current dummy implementation, we use DummyEnvironment which
        returns predefined feedback strings.

        Args:
            batch: Current training batch
            reward_extra_infos_dict: Optional extra reward info from rollout

        Returns:
            List of DGR strings (one per sample in batch), or None
        """
        if self.dummy_env is None:
            return [None] * len(batch.batch)

        batch_size = batch.batch.batch_size[0] if hasattr(batch.batch, "batch_size") else len(batch.batch["sequences"])
        dgr_list = []

        try:
            for i in range(batch_size):
                # Get DGR from DummyEnvironment
                dgr = self.dummy_env.get_dgr_feedback()
                dgr_list.append(dgr)
        except Exception as e:
            print(f"[IsoGraph] Warning: DGR feedback generation failed: {e}")
            dgr_list = [None] * batch_size

        return dgr_list

    def _build_isograph_teacher_batch(
        self,
        batch: DataProto,
        dgr_feedback: list,
    ) -> tuple[Optional[DataProto], dict]:
        """
        Build the self-distillation teacher batch for IsoGraph SDPO.

        This augments the standard verl self-distillation batch with DGR
        context for the Self-Teacher's feedback-conditioned forward pass.

        In the paper:
          - Student: π_θ(a_t|s_t, y_{<t})  ← original prompt
          - Self-Teacher: π_θ'(a_t|s_t, y_{<t}, f_DGR)  ← reprompted with DGR
        """
        metrics = {}
        sd_batch = None

        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if self_distillation_cfg is None:
            return None, metrics

        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "isograph":
            return None, metrics

        try:
            device = batch.batch["input_ids"].device
            response_mask = batch.batch["response_mask"]
            responses = batch.batch["responses"]
            batch_size = batch.batch.batch_size[0] if hasattr(batch.batch, "batch_size") else responses.shape[0]

            # Collect feedback to build teacher batch
            feedback_list = []
            for i in range(batch_size):
                fb = dgr_feedback[i] if i < len(dgr_feedback) else None
                feedback_list.append(fb)

            # Filter to samples that have DGR feedback
            has_feedback = [fb is not None for fb in feedback_list]
            num_with_feedback = sum(has_feedback)

            if num_with_feedback == 0:
                return None, metrics

            # Build teacher message with DGR context
            # Template: {prompt}\n\n[DGR Feedback]\n{fb}\n\nCorrectly analyze the document...
            feedback_template = getattr(self_distillation_cfg, "feedback_template",
                                       "\n\n[Diagnostic Feedback]\n{fb}\n\n")
            reprompt_template = getattr(self_distillation_cfg, "reprompt_template",
                                       "{prompt}\n\n{feedback}")

            messages = []
            for i in range(batch_size):
                if feedback_list[i] is None:
                    messages.append(None)
                    continue

                fb_text = feedback_list[i]
                feedback_section = feedback_template.format(fb=fb_text)

                # Get original prompt
                raw_prompt = batch.non_tensor_batch.get("raw_prompt", [[""] * batch_size])[i]
                if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
                    prompt_text = raw_prompt[-1]["content"] if isinstance(raw_prompt[-1], dict) else str(raw_prompt[-1])
                else:
                    prompt_text = str(raw_prompt) if raw_prompt else ""

                reprompt_text = reprompt_template.format(
                    prompt=prompt_text,
                    feedback=feedback_section,
                    solution="",
                )

                messages.append(reprompt_text)

            # Build teacher tensors for samples with feedback
            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            self_distillation_mask_list = []

            max_reprompt_len = getattr(self_distillation_cfg, "max_reprompt_len", 512)

            for i in range(batch_size):
                if messages[i] is None:
                    # No feedback: use original prompt
                    prompt_ids = batch.batch["input_ids"][i:i+1]
                    prompt_mask = batch.batch["attention_mask"][i:i+1]
                    response_ids = responses[i:i+1]
                    response_mask_i = response_mask[i:i+1]

                    teacher_input_ids = torch.cat([prompt_ids, response_ids], dim=1)
                    teacher_attention_mask = torch.cat([prompt_mask, response_mask_i], dim=1)
                    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
                    sd_mask_val = 0.0
                else:
                    # Encode the reprompted message
                    teacher_ids = self.tokenizer.encode(
                        messages[i],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_reprompt_len,
                    )
                    teacher_ids = torch.tensor([teacher_ids], dtype=torch.long, device=device)

                    prompt_ids = batch.batch["input_ids"][i:i+1]
                    response_ids = responses[i:i+1]
                    response_mask_i = response_mask[i:i+1]

                    # Teacher input: reprompt + response
                    teacher_input_ids = torch.cat([teacher_ids, response_ids], dim=1)
                    teacher_attention_mask = torch.cat([
                        torch.ones_like(teacher_ids),
                        response_mask_i,
                    ], dim=1)
                    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
                    sd_mask_val = 1.0

                teacher_input_ids_list.append(teacher_input_ids)
                teacher_attention_mask_list.append(teacher_attention_mask)
                teacher_position_ids_list.append(teacher_position_ids)
                self_distillation_mask_list.append(sd_mask_val)

            # Pad to same length
            max_len = max(t.size(1) for t in teacher_input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id or 0

            teacher_input_ids_padded = torch.full(
                (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
            )
            teacher_attention_mask_padded = torch.zeros(
                (batch_size, max_len), dtype=torch.long, device=device
            )
            teacher_position_ids_padded = torch.zeros(
                (batch_size, max_len), dtype=torch.long, device=device
            )

            for i, (ids, mask, pos) in enumerate(zip(
                teacher_input_ids_list, teacher_attention_mask_list, teacher_position_ids_list
            )):
                l = ids.size(1)
                teacher_input_ids_padded[i, :l] = ids
                teacher_attention_mask_padded[i, :l] = mask
                teacher_position_ids_padded[i, :l] = pos

            self_distillation_mask = torch.tensor(
                self_distillation_mask_list,
                dtype=torch.float32,
                device=device,
            )

            sd_batch = DataProto.from_dict(tensors={
                "teacher_input_ids": teacher_input_ids_padded,
                "teacher_attention_mask": teacher_attention_mask_padded,
                "teacher_position_ids": teacher_position_ids_padded,
                "self_distillation_mask": self_distillation_mask,
            })

            metrics["isograph/sd_batch_size"] = num_with_feedback
            metrics["isograph/sd_fraction"] = num_with_feedback / batch_size
            metrics["isograph/feedback_samples"] = num_with_feedback

        except Exception as e:
            print(f"[IsoGraph] Warning: Teacher batch building failed: {e}")
            metrics["isograph/sd_batch_error"] = str(e)

        return sd_batch, metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        """
        Override actor update to include EMA teacher update.

        After the standard PPO policy update, we update the EMA teacher model
        for the next iteration's dual-role forward pass.
        """
        actor_output = super()._update_actor(batch)

        # Update EMA teacher after policy step
        self._update_ema_teacher()

        # Log EMA update metrics
        if hasattr(actor_output, "meta_info") and "metrics" in actor_output.meta_info:
            actor_output.meta_info["metrics"]["isograph/teacher_update_count"] = self.teacher_update_count

        return actor_output

    def _update_ema_teacher(self):
        """
        Update EMA teacher model (Self-Teacher for next iteration).

        θ_teacher ← decay * θ_teacher + (1 - decay) * θ_student

        This is called after each policy update to maintain a stable
        Self-Teacher for the next iteration's dual-role forward pass.
        """
        try:
            actor_worker = self.actor_rollout_wg._workers[0]
            if not hasattr(actor_worker, "_isograph_ema_teacher"):
                # Create EMA teacher on first call
                student_model = actor_worker.module
                actor_worker._isograph_ema_teacher = deepcopy(student_model)
                for p in actor_worker._isograph_ema_teacher.parameters():
                    p.requires_grad = False
                actor_worker._isograph_ema_teacher.eval()
                print(f"[IsoGraph] EMA teacher initialized with decay={self.isograph_config.ema_decay}")

            ema_teacher = actor_worker._isograph_ema_teacher
            student_model = actor_worker.module
            decay = self.isograph_config.ema_decay

            with torch.no_grad():
                for (t_name, t_param), (s_name, s_param) in zip(
                    ema_teacher.named_parameters(),
                    student_model.named_parameters(),
                ):
                    t_param.data = decay * t_param.data + (1 - decay) * s_param.data

            self.teacher_update_count += 1

        except Exception as e:
            print(f"[IsoGraph] Warning: EMA teacher update failed: {e}")


# ============================================================================
# STEP 6: Custom TaskRunner that uses IsoGraphRayPPOTrainer
# ============================================================================

def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    """Forward to parent module-level function."""
    from verl.trainer.ppo.ray_trainer import apply_kl_penalty as _apply_kl_penalty
    return _apply_kl_penalty(data, kl_ctrl, kl_penalty)


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    """
    Override compute_advantage to inject IsoGraph pre-computed advantages.

    If batch already contains `isograph_advantages` (computed in
    _maybe_build_self_distillation_batch), we use those instead of
    recomputing GRPO/GAE advantages.
    """
    # Check if IsoGraph advantages are pre-computed
    if "isograph_advantages_computed" in data.batch:
        is_computed = data.batch["isograph_advantages_computed"].item()
        if is_computed and "isograph_advantages" in data.batch:
            advantages = data.batch.pop("isograph_advantages")
            data.batch["advantages"] = advantages
            # returns already has log π_teacher - log π_student (normalized)
            # No need for GRPO/GAE computation
            print("[IsoGraph] Using pre-computed IsoGraph advantages.")
            return data

    # Fall back to standard advantage computation
    from verl.trainer.ppo.ray_trainer import compute_advantage as _compute_advantage
    return _compute_advantage(
        data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )


# ============================================================================
# STEP 7: Main entry point (mirrors verl.trainer.main_ppo)
# ============================================================================

import hydra
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler


class IsoGraphTaskRunner:
    """
    Ray remote TaskRunner that creates IsoGraphRayPPOTrainer instead of RayPPOTrainer.

    This is the Ray-remote class that runs on a worker node. It sets up:
    1. Worker role mappings (ActorRollout, Critic, RefPolicy, RewardModel)
    2. Resource pools
    3. Datasets (train + validation)
    4. IsoGraphRayPPOTrainer with isograph_config
    5. Starts the training loop
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker. Same as TaskRunner but for IsoGraph."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # SDPO requires legacy worker implementation (teacher colocated with actor)
        if use_legacy_worker_impl == "disable":
            print("[IsoGraph] Overriding use_legacy_worker_impl=disable → enable for SDPO compatibility.")
            use_legacy_worker_impl = "enable"

        if use_legacy_worker_impl in ["auto", "enable"]:
            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker
            else:
                raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")
        else:
            raise NotImplementedError(f"use_legacy_worker_impl={use_legacy_worker_impl} not supported for SDPO")

        # For SDPO, we need a separate ref policy for KL regularization
        # (teacher is stored separately in _isograph_ema_teacher)
        actor_role = Role.ActorRolloutRef
        self.role_worker_mapping[actor_role] = ray.remote(actor_rollout_cls)
        self.mapping[actor_role] = "global_pool"
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config):
        """Add critic worker."""
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            else:
                raise NotImplementedError
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker for KL regularization."""
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward_model.enable_resource_pool:
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        return resource_pool_manager

    def run(self, config):
        """Execute IsoGraph SDPO training."""
        print(f"[IsoGraph] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        from pprint import pprint as pp
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        pp(OmegaConf.to_container(config, resolve=True))

        # ------------------------------------------------------------------
        # Worker setup (same as vanilla TaskRunner)
        # ------------------------------------------------------------------
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # ------------------------------------------------------------------
        # Model & tokenizer
        # ------------------------------------------------------------------
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # ------------------------------------------------------------------
        # Reward manager
        # ------------------------------------------------------------------
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # ------------------------------------------------------------------
        # Resource pool & datasets
        # ------------------------------------------------------------------
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ------------------------------------------------------------------
        # Extract IsoGraph config from YAML
        # ------------------------------------------------------------------
        isograph_cfg = config.actor_rollout_ref.actor.policy_loss.get("isograph", {})
        iso_sdpo_config = IsoGraphSDPOConfig(
            clip_ratio=isograph_cfg.get("clip_ratio", 0.2),
            beta=isograph_cfg.get("beta", 0.01),
            ema_decay=isograph_cfg.get("ema_decay", 0.99),
            normalize_advantage=isograph_cfg.get("normalize_advantage", True),
            loss_agg_mode=isograph_cfg.get("loss_agg_mode", "token-mean"),
            tau_node=isograph_cfg.get("tau_node", 0.8),
            tau_edge=isograph_cfg.get("tau_edge", 0.7),
            tau_order=isograph_cfg.get("tau_order", 0.6),
            fgw_alpha=isograph_cfg.get("fgw_alpha", 0.5),
        )

        oracle_graph_path = config.get("isograph", {}).get("oracle_graph_path", None)

        # ------------------------------------------------------------------
        # Create IsoGraphRayPPOTrainer
        # ------------------------------------------------------------------
        trainer = IsoGraphRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            isograph_config=iso_sdpo_config,
            oracle_graph_path=oracle_graph_path,
            use_dummy_env=True,
        )

        trainer.init_workers()
        trainer.fit()


# ============================================================================
# STEP 8: Hydra entry point
# ============================================================================

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point. Auto-detects CUDA/NPU device and runs training."""
    from verl.utils.device import auto_set_device
    auto_set_device(config)
    run_isograph_sdpo(config)


def run_isograph_sdpo(config, task_runner_class=None):
    """Initialize Ray and run IsoGraph SDPO training."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"[IsoGraph] ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(IsoGraphTaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
