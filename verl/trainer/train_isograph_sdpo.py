#!/usr/bin/env python3
"""
IsoGraph SDPO — End-to-End Training Script (verl-based).

Integration: Members A + B + C

This script provides a complete end-to-end training entry point for the
IsoGraph (Active-Symbolic SDPO) framework, built on top of verl's RayPPOTrainer.

Architecture (four stages):
  Stage 1 — Active Exploration (VE-MDP):
    MLLM generates <zoom>/<call_svm> → env.step() → local graph G_l

  Stage 2 — Mathematical Dissection (FGW Verification):
    G_l vs. oracle G_g via Fused Gromov-Wasserstein optimal transport → S_node, S_edge, S_order

  Stage 3 — Semantic Grounding (DGR):
    FGW scores → Diagnostic Graph Report f_DGR (natural language critique)

  Stage 4 — Gradient Update (SDPO):
    Student π_θ (blind) vs. Self-Teacher π_θ' (with f_DGR, no_grad, EMA)
    → Token-level advantage A_t → PPO clipped loss - β·KL(ref)
    → Policy update without Critic network

Usage:
  # Member C's real VE-MDP environment (recommended for full integration):
  ISOGRAPH_C_ROOT=/path/to/ISOGraph-C \
      bash examples/isograph_trainer/run_isograph_sdpo.sh

  # Dummy environment (no Member C dependency):
  bash examples/isograph_trainer/run_isograph_sdpo.sh

  # With Member B's data (data-B/page_*.json):
  ISOGRAPH_C_ROOT=/path/to/ISOGraph-C \
  ISOGRAPH_ORACLE_GRAPH_DIR=/path/to/data-B \
      bash examples/isograph_trainer/run_isograph_sdpo.sh
"""

import os
import sys
import math
import glob
import json
from copy import deepcopy
from collections import defaultdict
from typing import Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import ray

from omegaconf import OmegaConf

# Ensure verl and SDPO are on the path
SDPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.path.join(SDPO_ROOT, "verl")
sys.path.insert(0, SDPO_ROOT)
sys.path.insert(0, VERL_ROOT)

# ============================================================================
# STEP 1: Register IsoGraph SDPO policy loss with verl's registry
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
from verl.utils.torch_functional import masked_mean as verl_masked_mean
from verl.utils.debug import marked_timer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
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

# Environment factory: selects Member C's IsoGraphEnvironment when available
# and isograph.use_dummy_env=false, otherwise falls back to DummyEnvironment.
from verl.workers.rollout.isograph_env import (
    create_environment,
    get_environment_class,
    ActionType,
)

from verl.utils.dataset.rl_dataset import collate_fn

# ============================================================================
# STEP 4: IsoGraphRayPPOTrainer
# ============================================================================


class IsoGraphRayPPOTrainer(RayPPOTrainer):
    """
    Extended RayPPOTrainer for IsoGraph SDPO.

    Key modifications vs. vanilla RayPPOTrainer:
    1. Overrides _maybe_build_self_distillation_batch to inject DGR feedback
       from the VE-MDP environment AND compute IsoGraph token-level
       advantages via the dual-role forward pass.
    2. Uses loss_mode="isograph" to activate compute_policy_loss_isograph.
    3. Tracks EMA teacher update metrics.
    4. Supports per-sample oracle graphs from Member B's data directory.
    """

    def __init__(
        self,
        *args,
        isograph_config: Optional[IsoGraphSDPOConfig] = None,
        oracle_graph_path: Optional[str] = None,
        oracle_graph_dir: Optional[str] = None,
        use_dummy_env: bool = True,
        isograph_env=None,
        **kwargs,
    ):
        """
        Initialize IsoGraph trainer.

        Args:
            isograph_config: Hyperparameters for IsoGraph SDPO.
            oracle_graph_path: Path to single oracle graph JSON.
            oracle_graph_dir: Path to directory containing Member B's
                page_*.json oracle graphs. When set, loads oracle graphs
                per-sample by matching image_id in the parquet data.
            use_dummy_env: Whether to use DummyEnvironment (True) or
                Member C's IsoGraphEnvironment (False).
            isograph_env: Pre-constructed environment instance. When provided,
                overrides oracle_graph_path / use_dummy_env.
        """
        super().__init__(*args, **kwargs)

        self.isograph_config = isograph_config or IsoGraphSDPOConfig()
        self.oracle_graph_path = oracle_graph_path
        self.oracle_graph_dir = oracle_graph_dir
        self.use_dummy_env = use_dummy_env

        # ---- Build per-sample oracle graph index ----
        # Map: image_id -> oracle_graph_dict
        self._oracle_graph_index: dict[str, dict] = {}
        self._index_oracle_graphs()

        # ---- Create environment ----
        if isograph_env is not None:
            # External environment provided (e.g., constructed with Member C's class)
            self.isograph_env = isograph_env
            self.use_dummy_env = False
            print(f"[IsoGraph] Using provided environment: {type(isograph_env).__name__}")
        elif self.use_dummy_env:
            oracle_path = self.oracle_graph_path or os.path.join(
                SDPO_ROOT, "global_oracle_graph_demo.json"
            )
            self.isograph_env = create_environment(
                oracle_graph_path=oracle_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"[IsoGraph] DummyEnvironment initialized with oracle: {oracle_path}")
        else:
            # Member C's production IsoGraphEnvironment
            oracle_path = self.oracle_graph_path or self._default_oracle_path()
            self.isograph_env = create_environment(
                oracle_graph_path=oracle_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                svm_backend="dummy",  # or "onnx" when real SVM is available
            )
            print(
                f"[IsoGraph] Member C IsoGraphEnvironment initialized with "
                f"oracle: {oracle_path} (backend: {get_environment_class().__name__})"
            )

        # EMA teacher state
        self.teacher_update_count = 0
        self.isograph_metrics: dict = defaultdict(list)

    def init_workers(self):
        """Override: run parent init but force sync mode for HF rollout."""
        rollout_name = self.config.actor_rollout_ref.rollout.name
        if rollout_name == "hf":
            self._init_workers_sync()
        else:
            super().init_workers()

    def _init_workers_sync(self):
        """Full worker init copied from parent, but with async_rollout_mode=False."""
        from verl.single_controller.ray.base import create_colocated_worker_cls
        from verl.workers.config import FSDPEngineConfig

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = False
        print("[IsoGraph] HF rollout: using sync mode (AgentLoopManager skipped)")

    def _default_oracle_path(self) -> str:
        """Return a default oracle graph path."""
        return os.path.join(SDPO_ROOT, "global_oracle_graph_demo.json")

    def _index_oracle_graphs(self) -> None:
        """
        Load all oracle graphs from oracle_graph_dir (Member B's data directory)
        into an in-memory index keyed by image_id.

        This enables per-sample DGR feedback generation when training on
        Member B's data-B/page_*.json files.
        """
        if not self.oracle_graph_dir:
            return

        if not os.path.isdir(self.oracle_graph_dir):
            print(f"[IsoGraph] oracle_graph_dir not found: {self.oracle_graph_dir}")
            return

        pattern = os.path.join(self.oracle_graph_dir, "*.json")
        files = glob.glob(pattern)

        if not files:
            print(f"[IsoGraph] No JSON files found in {self.oracle_graph_dir}")
            return

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                image_id = data.get("image_id", os.path.basename(filepath))
                oracle_graph = data.get("oracle_graph", data)

                self._oracle_graph_index[image_id] = oracle_graph
            except Exception as e:
                print(f"[IsoGraph] Warning: Failed to load oracle graph {filepath}: {e}")

        print(
            f"[IsoGraph] Indexed {len(self._oracle_graph_index)} oracle graphs "
            f"from {self.oracle_graph_dir}"
        )

    def _resolve_oracle_graph(self, sample: Any) -> Optional[dict]:
        """
        Resolve the oracle graph for a single training sample.

        Priority:
          1. image_id in sample metadata → lookup in _oracle_graph_index
          2. oracle_graph field in sample → use directly (Member B embedded format)
          3. Single oracle_graph_path → use for all samples (demo mode)
        """
        # Priority 1: index lookup by image_id
        if hasattr(sample, "image_id") and sample.image_id:
            image_id = sample.image_id
            if image_id in self._oracle_graph_index:
                return self._oracle_graph_index[image_id]

        # Priority 2: embedded oracle_graph
        if hasattr(sample, "oracle_graph") and sample.oracle_graph:
            return sample.oracle_graph

        # Priority 3: single global oracle
        if self.oracle_graph_path and os.path.exists(self.oracle_graph_path):
            with open(self.oracle_graph_path, "r", encoding="utf-8") as f:
                return json.load(f).get("oracle_graph", {})

        return None

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
          - Builds a reprompted teacher batch

        IsoGraph SDPO additions:
          - Extracts/constructs DGR feedback from the VE-MDP environment
          - Computes IsoGraph token-level advantages via dual-role forward pass
          - Injects advantages directly into batch.batch["advantages"]

        The flow in ray_trainer.fit() after this call:
            self_distillation_batch, self_dist_metrics = _maybe_build_self_distillation_batch(...)
            batch = compute_advantage(batch, ...)  ← reads isograph_advantages_computed; skips GRPO
            actor_output = self._update_actor(batch)  ← dp_actor uses teacher forward + IsoGraph loss
        """
        # IsoGraph SDPO core does NOT require use_kl_loss=true.
        # When use_kl_loss=false, the reference model is NOT loaded (saves GPU memory).
        # The IsoGraph SDPO advantage A_t = log π_teacher - log π_student is still computed.
        # The KL penalty term -β·KL(π_θ||π_ref) is simply skipped (ref_log_probs=None
        # is handled gracefully by compute_policy_loss_isograph at line 1318).
        # This allows training on single-GPU machines with limited VRAM (e.g. 62 GB RTX A6000).
        use_kl_loss = self.config.actor_rollout_ref.actor.get("use_kl_loss", False)
        if not use_kl_loss:
            print(
                "[IsoGraph] WARNING: use_kl_loss=false. "
                "Reference model will NOT be loaded (saves ~16 GB GPU memory). "
                "KL penalty term (-β·KL) is skipped. "
                "Training will use only the IsoGraph SDPO advantage (Teacher - Student)."
            )

        # ---- Compute IsoGraph token-level advantages ----
        iso_metrics = self._compute_isograph_advantages(batch)

        # ---- Get DGR feedback from VE-MDP environment ----
        dgr_feedback = self._get_dgr_feedback(batch, reward_extra_infos_dict)

        # ---- Build self-distillation batch with DGR context ----
        sd_batch, sd_metrics = self._build_isograph_teacher_batch(batch, dgr_feedback)

        all_metrics = {**iso_metrics, **sd_metrics}

        # CRITICAL: Inject IsoGraph advantages so they take precedence over GRPO
        if "isograph_advantages" in batch.batch:
            batch.batch["advantages"] = batch.batch["isograph_advantages"]
            all_metrics["isograph/advantages_injected"] = True

        if sd_batch is not None:
            return sd_batch, all_metrics

        # When no DGR feedback is available, create dummy teacher tensors
        # (copy of student with self_distillation_mask=0) to satisfy dp_actor assertions.
        device = batch.batch["input_ids"].device
        if "position_ids" in batch.batch and batch.batch["position_ids"].dim() == 3:
            teacher_pos_ids = batch.batch["position_ids"].clone()
        else:
            # Create 3D mrope-format position_ids from attention_mask
            pos_1d = compute_position_id_with_mask(batch.batch["attention_mask"])
            teacher_pos_ids = pos_1d.unsqueeze(1).expand(-1, 4, -1)
        dummy_sd = DataProto.from_dict(tensors={
            "teacher_input_ids": batch.batch["input_ids"].clone(),
            "teacher_attention_mask": batch.batch["attention_mask"].clone(),
            "teacher_position_ids": teacher_pos_ids,
            "self_distillation_mask": torch.zeros(batch.batch["input_ids"].shape[0], device=device),
        })
        return dummy_sd, all_metrics

    def _compute_isograph_advantages(self, batch: DataProto) -> dict:
        """
        Set up IsoGraph advantages placeholder for the SDPO loss computation.

        In IsoGraph SDPO, the true token-level advantages are computed inside
        ``compute_policy_loss_isograph`` (dp_actor.py) via the dual forward
        pass: student on original input, teacher on DGR-augmented input.

        Here we just:
          1. Verify loss_mode == "isograph"
          2. Initialize ``batch.batch["isograph_advantages"]`` as zeros
          3. Set the flag so ``compute_advantage()`` skips GRPO

        The real advantage A_t = log π_teacher(a|s,DGR) - log π_student(a|s)
        is computed inside the SDPO loss function itself.
        """
        metrics = {}

        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "isograph":
            raise ValueError(
                f"[IsoGraph] FATAL: loss_mode is '{loss_mode}' but "
                f"IsoGraphRayPPOTrainer is active. "
                f"Set: actor_rollout_ref.actor.policy_loss.loss_mode=isograph"
            )

        try:
            sequences = batch.batch["sequences"]
            response_mask = batch.batch.get("response_mask")
            if response_mask is None:
                response_mask = compute_response_mask(batch)

            device = sequences.device
            batch_size = sequences.shape[0]
            resp_len = response_mask.shape[1] if response_mask is not None else 1

            # Initialize zero advantages (real computation in dp_actor SDPO loss)
            if "isograph_advantages" not in batch.batch:
                batch.batch["isograph_advantages"] = torch.zeros(
                    batch_size, resp_len, device=device
                )
            batch.batch["isograph_advantages_computed"] = torch.tensor(True, device=device)

        except Exception as e:
            metrics["isograph/adv_error"] = str(e)
            print(f"[IsoGraph] Warning: advantage setup failed: {e}")

        return metrics

    def _get_dgr_feedback(
        self,
        batch: DataProto,
        reward_extra_infos_dict: Optional[dict] = None,
    ) -> list:
        """
        Generate DGR (Diagnostic Graph Report) feedback for each sample in batch.

        Member C's IsoGraphEnvironment runs FGW optimal transport between
        the model's predicted local graph (parsed from its text response)
        and the oracle graph, then produces a natural-language critique (DGR).

        Oracle graph resolution priority:
          1. ``_oracle_graph_index[image_id]``  (loaded from oracle_graph_dir)
          2. ``reward_model.ground_truth`` embedded in parquet
          3. ``self.isograph_env.oracle_graph`` (global fallback)

        Args:
            batch: Current training batch
            reward_extra_infos_dict: Extra reward info from rollout

        Returns:
            List of DGR strings (one per sample). Never returns None entries
            when the environment is available — worst case returns a DGR
            describing a fully-missing prediction.
        """
        batch_size = (
            batch.batch.batch_size[0] if hasattr(batch.batch, "batch_size")
            else batch.batch["sequences"].shape[0]
        )

        if self.isograph_env is None:
            return [None] * batch_size

        # ---- Decode model responses ----
        response_texts = self._decode_responses(batch)

        # ---- Resolve per-sample oracle graphs ----
        ntb = batch.non_tensor_batch if hasattr(batch, "non_tensor_batch") else {}

        # Debug: print available keys once
        if not hasattr(self, "_dgr_debug_printed"):
            self._dgr_debug_printed = True
            print(f"[IsoGraph DEBUG] non_tensor_batch keys: {list(ntb.keys()) if ntb else 'NONE'}")
            if ntb:
                for k in list(ntb.keys())[:6]:
                    v = ntb[k]
                    if isinstance(v, (list, tuple)):
                        sample = v[0] if len(v) > 0 else "EMPTY"
                        print(f"  {k}: type={type(v).__name__}, len={len(v)}, sample[0] type={type(sample).__name__}, val={str(sample)[:120]}")
                    elif hasattr(v, '__len__'):
                        sample = v[0] if len(v) > 0 else "EMPTY"
                        print(f"  {k}: type={type(v).__name__}, len={len(v)}, sample[0] type={type(sample).__name__}, val={str(sample)[:120]}")
                    else:
                        print(f"  {k}: type={type(v).__name__}, value={str(v)[:120]}")

        # image_id is NOT in non_tensor_batch (verl doesn't propagate custom columns)
        # We extract it from reward_model.ground_truth instead
        image_ids = []

        # reward_model column contains ground_truth
        reward_models_raw = ntb.get("reward_model", [])
        # May be numpy array, list, or single value
        if hasattr(reward_models_raw, 'tolist'):
            reward_models = reward_models_raw.tolist()
        elif isinstance(reward_models_raw, (list, tuple)):
            reward_models = list(reward_models_raw)
        else:
            reward_models = [reward_models_raw] * batch_size

        # Pre-parse oracle graphs and extract image_ids
        oracle_graphs_cache = []
        scenarios_cache = []
        for rm in reward_models:
            og = None
            iid = None
            scenario = None
            if isinstance(rm, dict):
                gt_raw = rm.get("ground_truth")
                if gt_raw:
                    try:
                        gt_data = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
                        og = gt_data.get("oracle_graph", gt_data)
                        iid = gt_data.get("image_id")
                        scenario = gt_data.get("adversarial_scenario") or gt_data.get("hallucination_type")
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
            oracle_graphs_cache.append(og)
            image_ids.append(iid)
            scenarios_cache.append(scenario)

        dgr_list = []

        for i in range(batch_size):
            try:
                # ---- Resolve oracle graph for this sample ----
                oracle_graph = None

                # Priority 1: index lookup by image_id (extracted from ground_truth)
                iid = image_ids[i] if i < len(image_ids) else None
                if iid and iid in self._oracle_graph_index:
                    oracle_graph = self._oracle_graph_index[iid]

                # Priority 2: use cached oracle from reward_model.ground_truth
                if oracle_graph is None and i < len(oracle_graphs_cache):
                    oracle_graph = oracle_graphs_cache[i]

                # ---- Get response text ----
                resp_text = response_texts[i] if i < len(response_texts) else ""

                # ---- Get image_id and scenario for DGR report ----
                img_id = image_ids[i] if i < len(image_ids) else "sample"
                scenario = scenarios_cache[i] if i < len(scenarios_cache) else None

                # ---- Call Member C's convenience method ----
                dgr = self.isograph_env.get_dgr_feedback_for_response(
                    response_text=resp_text,
                    oracle_graph=oracle_graph,
                    image_id=img_id,
                    scenario=scenario,
                )
                dgr_list.append(dgr)

            except Exception as e:
                print(f"[IsoGraph] Warning: DGR feedback generation failed for sample {i}: {e}")
                dgr_list.append(None)

        num_valid = sum(1 for d in dgr_list if d is not None)
        if num_valid > 0:
            # Print first DGR for monitoring
            first_valid = next(d for d in dgr_list if d is not None)
            preview = first_valid[:200].replace("\n", " ")
            print(f"[IsoGraph] DGR feedback: {num_valid}/{batch_size} samples. "
                  f"Preview: {preview}...")

        return dgr_list

    def _decode_responses(self, batch: DataProto) -> list[str]:
        """Decode model response tokens back to text strings.

        Uses ``batch.batch["responses"]`` if available (padded response-only
        tokens), otherwise extracts the response portion from
        ``batch.batch["sequences"]`` using ``attention_mask``.
        """
        try:
            if "responses" in batch.batch:
                response_ids = batch.batch["responses"]
            elif "sequences" in batch.batch and "attention_mask" in batch.batch:
                sequences = batch.batch["sequences"]
                attention_mask = batch.batch["attention_mask"]
                # Response starts where prompt ends
                prompt_lens = attention_mask.sum(dim=1).long()
                # Get response portion
                response_ids = []
                for idx in range(sequences.shape[0]):
                    pl = prompt_lens[idx].item()
                    resp = sequences[idx, pl:]
                    response_ids.append(resp)
                # Pad to same length
                max_len = max(r.shape[0] for r in response_ids)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = torch.full((len(response_ids), max_len), pad_id,
                                    dtype=torch.long, device=sequences.device)
                for idx, r in enumerate(response_ids):
                    padded[idx, :r.shape[0]] = r
                response_ids = padded
            else:
                return [""] * (batch.batch["input_ids"].shape[0]
                               if "input_ids" in batch.batch else 1)

            texts = []
            for idx in range(response_ids.shape[0]):
                tokens = response_ids[idx]
                # Remove padding
                if self.tokenizer.pad_token_id is not None:
                    tokens = tokens[tokens != self.tokenizer.pad_token_id]
                # Remove EOS
                if self.tokenizer.eos_token_id is not None:
                    tokens = tokens[tokens != self.tokenizer.eos_token_id]
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                texts.append(text)
            return texts
        except Exception as e:
            print(f"[IsoGraph] Warning: Response decoding failed: {e}")
            bs = batch.batch["sequences"].shape[0] if "sequences" in batch.batch else 1
            return [""] * bs

    def _build_isograph_teacher_batch(
        self,
        batch: DataProto,
        dgr_feedback: list,
    ) -> tuple[Optional[DataProto], dict]:
        """
        Build the self-distillation teacher batch for IsoGraph SDPO.

        Teacher input structure: [DGR_prompt | response]
        Teacher attention mask: [1s for DGR_prompt | response_mask]
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
            batch_size = (
                batch.batch.batch_size[0] if hasattr(batch.batch, "batch_size")
                else responses.shape[0]
            )
            student_seq_len = batch.batch["input_ids"].shape[1]
            response_length = responses.shape[1]

            # Filter to samples that have DGR feedback
            num_with_feedback = sum(1 for fb in dgr_feedback if fb is not None)
            if num_with_feedback == 0:
                return None, metrics

            feedback_template = getattr(
                self_distillation_cfg, "feedback_template",
                "\n\n[Diagnostic Feedback]\n{fb}\n\n"
            )
            reprompt_template = getattr(
                self_distillation_cfg, "reprompt_template",
                "{prompt}\n\n{feedback}"
            )

            messages = []
            for i in range(batch_size):
                fb = dgr_feedback[i] if i < len(dgr_feedback) else None
                if fb is None:
                    messages.append(None)
                    continue

                # Config uses {feedback_raw}, code fallback uses {fb}
                try:
                    feedback_section = feedback_template.format(feedback_raw=fb, fb=fb)
                except (KeyError, IndexError):
                    feedback_section = f"\n\n[Diagnostic Feedback]\n{fb}\n\n"

                # Get original prompt text from non_tensor_batch
                raw_prompt = [[""] * batch_size]
                if hasattr(batch, "non_tensor_batch"):
                    raw_prompt = batch.non_tensor_batch.get("raw_prompt", raw_prompt)
                prompt_text = ""
                if i < len(raw_prompt) and raw_prompt[i]:
                    if isinstance(raw_prompt[i], list) and len(raw_prompt[i]) > 0:
                        last_item = raw_prompt[i][-1]
                        prompt_text = last_item.get("content", str(last_item)) \
                            if isinstance(last_item, dict) else str(last_item)
                    else:
                        prompt_text = str(raw_prompt[i]) if raw_prompt[i] else ""

                reprompt_text = reprompt_template.format(
                    prompt=prompt_text,
                    feedback=feedback_section,
                    solution="",
                )
                messages.append(reprompt_text)

            # Build teacher tensors
            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            self_distillation_mask_list = []

            max_reprompt_len = getattr(self_distillation_cfg, "max_reprompt_len", 512)
            teacher_prompt_max_len = max(1, max_reprompt_len - response_length)
            pad_token_id = self.tokenizer.pad_token_id or 0

            for i in range(batch_size):
                if messages[i] is None:
                    student_input = batch.batch["input_ids"][i]
                    student_mask = batch.batch["attention_mask"][i]
                    teacher_input_ids_list.append(student_input.unsqueeze(0))
                    teacher_attention_mask_list.append(student_mask.unsqueeze(0))
                    # Use student's position_ids if available (already 3D mrope)
                    if "position_ids" in batch.batch and batch.batch["position_ids"].dim() == 3:
                        teacher_position_ids_list.append(
                            batch.batch["position_ids"][i].unsqueeze(0)
                        )
                    else:
                        pos_1d = compute_position_id_with_mask(student_mask.unsqueeze(0))
                        pos_3d = pos_1d.unsqueeze(1).expand(-1, 4, -1)
                        teacher_position_ids_list.append(pos_3d)
                    self_distillation_mask_list.append(0.0)
                    continue

                # Encode DGR-prompted message
                teacher_ids = self.tokenizer.encode(
                    messages[i],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=teacher_prompt_max_len,
                )
                teacher_ids = torch.tensor(teacher_ids, dtype=torch.long, device=device)

                # Teacher input: [DGR_prompt | response]
                response_ids = responses[i]
                resp_valid_mask = (
                    response_mask[i, -response_length:]
                    if response_mask is not None
                    else torch.ones(response_length, device=device, dtype=torch.long)
                )
                teacher_input = torch.cat([teacher_ids, response_ids], dim=0)
                teacher_mask = torch.cat([
                    torch.ones_like(teacher_ids),
                    resp_valid_mask,
                ], dim=0)

                # LEFT-PAD to student_seq_len so that response stays at the END
                # (dp_actor extracts log_probs from the last response_length positions)
                teacher_input_len = teacher_input.shape[0]
                if teacher_input_len < student_seq_len:
                    pad_len = student_seq_len - teacher_input_len
                    teacher_input = torch.cat([
                        torch.full((pad_len,), pad_token_id, device=device, dtype=teacher_input.dtype),
                        teacher_input,
                    ])
                    teacher_mask = torch.cat([
                        torch.zeros(pad_len, device=device, dtype=teacher_mask.dtype),
                        teacher_mask,
                    ])
                elif teacher_input_len > student_seq_len:
                    # Truncate from the LEFT (keep response at end)
                    excess = teacher_input_len - student_seq_len
                    teacher_input = teacher_input[excess:]
                    teacher_mask = teacher_mask[excess:]

                teacher_input_ids_list.append(teacher_input.unsqueeze(0))
                teacher_attention_mask_list.append(teacher_mask.unsqueeze(0))
                # Create 3D mrope position_ids (bsz=1, 4, seqlen) for Qwen2.5-VL
                # For text-only teacher input (no vision tokens), all 4 channels
                # (text, temporal, height, width) are identical sequential positions.
                pos_1d = compute_position_id_with_mask(teacher_mask.unsqueeze(0))  # (1, seqlen)
                pos_3d = pos_1d.unsqueeze(1).expand(-1, 4, -1)  # (1, 4, seqlen)
                teacher_position_ids_list.append(pos_3d)
                self_distillation_mask_list.append(1.0)

            teacher_input_ids = torch.cat(teacher_input_ids_list, dim=0)
            teacher_attention_mask = torch.cat(teacher_attention_mask_list, dim=0)
            teacher_position_ids = torch.cat(teacher_position_ids_list, dim=0)
            self_distillation_mask = torch.tensor(
                self_distillation_mask_list,
                dtype=torch.float32,
                device=device,
            )

            sd_batch = DataProto.from_dict(tensors={
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": teacher_attention_mask,
                "teacher_position_ids": teacher_position_ids,
                "self_distillation_mask": self_distillation_mask,
            })

            metrics["isograph/sd_batch_size"] = num_with_feedback
            metrics["isograph/sd_fraction"] = num_with_feedback / batch_size

        except Exception as e:
            print(f"[IsoGraph] Warning: Teacher batch building failed: {e}")
            metrics["isograph/sd_batch_error"] = str(e)

        return sd_batch, metrics

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Override actor update to include EMA teacher update."""
        actor_output = super()._update_actor(batch)
        self._update_ema_teacher()

        if hasattr(actor_output, "meta_info") and "metrics" in actor_output.meta_info:
            actor_output.meta_info["metrics"]["isograph/teacher_update_count"] = (
                self.teacher_update_count
            )

        return actor_output

    def _update_ema_teacher(self):
        """
        Update EMA teacher model (Self-Teacher for next iteration).

        θ_teacher ← decay * θ_teacher + (1 - decay) * θ_student
        """
        try:
            actor_worker = self.actor_rollout_wg._workers[0]

            if not hasattr(actor_worker, "_isograph_ema_teacher"):
                student_model = actor_worker.module
                actor_worker._isograph_ema_teacher = deepcopy(student_model)
                for p in actor_worker._isograph_ema_teacher.parameters():
                    p.requires_grad = False
                actor_worker._isograph_ema_teacher.eval()
                print(
                    f"[IsoGraph] EMA teacher initialized (decay={self.isograph_config.ema_decay})"
                )

            ema_teacher = actor_worker._isograph_ema_teacher
            student_model = actor_worker.module
            decay = self.isograph_config.ema_decay

            with torch.no_grad():
                for (t_name, t_param), (s_name, s_param) in zip(
                    ema_teacher.named_parameters(),
                    student_model.named_parameters(),
                ):
                    t_param.data = decay * t_param.data + (1 - decay) * s_param.data

            # Wire EMA teacher to actor_worker's teacher_module
            actor_worker.teacher_module = ema_teacher

            self.teacher_update_count += 1

        except Exception as e:
            print(f"[IsoGraph] Warning: EMA teacher update failed: {e}")


# ============================================================================
# STEP 5: Custom TaskRunner
# ============================================================================

def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    """Forward to parent module-level function."""
    from verl.trainer.ppo.ray_trainer import apply_kl_penalty as _apply_kl_penalty
    return _apply_kl_penalty(data, kl_ctrl, kl_penalty)


# ============================================================================
# STEP 6: Hydra entry point
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
    Ray remote TaskRunner that creates IsoGraphRayPPOTrainer.

    Sets up:
    1. Worker role mappings (ActorRollout, Critic, RefPolicy, RewardModel)
    2. Resource pools
    3. Datasets (train + validation) from Member B's data directory
    4. IsoGraphRayPPOTrainer with isograph_config
    5. Starts the training loop
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy == "disable":
            use_legacy = "enable"  # SDPO requires legacy implementation

        if use_legacy in ["auto", "enable"]:
            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker
            else:
                raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")
        else:
            raise NotImplementedError(f"use_legacy_worker_impl={use_legacy} not supported for SDPO")

        actor_role = Role.ActorRolloutRef
        self.role_worker_mapping[actor_role] = ray.remote(actor_rollout_cls)
        self.mapping[actor_role] = "global_pool"
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config):
        """Add critic worker."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
            self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
            self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        else:
            raise NotImplementedError

        self.mapping[Role.Critic] = "global_pool"

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = (
                "reward_pool" if config.reward_model.enable_resource_pool else "global_pool"
            )

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
            return
        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward_model.enable_resource_pool:
            resource_pool_spec["reward_pool"] = (
                [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            )

        return ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

    def run(self, config):
        """Execute IsoGraph SDPO training."""
        import socket
        print(f"[IsoGraph] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        from pprint import pprint as pp
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)
        pp(OmegaConf.to_container(config, resolve=True))

        # Worker setup
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Model & tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Reward manager
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Resource pool & datasets
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

        # Extract IsoGraph config
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

        # IsoGraph environment configuration
        isograph_config = config.get("isograph", {})
        oracle_graph_path = isograph_config.get("oracle_graph_path", None)
        oracle_graph_dir = isograph_config.get("oracle_graph_dir", None)
        use_dummy_env = isograph_config.get("use_dummy_env", True)

        print(
            f"[IsoGraph] Config: oracle_graph_path={oracle_graph_path}, "
            f"oracle_graph_dir={oracle_graph_dir}, use_dummy_env={use_dummy_env}"
        )

        # Create trainer
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
            oracle_graph_dir=oracle_graph_dir,
            use_dummy_env=use_dummy_env,
        )

        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point."""
    auto_set_device(config)
    run_isograph_sdpo(config)


def run_isograph_sdpo(config, task_runner_class=None):
    """Initialize Ray and run IsoGraph SDPO training."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()

        pythonpath = os.environ.get("PYTHONPATH", "")
        if pythonpath:
            default_runtime_env.setdefault("env_vars", {})["PYTHONPATH"] = pythonpath

        isograph_c_root = os.environ.get("ISOGRAPH_C_ROOT", "")
        if isograph_c_root:
            default_runtime_env["env_vars"]["ISOGRAPH_C_ROOT"] = isograph_c_root

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
