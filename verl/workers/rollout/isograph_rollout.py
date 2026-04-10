# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
IsoGraph Rollout - Intercepted rollout with VE-MDP environment interaction.

This module provides the IsoGraphRollout class that integrates the Action Interceptor
with the verl rollout system, enabling MLLMs to interact with the VE-MDP environment
during autoregressive generation.

Key features:
1. Intercepts special action tokens (<zoom>, <call_svm>) during generation
2. Suspends generation to call environment.step() for visual evidence feedback
3. Appends environment feedback to context and resumes generation
4. Collects complete trajectories with (state, action, env_feedback) alternation
5. Computes log probabilities for all tokens including those generated after env feedback

Environment: Automatically selects Member C's IsoGraphEnvironment (production VE-MDP)
when available and isograph.use_dummy_env=false; otherwise uses DummyEnvironment.
"""

import contextlib
import json
import os

import torch
import torch.distributed
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig

from .base import BaseRollout
from .isograph_env import create_environment, get_environment_class

try:
    from .action_interceptor import ActionInterceptor, InterceptedTrajectory
except ImportError:
    ActionInterceptor = None
    InterceptedTrajectory = None


__all__ = ["IsoGraphRollout"]


@dataclass
class IsoGraphRolloutConfig:
    """Configuration for IsoGraph Rollout."""
    # Environment settings
    use_dummy_env: bool = True
    max_env_interactions: int = 10
    # Environment construction kwargs
    oracle_graph_path: Optional[str] = None
    image_path: Optional[str] = None
    # Member C: SVM backend ("dummy" or "onnx")
    svm_backend: str = "dummy"
    svm_model_path: Optional[str] = None

    # Generation settings
    max_context_length: int = 4096
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    do_sample: bool = True

    # Trajectory collection
    collect_trajectory: bool = True
    collect_log_probs: bool = True

    # Device
    device: str = "cuda"


class IsoGraphRollout(BaseRollout):
    """
    IsoGraph Rollout with Action Interception and VE-MDP Environment Interaction.
    
    This rollout class wraps the standard MLLM generation process to enable
    interactive environment feedback during autoregressive generation.
    
    Key workflow:
    1. Start with prompt tokens
    2. Autoregressive generation: sample next token, append to sequence
    3. Check for special action tokens (<zoom>, <call_svm>)
    4. If action detected:
       - Suspend generation
       - Call env.step(action) to get visual evidence feedback
       - Append feedback tokens to context
       - Resume generation
    5. Continue until EOS or max_new_tokens reached
    6. Return full sequence and collected trajectory
    
    The collected trajectory contains:
    - All tokens (prompt + generated + env feedback)
    - Action tokens and their corresponding env feedback
    - Log probabilities for all action tokens
    
    In the full IsoGraph framework, this would integrate with:
    - VE-MDP (Visual-Evidence Markov Decision Process) for image interaction
    - DGR (Diagnostic Graph Report) for rich text diagnostic feedback
    """
    
    def __init__(
        self,
        module: nn.Module,
        config: Any,
        tokenizer: Any = None,
        model_config: Any = None,
        device_mesh: Any = None,
        environment: Optional[Any] = None,
        use_dummy_env: Optional[bool] = None,
        oracle_graph_path: Optional[str] = None,
        image_path: Optional[str] = None,
        svm_backend: str = "dummy",
        svm_model_path: Optional[str] = None,
        oracle_graph_dir: Optional[str] = None,
    ):
        """
        Initialize IsoGraph Rollout.

        Args:
            module: The MLLM module (must support HF-style forward)
            config: Configuration object with rollout settings
            tokenizer: HuggingFace tokenizer
            model_config: HFModelConfig (optional, for BaseRollout compatibility)
            device_mesh: DeviceMesh (optional)
            environment: Pre-constructed environment instance (overrides factory).
                When provided, use_dummy_env / oracle_graph_path / svm_backend
                are ignored.
            use_dummy_env: Force DummyEnvironment (True) or Member C's IsoGraphEnvironment (False).
                Defaults to config.isograph.use_dummy_env if available.
            oracle_graph_path: Path to oracle graph JSON
            image_path: Path to source image (Member C only)
            svm_backend: "dummy" or "onnx" (Member C only)
            svm_model_path: Path to ONNX SVM model (Member C only)
            oracle_graph_dir: Directory containing page_*.json oracle graphs (Member B data)
        """
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = (
            omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
            if model_config is not None else None
        )
        self.device_mesh = device_mesh
        self.module = module
        self.tokenizer = tokenizer

        # Determine whether to use DummyEnvironment or Member C's IsoGraphEnvironment
        if environment is not None:
            self.environment = environment
        elif use_dummy_env is True:
            # Explicitly force DummyEnvironment
            self.environment = create_environment(
                oracle_graph_path=oracle_graph_path or self.config.get("oracle_graph_path"),
                device=str(next(module.parameters()).device),
            )
        elif use_dummy_env is False:
            # Explicitly force Member C's IsoGraphEnvironment
            self.environment = create_environment(
                oracle_graph_path=oracle_graph_path or self.config.get("oracle_graph_path"),
                image_path=image_path or self.config.get("image_path"),
                device=str(next(module.parameters()).device),
                svm_backend=svm_backend,
                svm_model_path=svm_model_path,
            )
        else:
            # "auto": use config value or default to Member C if available
            cfg_dummy = self.config.get("use_dummy_env", True)
            self.environment = create_environment(
                oracle_graph_path=oracle_graph_path or self.config.get("oracle_graph_path"),
                image_path=image_path or self.config.get("image_path"),
                device=str(next(module.parameters()).device),
                svm_backend=svm_backend if not cfg_dummy else "dummy",
                svm_model_path=svm_model_path,
            )

        print(
            f"[IsoGraphRollout] Using environment: {type(self.environment).__name__} "
            f"(oracle_graph_path={oracle_graph_path or self.config.get('oracle_graph_path')})"
        )

        if ActionInterceptor is not None:
            self.interceptor = ActionInterceptor(
                module=module,
                tokenizer=tokenizer,
                environment=self.environment,
                max_interactions=self.config.get("max_env_interactions", 10),
                max_context_length=self.config.get("max_context_length", 4096),
                device=str(next(module.parameters()).device),
            )
        else:
            self.interceptor = None
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using standard HF generate, with tokenization if needed.

        Compatible with both old-style (pre-tokenized input_ids) and new-style
        (raw_prompt messages only) DataProto from verl's data pipeline.
        """
        has_input_ids = prompts.batch is not None and "input_ids" in prompts.batch.keys()

        if not has_input_ids:
            prompts = self._tokenize_raw_prompts(prompts)

        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def _tokenize_raw_prompts(self, prompts: DataProto) -> DataProto:
        """Tokenize raw_prompt messages into input_ids / attention_mask / position_ids.

        For VLMs like Qwen2.5-VL that use 3D mrope position_ids, constructs
        the correct (3, batch_size, seq_len) tensor.
        """
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            raise ValueError(
                "IsoGraphRollout received data without input_ids or raw_prompt. "
                "Ensure the dataset provides at least one of these fields."
            )

        all_input_ids = []
        all_attention_masks = []
        max_len = self.config.get("prompt_length", 4096)

        for raw in raw_prompts:
            messages = raw if isinstance(raw, list) else json.loads(raw)
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer(
                text, return_tensors="pt", max_length=max_len, truncation=True, padding=False,
            )
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_masks.append(encoded["attention_mask"].squeeze(0))

        max_prompt_len = max(ids.size(0) for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_ids, padded_masks = [], []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_prompt_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
                mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            padded_ids.append(ids)
            padded_masks.append(mask)

        device = next(self.module.parameters()).device
        batch_ids = torch.stack(padded_ids).to(device)
        batch_masks = torch.stack(padded_masks).to(device)

        pos_1d = batch_masks.long().cumsum(-1) - 1
        pos_1d.masked_fill_(batch_masks == 0, 0)

        model_type = getattr(self.module.config, "model_type", "")
        self._is_vlm_mrope = model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe")
        if self._is_vlm_mrope:
            position_ids = pos_1d.unsqueeze(1).expand(-1, 4, -1).contiguous()
        else:
            position_ids = pos_1d

        new_batch = TensorDict(
            {"input_ids": batch_ids, "attention_mask": batch_masks, "position_ids": position_ids},
            batch_size=batch_ids.size(0),
        )
        return DataProto(batch=new_batch, non_tensor_batch=prompts.non_tensor_batch, meta_info=prompts.meta_info)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        """Single-minibatch generation using HF model.generate(), same logic as HFRollout."""
        do_sample = prompts.meta_info.get("do_sample", self.config.get("do_sample", True))
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get("temperature", self.config.get("temperature", 1.0))
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if not do_sample:
            kwargs = {"do_sample": False, "num_beams": 1}
        elif is_validate:
            kwargs = {
                "do_sample": True, "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            kwargs = {
                "do_sample": True, "num_beams": 1,
                "top_p": top_p, "top_k": top_k,
                "temperature": temperature, "num_return_sequences": 1,
            }

        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        has_position_ids = "position_ids" in prompts.batch.keys()
        position_ids = prompts.batch["position_ids"] if has_position_ids else None

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        if has_position_ids and position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()

        gen_kwargs = dict(
            input_ids=idx,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=response_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=generation_config,
            output_scores=False,
            return_dict_in_generate=True,
            use_cache=True,
        )
        if has_position_ids:
            gen_kwargs["position_ids"] = position_ids

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(**gen_kwargs)

        seq = output.sequences
        generated_batch_size = seq.size(0)

        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            delta_tokens = pad_token_id * torch.ones(
                size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype
            )
            seq = torch.cat((seq, delta_tokens), dim=1)
        elif delta_length < 0:
            seq = seq[:, :sequence_length]

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]
        resp_len = response.size(1)

        if has_position_ids:
            if position_ids.dim() == 3:
                last_pos = position_ids[:, :, -1:]
                delta = torch.arange(1, resp_len + 1, device=position_ids.device).view(1, 1, -1)
                delta = delta.expand(position_ids.size(0), position_ids.size(1), -1)
                response_position_ids = last_pos + delta
                position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                position_ids = position_ids.transpose(0, 1).contiguous()
            else:
                delta_position_id = torch.arange(1, resp_len + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
                response_position_ids = position_ids[:, -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        td = {
            "prompts": prompt,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
        }
        if has_position_ids:
            td["position_ids"] = position_ids

        batch = TensorDict(td, batch_size=generated_batch_size)

        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch)
    
    async def update_weights(self, weights: Any, **kwargs) -> None:
        """Update rollout weights from actor (colocated mode, weights are shared)."""
        pass

    async def release(self, **kwargs) -> None:
        """Release resources to free up VRAM for actor training."""
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def resume(self, **kwargs) -> None:
        """Resume rollout (e.g., re-allocate KV cache buffers if they were released)."""
        pass
