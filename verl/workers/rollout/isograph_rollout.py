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
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig

from .base import BaseRollout
from .action_interceptor import ActionInterceptor, InterceptedTrajectory
from .isograph_env import DummyEnvironment


__all__ = ["IsoGraphRollout"]


@dataclass
class IsoGraphRolloutConfig:
    """Configuration for IsoGraph Rollout."""
    # Environment settings
    use_dummy_env: bool = True
    max_env_interactions: int = 10
    
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
        environment: Optional[DummyEnvironment] = None,
    ):
        """
        Initialize IsoGraph Rollout.

        Args:
            module: The MLLM module (must support HF-style forward)
            config: Configuration object with rollout settings (RolloutConfig or dict)
            tokenizer: HuggingFace tokenizer for tokenization (optional; lazy-loads from module if None)
            model_config: HFModelConfig (optional, for BaseRollout compatibility)
            device_mesh: DeviceMesh (optional, for BaseRollout compatibility)
            environment: Environment for action execution (DummyEnvironment or VE-MDP)
        """
        # Store attributes directly instead of calling BaseRollout.__init__
        # (BaseRollout.__init__ requires config/model_config/device_mesh but
        # HFRollout also bypasses it, so we follow the same pattern.)
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = (
            omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
            if model_config is not None else None
        )
        self.device_mesh = device_mesh
        self.module = module
        self.tokenizer = tokenizer
        
        # Create environment if not provided
        if environment is None:
            device = next(module.parameters()).device
            self.environment = DummyEnvironment(device=str(device))
        else:
            self.environment = environment
        
        # Create action interceptor
        self.interceptor = ActionInterceptor(
            module=module,
            tokenizer=tokenizer,
            environment=self.environment,
            max_interactions=self.config.get("max_env_interactions", 10),
            max_context_length=self.config.get("max_context_length", 4096),
            device=str(next(module.parameters()).device),
        )
        
        # Store generation config
        self.generation_config = {
            "temperature": self.config.get("temperature", 1.0),
            "top_k": self.config.get("top_k"),
            "top_p": self.config.get("top_p", 1.0),
            "do_sample": self.config.get("do_sample", True),
        }
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with action interception and environment interaction.
        
        This is the main entry point for the rollout system. It generates sequences
        by autoregressive sampling with interception of special action tokens.
        
        Args:
            prompts: DataProto containing:
                - batch["input_ids"]: (batch_size, prompt_len) input token IDs
                - batch["attention_mask"]: (batch_size, prompt_len) attention mask
                - batch["position_ids"]: (batch_size, prompt_len) position IDs
                - meta_info["eos_token_id"]: List of EOS token IDs
                - meta_info["pad_token_id"]: Pad token ID
                - Optional: meta_info["temperature"], meta_info["top_k"], etc.
        
        Returns:
            DataProto containing:
                - batch["input_ids"]: Full sequence tokens (prompt + generated + env feedback)
                - batch["responses"]: Only generated tokens (excluding prompt)
                - batch["sequences"]: Alias for full sequence
                - batch["old_log_probs"]: Log probabilities for response tokens
                - batch["attention_mask"]: Updated attention mask
                - batch["position_ids"]: Updated position IDs
                - batch["num_env_interactions"]: Number of env interactions per sample
                - batch["trajectory"]: (optional) Full trajectory data
                - meta_info["env_interactions"]: Total env interactions in batch
        """
        self.module.eval()
        
        # Extract input data
        idx = prompts.batch["input_ids"]  # (batch_size, prompt_len)
        attention_mask = prompts.batch["attention_mask"]  # (batch_size, prompt_len)
        position_ids = prompts.batch["position_ids"]  # (batch_size, prompt_len)
        
        # Get token IDs
        eos_token_ids = prompts.meta_info["eos_token_id"]
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        pad_token_id = prompts.meta_info.get("pad_token_id", 0)
        
        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        
        # Override generation config from prompts meta_info if provided
        temperature = prompts.meta_info.get("temperature", self.generation_config["temperature"])
        top_k = prompts.meta_info.get("top_k", self.generation_config["top_k"])
        top_p = prompts.meta_info.get("top_p", self.generation_config["top_p"])
        do_sample = prompts.meta_info.get("do_sample", self.generation_config["do_sample"])
        max_new_tokens = prompts.meta_info.get("max_new_tokens", self.config.response_length)
        
        # Collect results for batch
        all_sequences = []
        all_trajectories = []
        all_num_interactions = []
        all_log_probs = []
        
        # Process each sample in batch
        # NOTE: For efficiency, this processes samples one by one.
        # For true batch processing with variable-length interaction,
        # a more sophisticated implementation with padding would be needed.
        for i in range(batch_size):
            prompt_i = idx[i:i+1]  # (1, prompt_len)
            mask_i = attention_mask[i:i+1]  # (1, prompt_len)
            pos_i = position_ids[i:i+1]  # (1, prompt_len)
            
            # Generate with interception
            sequence, trajectory = self.interceptor.generate_with_interaction(
                prompt_ids=prompt_i,
                attention_mask=mask_i,
                position_ids=pos_i,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                collect_trajectory=True,
            )
            
            # Extract response (tokens after prompt)
            response = sequence[0, prompt_length:]
            response_length = response.size(0)
            
            # Compute log probs for response tokens
            # We need to re-run forward to get logits for all response positions
            if self.config.get("collect_log_probs", True):
                log_probs = self._compute_response_log_probs(
                    prompt_i, response, mask_i, pos_i, temperature, top_k, top_p, do_sample
                )
            else:
                log_probs = torch.zeros((1, response_length), device=sequence.device)
            
            all_sequences.append(sequence[0])
            all_trajectories.append(trajectory)
            all_num_interactions.append(trajectory.total_env_interactions)
            all_log_probs.append(log_probs[0])
        
        # Pad sequences to same length
        max_seq_len = max(s.size(0) for s in all_sequences)
        
        # Stack sequences and pad
        padded_sequences = []
        padded_masks = []
        padded_positions = []
        padded_log_probs = []
        
        for seq in all_sequences:
            seq_len = seq.size(0)
            padding_len = max_seq_len - seq_len
            
            if padding_len > 0:
                pad_tokens = torch.full((padding_len,), pad_token_id, device=seq.device, dtype=seq.dtype)
                padded_seq = torch.cat([seq, pad_tokens])
                
                pad_mask = torch.zeros(padding_len, device=seq.device, dtype=torch.long)
                padded_mask = torch.cat([torch.ones(seq_len, device=seq.device, dtype=torch.long), pad_mask])
                
                # For position IDs, we need to handle this more carefully
                # Just pad with the last position + offset
                last_pos = position_ids[0, -1].item()
                pad_positions = torch.arange(
                    last_pos + 1, last_pos + 1 + padding_len, device=seq.device
                )
                padded_pos = torch.cat([position_ids[0][:seq_len] if seq_len <= position_ids[0].size(0) 
                                       else torch.arange(seq_len, device=seq.device), 
                                       pad_positions])
            else:
                padded_seq = seq
                padded_mask = torch.ones(seq_len, device=seq.device, dtype=torch.long)
                padded_pos = position_ids[0][:seq_len] if seq_len <= position_ids[0].size(0) else torch.arange(seq_len, device=seq.device)
            
            padded_sequences.append(padded_seq)
            padded_masks.append(padded_mask)
            padded_positions.append(padded_pos)
        
        # Concatenate batch
        full_sequences = torch.stack(padded_sequences)  # (batch_size, max_seq_len)
        full_masks = torch.stack(padded_masks)
        full_positions = torch.stack(padded_positions)

        # Compute response_length from the first sample's sequence.
        # Since env feedback makes lengths variable, we use the max_seq_len
        # and treat everything after prompt_length as the response.
        response_length = max_seq_len - prompt_length  # may be 0 for some samples

        # Extract response-only portion for each sample, padded to same response_length
        padded_responses = []
        for seq in all_sequences:
            seq_len = seq.size(0)
            resp = seq[prompt_length:]  # response portion (may be shorter than response_length)
            pad_len = response_length - resp.size(0)
            if pad_len > 0:
                resp = torch.cat([resp, torch.full((pad_len,), pad_token_id, device=resp.device, dtype=resp.dtype)])
            padded_responses.append(resp)
        full_responses = torch.stack(padded_responses)  # (batch_size, response_length)

        # Extend attention_mask: original prompt mask + response mask (1 for each response token)
        # This matches hf_rollout's convention: response_mask is embedded in attention_mask
        padded_response_masks = []
        for i in range(batch_size):
            seq = all_sequences[i]
            seq_len = seq.size(0)
            resp_mask = torch.ones(response_length, device=seq.device, dtype=full_masks.dtype)
            if seq_len < max_seq_len:
                # This sample is shorter than max_seq_len: pad attention mask too
                extra_pad = max_seq_len - seq_len
                extra_resp_pad = max(0, response_length - (seq_len - prompt_length))
                attn_pad = torch.zeros(extra_pad, device=seq.device, dtype=full_masks.dtype)
            else:
                attn_pad = None
            padded_response_masks.append(resp_mask)
        full_response_masks = torch.stack(padded_response_masks)  # (batch_size, response_length)

        # Get response log probs (already computed per sample, pad to response_length)
        padded_log_probs = []
        for lp in all_log_probs:
            lp_len = lp.size(0)
            pad_len = response_length - lp_len
            if pad_len > 0:
                lp = torch.cat([lp, torch.zeros(pad_len, device=lp.device, dtype=lp.dtype)])
            padded_log_probs.append(lp)
        full_log_probs = torch.stack(padded_log_probs)  # (batch_size, response_length)

        # Build response_mask: [batch, max_seq_len] with 1 for response tokens, 0 for prompt/padding.
        # This is used by _build_isograph_teacher_batch to identify the response portion.
        full_response_mask = torch.zeros((batch_size, max_seq_len), device=full_sequences.device, dtype=torch.long)
        for i, seq in enumerate(all_sequences):
            seq_len = seq.size(0)
            if seq_len > prompt_length:
                full_response_mask[i, prompt_length:seq_len] = 1

        # Prepare batch output.
        # NOTE: We do set "response_mask" because _build_isograph_teacher_batch needs it
        # to identify the response region in the teacher input sequence.
        batch = TensorDict(
            {
                "prompts": idx,                         # (batch, prompt_length)
                "input_ids": full_sequences,             # (batch, max_seq_len) full seq
                "responses": full_responses,             # (batch, response_length) response-only
                "sequences": full_sequences,            # alias for full seq
                "old_log_probs": full_log_probs,         # (batch, response_length)
                "attention_mask": full_masks,            # (batch, max_seq_len)
                "position_ids": full_positions,          # (batch, max_seq_len)
                "response_mask": full_response_mask,     # (batch, max_seq_len) 1 for response
                "num_env_interactions": torch.tensor(all_num_interactions, device=full_sequences.device),
            },
            batch_size=batch_size,
        )
        
        # Store trajectories in non_tensor_batch for potential downstream use
        non_tensor_batch = {
            "trajectories": all_trajectories,
        }
        
        # Create output DataProto
        output = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "env_interactions": sum(all_num_interactions),
                "batch_env_interactions": all_num_interactions,
            }
        )
        
        self.module.train()
        
        return output
    
    @torch.no_grad()
    def _compute_response_log_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Compute log probabilities for response tokens.
        
        This runs a forward pass through the model and computes log probs
        for each token in the response.
        
        Args:
            prompt_ids: (1, prompt_len) prompt token IDs
            response_ids: (response_len,) response token IDs
            attention_mask: (1, prompt_len) attention mask
            position_ids: (1, prompt_len) position IDs
            temperature, top_k, top_p, do_sample: Sampling parameters
            
        Returns:
            log_probs: (1, response_len) log probabilities for each response token
        """
        response_len = response_ids.size(0)
        
        # Concatenate prompt and response for forward pass
        # We need to compute log probs for each position
        full_ids = torch.cat([prompt_ids, response_ids.unsqueeze(0)], dim=1)
        
        # Extend attention mask and position ids
        full_mask = F.pad(attention_mask, (0, response_len), value=1)
        
        # Extend position ids
        last_pos = position_ids[0, -1].item()
        extra_positions = torch.arange(
            last_pos + 1, last_pos + 1 + response_len,
            device=position_ids.device
        ).unsqueeze(0)
        full_positions = torch.cat([position_ids, extra_positions], dim=1)
        
        # Forward pass
        output = self.module(
            input_ids=full_ids,
            attention_mask=full_mask,
            position_ids=full_positions,
        )
        
        # Get logits
        logits = output.logits  # (1, prompt_len + response_len, vocab_size)
        
        # Compute log probs for response tokens (shift by 1 for prediction target)
        # Logits at position t predict token at position t+1
        prompt_len = prompt_ids.size(1)
        response_logits = logits[0, prompt_len - 1:prompt_len + response_len - 1, :]  # (response_len, vocab_size)
        response_tokens = response_ids  # (response_len,)
        
        # Compute log probs
        log_probs = logprobs_from_logits(
            logits=response_logits.unsqueeze(0),
            labels=response_tokens.unsqueeze(0)
        )  # (1, response_len)
        
        return log_probs
    
    def update_environment(self, environment: DummyEnvironment):
        """Update the environment used for action execution."""
        self.environment = environment
        self.interceptor.environment = environment
    
    def get_trajectories(self, output: DataProto) -> List[InterceptedTrajectory]:
        """Extract trajectories from rollout output."""
        return output.non_tensor_batch.get("trajectories", [])
    
    def get_trajectory_stats(self, output: DataProto) -> Dict[str, Any]:
        """Get statistics about the trajectories in the rollout output."""
        trajectories = self.get_trajectories(output)
        
        if not trajectories:
            return {}
        
        total_interactions = sum(t.total_env_interactions for t in trajectories)
        avg_interactions = total_interactions / len(trajectories) if trajectories else 0
        
        total_tokens = [len(t.full_sequence) for t in trajectories]
        avg_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0
        
        return {
            "num_samples": len(trajectories),
            "total_env_interactions": total_interactions,
            "avg_env_interactions_per_sample": avg_interactions,
            "avg_total_tokens": avg_tokens,
            "min_tokens": min(total_tokens) if total_tokens else 0,
            "max_tokens": max(total_tokens) if total_tokens else 0,
        }
