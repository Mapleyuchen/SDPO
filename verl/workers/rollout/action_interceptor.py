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
Action Interceptor for IsoGraph (Active-Symbolic SDPO) Framework.

This module implements the core interception mechanism for VE-MDP interaction:
1. Monitors MLLM autoregressive generation for special action tokens
2. Suspends generation when <zoom> or <call_svm> tokens are detected
3. Calls environment.step() to get visual evidence feedback
4. Resumes generation with environment feedback appended to context
"""

import re
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .isograph_env import DummyEnvironment, ActionResult, ActionType


class InterceptState(Enum):
    """States during interceptive generation."""
    GENERATING = "generating"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    TERMINATED = "terminated"


@dataclass
class TrajectoryStep:
    """Single step in the trajectory during interceptive rollout."""
    state: str  # Current context (partial generated text)
    action: Optional[str]  # Action taken (if any)
    env_feedback: Optional[str]  # Environment feedback received
    tokens_generated: torch.Tensor  # Tokens generated at this step
    log_probs: Optional[torch.Tensor]  # Log probs for these tokens


@dataclass
class InterceptedTrajectory:
    """Complete trajectory collected during interceptive rollout."""
    prompt: str
    full_sequence: List[int]  # All tokens including env feedback
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    total_env_interactions: int = 0
    final_state: str = ""


class ActionInterceptor:
    """
    Core Action Interceptor for IsoGraph VE-MDP interaction.
    
    This class wraps the MLLM generation process to:
    1. Detect special action tokens (<zoom>, <call_svm>) during generation
    2. Suspend generation and call environment for feedback
    3. Inject feedback into the context and resume generation
    4. Collect complete trajectories with (state, action, env_feedback) alternation
    
    Usage:
        interceptor = ActionInterceptor(module, tokenizer, env)
        trajectory = interceptor.generate_with_interaction(prompt_ids, ...)
    """
    
    # Special tokens to intercept
    ACTION_TOKENS = ['<zoom>', '<call_svm>']
    ACTION_END_TOKENS = ['</zoom>']
    
    def __init__(
        self,
        module: torch.nn.Module,
        tokenizer: Any,
        environment: Optional[DummyEnvironment] = None,
        max_interactions: int = 10,
        max_context_length: int = 4096,
        device: str = "cuda",
    ):
        """
        Initialize Action Interceptor.
        
        Args:
            module: The MLLM module (must support HF-style forward: input_ids, attention_mask, position_ids)
            tokenizer: HuggingFace tokenizer for decoding
            environment: Environment for action execution (DummyEnvironment or VE-MDP)
            max_interactions: Maximum number of env interactions per trajectory
            max_context_length: Maximum context length to prevent OOM
            device: Device for tensor operations
        """
        self.module = module
        self.tokenizer = tokenizer
        self.environment = environment or DummyEnvironment(device=device)
        self.max_interactions = max_interactions
        self.max_context_length = max_context_length
        self.device = device
        
        # Pre-compile action patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for action detection."""
        self.zoom_pattern = re.compile(r'<zoom>\s*\[.*?\]\s*</zoom>', re.IGNORECASE)
        self.call_svm_pattern = re.compile(r'<call_svm>', re.IGNORECASE)
        self.action_pattern = re.compile(
            r'(<zoom>\s*\[.*?\]\s*</zoom>|<call_svm>)', 
            re.IGNORECASE
        )
    
    def _find_action_in_text(self, text: str) -> Optional[str]:
        """Find action token in generated text."""
        match = self.action_pattern.search(text)
        return match.group(1) if match else None
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    @torch.no_grad()
    def _single_step_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform single-step forward pass and sample next token.
        
        Args:
            input_ids: (batch_size, seq_len) input token IDs
            attention_mask: (batch_size, seq_len) attention mask
            position_ids: (batch_size, seq_len) position IDs
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            
        Returns:
            Tuple of (next_token_ids, log_probs)
                next_token_ids: (batch_size, 1) sampled token IDs
                log_probs: (batch_size, 1) log probabilities of sampled tokens
        """
        # Forward pass
        output = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        # Get logits for last position
        logits = output.logits[:, -1, :]  # (batch_size, vocab_size)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample or greedy
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        
        # Compute log prob of sampled token
        log_prob = torch.log(probs.gather(1, next_token) + 1e-8)
        
        return next_token, log_prob
    
    @torch.no_grad()
    def _decode_to_text(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        # Handle batch dimension
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
    
    @torch.no_grad()
    def _get_token_id(self, token_str: str) -> Optional[int]:
        """Get token ID for a special token string."""
        tokens = self.tokenizer.encode(token_str, add_special_tokens=False)
        return tokens[0] if tokens else None
    
    def generate_with_interaction(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        eos_token_ids: List[int],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
        collect_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, InterceptedTrajectory]:
        """
        Generate sequence with action interception and environment interaction.
        
        This is the main entry point for interceptive rollout. It performs autoregressive
        generation with suspension when special action tokens are detected.
        
        Args:
            prompt_ids: (batch_size, prompt_len) input prompt token IDs
            attention_mask: (batch_size, prompt_len) attention mask
            position_ids: (batch_size, prompt_len) position IDs
            eos_token_ids: List of EOS token IDs to stop generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to sample (True) or use greedy (False)
            collect_trajectory: Whether to collect full trajectory with env interactions
            
        Returns:
            Tuple of (full_sequence, trajectory)
                full_sequence: Complete generated sequence including env feedback tokens
                trajectory: InterceptedTrajectory with all steps and interactions
        """
        self.module.eval()
        
        # Initialize sequence state
        current_ids = prompt_ids.clone()
        current_mask = attention_mask.clone()
        current_positions = position_ids.clone()
        
        prompt_length = prompt_ids.size(1)
        interaction_count = 0
        trajectory = InterceptedTrajectory(
            prompt=self._decode_to_text(prompt_ids),
            full_sequence=current_ids.squeeze(0).tolist(),
        )
        
        # Generation loop
        for step in range(max_new_tokens):
            # Check context length limit
            if current_ids.size(1) >= self.max_context_length:
                break
            
            # Single forward step
            next_token, log_prob = self._single_step_forward(
                current_ids, current_mask, current_positions,
                temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample
            )
            
            # Decode the new token
            new_token_text = self._decode_to_text(next_token)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = F.pad(current_mask, (0, 1), value=1)
            current_positions = torch.cat([
                current_positions, 
                current_positions[:, -1:] + 1
            ], dim=1)
            
            # Check for EOS
            is_eos = any(next_token.item() == eos_id for eos_id in eos_token_ids)
            if is_eos:
                if collect_trajectory:
                    trajectory.trajectory_steps.append(TrajectoryStep(
                        state=self._decode_to_text(current_ids),
                        action=None,
                        env_feedback=None,
                        tokens_generated=next_token,
                        log_probs=log_prob,
                    ))
                break
            
            # Check if we need to intercept (accumulate partial action tokens)
            if collect_trajectory:
                partial_text = self._decode_to_text(current_ids)
                
                # Check if full action detected
                action_match = self._find_action_in_text(partial_text)
                
                if action_match:
                    # Suspend generation and call environment
                    action_result = self.environment.step(action_match)
                    interaction_count += 1
                    
                    # Record trajectory step with action and feedback
                    trajectory.trajectory_steps.append(TrajectoryStep(
                        state=partial_text,
                        action=action_match,
                        env_feedback=action_result.feedback,
                        tokens_generated=next_token,
                        log_probs=log_prob,
                    ))
                    
                    # Append env feedback to context
                    feedback_tokens = self._tokenize_text(action_result.feedback)
                    
                    if feedback_tokens:
                        feedback_tensor = torch.tensor(
                            [feedback_tokens], 
                            device=self.device, 
                            dtype=current_ids.dtype
                        )
                        current_ids = torch.cat([current_ids, feedback_tensor], dim=1)
                        
                        # Update attention mask
                        new_mask = torch.ones(
                            (1, len(feedback_tokens)), 
                            device=self.device, 
                            dtype=current_mask.dtype
                        )
                        current_mask = torch.cat([current_mask, new_mask], dim=1)
                        
                        # Update position IDs
                        feedback_positions = torch.arange(
                            current_positions[:, -1:].item() + 1,
                            current_positions[:, -1:].item() + 1 + len(feedback_tokens),
                            device=self.device
                        ).unsqueeze(0)
                        current_positions = torch.cat([current_positions, feedback_positions], dim=1)
                    
                    # Check interaction limit
                    if interaction_count >= self.max_interactions:
                        break
                else:
                    # Normal generation step
                    trajectory.trajectory_steps.append(TrajectoryStep(
                        state=partial_text,
                        action=None,
                        env_feedback=None,
                        tokens_generated=next_token,
                        log_probs=log_prob,
                    ))
        
        # Complete trajectory
        trajectory.full_sequence = current_ids.squeeze(0).tolist()
        trajectory.total_env_interactions = interaction_count
        trajectory.final_state = self._decode_to_text(current_ids)
        
        self.module.train()
        
        return current_ids, trajectory
    
    def generate_batch_with_interaction(
        self,
        prompts: "DataProto",  # type: ignore # Forward reference
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[List[torch.Tensor], List[InterceptedTrajectory]]:
        """
        Batch generation with action interception.
        
        Args:
            prompts: DataProto containing batch of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to sample
            
        Returns:
            Tuple of (sequences, trajectories)
                sequences: List of generated sequences (one per prompt)
                trajectories: List of InterceptedTrajectory objects
        """
        from verl import DataProto
        
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_ids = prompts.meta_info["eos_token_id"]
        
        batch_size = idx.size(0)
        
        sequences = []
        trajectories = []
        
        for i in range(batch_size):
            prompt_i = idx[i:i+1]
            mask_i = attention_mask[i:i+1]
            pos_i = position_ids[i:i+1]
            
            seq, traj = self.generate_with_interaction(
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
            
            sequences.append(seq.squeeze(0))
            trajectories.append(traj)
        
        return sequences, trajectories
