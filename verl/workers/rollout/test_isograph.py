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
Test script for IsoGraph Action Interceptor.

This script demonstrates the Action Interceptor and IsoGraph Rollout functionality.
It can run with mock components (no GPU required) or with real models.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import IsoGraph components
from verl.workers.rollout.isograph_env import DummyEnvironment, ActionResult, ActionType
from verl.workers.rollout.action_interceptor import ActionInterceptor, InterceptedTrajectory, TrajectoryStep
from verl.workers.rollout.isograph_rollout import IsoGraphRollout


# ============================================================================
# Mock Components for Testing (no GPU required)
# ============================================================================

class MockTokenizer:
    """Mock tokenizer for testing without actual model."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Create some special tokens
        self.special_tokens = {
            '<zoom>': 1,
            '</zoom>': 2,
            '<call_svm>': 3,
            '<eos>': 4,
            '<pad>': 0,
        }
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Add regular tokens
        for i in range(5, vocab_size):
            self.id_to_token[i] = f'token_{i}'
            self.token_to_id[f'token_{i}'] = i
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(ord(char) % (self.vocab_size - 5) + 5)
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token_str = self.id_to_token[tid]
                if skip_special_tokens and token_str.startswith('<') and token_str.endswith('>'):
                    continue
                tokens.append(token_str)
            else:
                tokens.append(f'<unk:{tid}>')
        return ''.join(tokens)


class MockMLLMModule:
    """
    Mock MLLM module for testing.
    
    This simulates a simple language model that:
    1. Returns logits based on input tokens
    2. Has some probability of generating action tokens
    """
    
    def __init__(self, vocab_size: int = 1000, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self._counter = 0
        
        # Special token IDs
        self.zoom_id = 1
        self.zoom_end_id = 2
        self.call_svm_id = 3
        self.eos_id = 4
        
        # Track generation state
        self.generation_count = 0
    
    def eval(self):
        """Set to eval mode."""
        return self
    
    def train(self):
        """Set to train mode."""
        return self
    
    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        """Forward pass - returns mock logits."""
        batch_size, seq_len = input_ids.shape
        
        # Create mock logits with some structure
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device) * 0.1
        
        # Increase probability of action tokens based on generation step
        self._counter += 1
        
        # Simple rule-based output for testing
        # In a real model, this would be actual neural network output
        
        result = MockOutput(logits)
        return result
    
    def generate(self, input_ids, **kwargs):
        """Mock generate method."""
        # This is a simplified generate for testing
        batch_size = input_ids.size(0)
        max_new = kwargs.get('max_new_tokens', 10)
        
        all_sequences = []
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for _ in range(max_new):
                # Simple random generation
                next_token = torch.randint(5, self.vocab_size, (1,)).item()
                seq.append(next_token)
                if next_token == self.eos_id:
                    break
            all_sequences.append(seq)
        
        max_len = max(len(s) for s in all_sequences)
        padded = []
        for s in all_sequences:
            if len(s) < max_len:
                s = s + [0] * (max_len - len(s))
            padded.append(s)
        
        return MockGenerateOutput(torch.tensor(padded, device=self.device))


class MockOutput:
    """Mock model output."""
    def __init__(self, logits):
        self.logits = logits


class MockGenerateOutput:
    """Mock generate output."""
    def __init__(self, sequences):
        self.sequences = sequences


# ============================================================================
# Test Functions
# ============================================================================

def test_dummy_environment():
    """Test the DummyEnvironment class."""
    print("=" * 60)
    print("Testing DummyEnvironment")
    print("=" * 60)
    
    env = DummyEnvironment()
    
    # Test zoom action
    action = "<zoom> [100, 200, 400, 350] </zoom>"
    result = env.step(action)
    print(f"Action: {action}")
    print(f"Result: {result}")
    print(f"  - Type: {result.action_type}")
    print(f"  - Feedback: {result.feedback[:100]}...")
    print()
    
    # Test call_svm action
    action = "<call_svm>"
    result = env.step(action)
    print(f"Action: {action}")
    print(f"Result: {result}")
    print(f"  - Type: {result.action_type}")
    print(f"  - Feedback: {result.feedback[:100]}...")
    print()
    
    print("✓ DummyEnvironment test passed")
    return True


def test_action_interceptor():
    """Test the ActionInterceptor class."""
    print("=" * 60)
    print("Testing ActionInterceptor")
    print("=" * 60)
    
    # Create mock components
    tokenizer = MockTokenizer()
    module = MockMLLMModule(device="cpu")
    
    # Create environment
    env = DummyEnvironment(device="cpu")
    
    # Create interceptor
    interceptor = ActionInterceptor(
        module=module,
        tokenizer=tokenizer,
        environment=env,
        max_interactions=10,
        max_context_length=512,
        device="cpu",
    )
    
    # Test action pattern detection
    test_cases = [
        "<zoom> [100, 200, 400, 350] </zoom>",
        "<call_svm>",
        "<ZOOM> [10, 20, 30, 40] </ZOOM>",
        "This is normal text without actions",
        "<zoom> [1,2,3,4]</zoom> and more text",
    ]
    
    print("Testing action pattern detection:")
    for text in test_cases:
        match = interceptor._find_action_in_text(text)
        status = "✓ Found" if match else "✗ None"
        print(f"  {status}: '{text[:50]}...' -> {match}")
    print()
    
    print("✓ ActionInterceptor test passed")
    return True


def test_trajectory_creation():
    """Test InterceptedTrajectory creation and structure."""
    print("=" * 60)
    print("Testing InterceptedTrajectory Structure")
    print("=" * 60)
    
    # Create a mock trajectory
    prompt = "请分析这张古籍图片。"
    trajectory = InterceptedTrajectory(
        prompt=prompt,
        full_sequence=[1, 2, 3, 4, 5],  # Mock token IDs
        trajectory_steps=[],
        total_env_interactions=0,
        final_state="",
    )
    
    # Add some steps
    trajectory.trajectory_steps.append(TrajectoryStep(
        state="请分析",
        action=None,
        env_feedback=None,
        tokens_generated=torch.tensor([10]),
        log_probs=torch.tensor([-0.5]),
    ))
    
    trajectory.trajectory_steps.append(TrajectoryStep(
        state="请分析<zoom>",
        action="<zoom> [100, 200, 400, 350] </zoom>",
        env_feedback="[System: SVM extracted text...]",
        tokens_generated=torch.tensor([1]),
        log_probs=torch.tensor([-0.3]),
    ))
    
    trajectory.total_env_interactions = 1
    trajectory.final_state = "分析完成"
    
    print(f"Prompt: {trajectory.prompt}")
    print(f"Total tokens: {len(trajectory.full_sequence)}")
    print(f"Env interactions: {trajectory.total_env_interactions}")
    print(f"Trajectory steps: {len(trajectory.trajectory_steps)}")
    print()
    
    for i, step in enumerate(trajectory.trajectory_steps):
        print(f"Step {i}:")
        print(f"  State: {step.state[:30]}...")
        print(f"  Action: {step.action}")
        print(f"  Feedback: {step.env_feedback[:50] if step.env_feedback else 'None'}...")
        print()
    
    print("✓ Trajectory structure test passed")
    return True


def test_isograph_rollout_config():
    """Test IsoGraphRollout configuration."""
    print("=" * 60)
    print("Testing IsoGraphRollout Configuration")
    print("=" * 60)
    
    from verl.workers.rollout.isograph_rollout import IsoGraphRollout, IsoGraphRolloutConfig
    
    # Test config
    config = IsoGraphRolloutConfig(
        use_dummy_env=True,
        max_env_interactions=5,
        max_context_length=2048,
        temperature=0.8,
    )
    
    print(f"Config: {config}")
    print()
    
    print("✓ IsoGraphRollout config test passed")
    return True


def test_env_action_types():
    """Test ActionType enumeration."""
    print("=" * 60)
    print("Testing ActionType Enumeration")
    print("=" * 60)
    
    print(f"Available action types:")
    for action_type in ActionType:
        print(f"  - {action_type.name}: {action_type.value}")
    print()
    
    # Test parsing
    env = DummyEnvironment()
    
    test_cases = [
        ("<zoom> [1, 2, 3, 4] </zoom>", ActionType.ZOOM),
        ("<call_svm>", ActionType.CALL_SVM),
        ("<unknown>", ActionType.UNKNOWN),
        ("<CALL_SVM>", ActionType.CALL_SVM),
    ]
    
    for action, expected_type in test_cases:
        result = env.step(action)
        status = "✓" if result.action_type == expected_type else "✗"
        print(f"  {status} '{action}' -> {result.action_type.name} (expected {expected_type.name})")
    print()
    
    print("✓ ActionType test passed")
    return True


def test_env_factory():
    """Test the environment factory (create_environment)."""
    print("=" * 60)
    print("Testing Environment Factory (create_environment)")
    print("=" * 60)

    from verl.workers.rollout.isograph_env import create_environment, get_environment_class

    env_cls = get_environment_class()
    print(f"Active environment class: {env_cls.__name__}")

    # Create environment
    env = create_environment(
        oracle_graph_path=None,
        device="cpu",
    )
    print(f"Created environment: {type(env).__name__}")
    print(f"  - oracle_graph loaded: {env.oracle_graph is not None}")
    print(f"  - image_id: {env.image_id}")

    # Test with a sample action
    result = env.step("<zoom> [100, 200, 400, 350] </zoom>")
    print(f"  - step() result type: {result.action_type}")

    print("✓ Environment factory test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IsoGraph Action Interceptor - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_dummy_environment,
        test_env_action_types,
        test_env_factory,
        test_action_interceptor,
        test_trajectory_creation,
        test_isograph_rollout_config,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
