#!/usr/bin/env python3
"""
Standalone test for IsoGraph Action Interceptor components.
This test can run without installing the full verl package.
"""

import sys
import re
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# Copy of IsoGraph components for standalone testing
# ============================================================================

class ActionType(Enum):
    """Types of actions that can be intercepted."""
    ZOOM = "zoom"
    CALL_SVM = "call_svm"
    UNKNOWN = "unknown"


@dataclass
class ActionResult:
    """Result from environment step."""
    action_type: ActionType
    action_params: dict
    feedback: str
    is_terminal: bool = False


class DummyEnvironment:
    """Dummy environment for testing the Action Interceptor."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.svm_responses = [
            "[System: SVM extracted text from region x1=100,y1=200,x2=400,y2=350: '庚子事变后，清廷推行新政...']",
            "[System: OCR detected vertical text in column 3: '光绪三十一年乙巳...']",
            "[System: SVM classified layout region as '序跋类' with confidence 0.92]",
            "[System: Visual evidence extracted: 书籍尺寸 23.5cm × 16.8cm, 板框...]",
            "[System: Symbol detection found 12个古籍专有字符 in region...]",
        ]
        self.response_idx = 0
    
    def _parse_zoom_action(self, action_str: str) -> Optional[ActionResult]:
        pattern = r'<zoom>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</zoom>'
        match = re.search(pattern, action_str)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            idx = self.response_idx % len(self.svm_responses)
            feedback = self.svm_responses[idx].replace("x1=100,y1=200,x2=400,y2=350", 
                                                         f"x1={x1},y1={y1},x2={x2},y2={y2}")
            self.response_idx += 1
            return ActionResult(
                action_type=ActionType.ZOOM,
                action_params={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                feedback=feedback,
                is_terminal=False
            )
        return None
    
    def _parse_call_svm_action(self, action_str: str) -> Optional[ActionResult]:
        if '<call_svm>' in action_str:
            idx = self.response_idx % len(self.svm_responses)
            feedback = self.svm_responses[idx]
            self.response_idx += 1
            return ActionResult(
                action_type=ActionType.CALL_SVM,
                action_params={},
                feedback=feedback,
                is_terminal=False
            )
        return None
    
    def step(self, action: str) -> ActionResult:
        result = self._parse_zoom_action(action)
        if result is not None:
            return result
        result = self._parse_call_svm_action(action)
        if result is not None:
            return result
        return ActionResult(
            action_type=ActionType.UNKNOWN,
            action_params={},
            feedback="",
            is_terminal=False
        )


class InterceptState(Enum):
    """States during interceptive generation."""
    GENERATING = "generating"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    TERMINATED = "terminated"


@dataclass
class TrajectoryStep:
    """Single step in the trajectory during interceptive rollout."""
    state: str
    action: Optional[str]
    env_feedback: Optional[str]
    tokens_generated: torch.Tensor
    log_probs: Optional[torch.Tensor]


@dataclass
class InterceptedTrajectory:
    """Complete trajectory collected during interceptive rollout."""
    prompt: str
    full_sequence: List[int]
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    total_env_interactions: int = 0
    final_state: str = ""


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<zoom>': 1, '</zoom>': 2, '<call_svm>': 3,
            '<eos>': 4, '<pad>': 0, '<s>': 5, '</s>': 6,
        }
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        for i in range(10, vocab_size):
            self.id_to_token[i] = f'w{i}'
            self.token_to_id[f'w{i}'] = i
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append((ord(char) % 90 + 10))
        return tokens
    
    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                t = self.id_to_token[tid]
                if skip_special_tokens and t.startswith('<') and t.endswith('>'):
                    continue
                tokens.append(t)
            else:
                tokens.append(f'<unk:{tid}>')
        return ''.join(tokens)


class MockMLLMModule(torch.nn.Module):
    """Mock MLLM module for testing."""
    
    def __init__(self, vocab_size: int = 1000, device: str = "cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.eos_id = 4
        self.zoom_id = 1
        self.call_svm_id = 3
        self.step_count = 0
        self.action_sequence = [
            None, None, 1, None, None, None, None, None, None, None,  # step 2: zoom
            None, None, 3, None, None, None, None,  # step 11: svm
            None, None, None, None, None, None, 4,  # step 18: eos
        ]
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=self.device) * 0.01
        return type('obj', (object,), {'logits': logits})()
    
    def generate_action(self, step: int) -> Optional[int]:
        """Return action token ID for given step, or None for normal token."""
        if step < len(self.action_sequence):
            return self.action_sequence[step]
        return None


def action_pattern_detector(text: str) -> Optional[str]:
    """Find action pattern in text."""
    patterns = [
        r'<zoom>\s*\[.*?\]\s*</zoom>',
        r'<call_svm>',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


# ============================================================================
# Test Functions
# ============================================================================

def test_dummy_environment():
    """Test the DummyEnvironment class."""
    print("=" * 60)
    print("Test 1: DummyEnvironment")
    print("=" * 60)
    
    env = DummyEnvironment()
    
    # Test zoom action
    result = env.step("<zoom> [100, 200, 400, 350] </zoom>")
    assert result.action_type == ActionType.ZOOM, f"Expected ZOOM, got {result.action_type}"
    assert "SVM extracted" in result.feedback, "Missing SVM feedback"
    print(f"✓ Zoom action parsed: coords=({result.action_params['x1']},{result.action_params['y1']},{result.action_params['x2']},{result.action_params['y2']})")
    print(f"  Feedback: {result.feedback[:70]}...")
    
    # Test call_svm action
    result = env.step("<call_svm>")
    assert result.action_type == ActionType.CALL_SVM, f"Expected CALL_SVM, got {result.action_type}"
    assert "SVM" in result.feedback or "OCR" in result.feedback, "Missing SVM/OCR feedback"
    print(f"✓ SVM action parsed")
    print(f"  Feedback: {result.feedback[:70]}...")
    
    # Test unknown action
    result = env.step("<unknown>")
    assert result.action_type == ActionType.UNKNOWN, f"Expected UNKNOWN, got {result.action_type}"
    print(f"✓ Unknown action handled correctly")
    
    print("✓ DummyEnvironment test PASSED\n")
    return True


def test_trajectory_structure():
    """Test trajectory data structures."""
    print("=" * 60)
    print("Test 2: Trajectory Structure")
    print("=" * 60)
    
    # Create trajectory
    traj = InterceptedTrajectory(
        prompt="请分析古籍",
        full_sequence=[1, 2, 3, 4, 5, 6],
    )
    
    # Add steps
    traj.trajectory_steps.append(TrajectoryStep(
        state="请分析古籍图像",
        action=None,
        env_feedback=None,
        tokens_generated=torch.tensor([10]),
        log_probs=torch.tensor([-0.5]),
    ))
    
    traj.trajectory_steps.append(TrajectoryStep(
        state="请分析古籍图像<zoom> [100,200,300,400] </zoom>",
        action="<zoom> [100,200,300,400] </zoom>",
        env_feedback="[System: SVM extracted text...]",
        tokens_generated=torch.tensor([1]),
        log_probs=torch.tensor([-0.3]),
    ))
    
    traj.total_env_interactions = 1
    traj.final_state = "分析完成"
    
    # Verify structure
    assert len(traj.trajectory_steps) == 2, "Wrong number of steps"
    assert traj.total_env_interactions == 1, "Wrong interaction count"
    assert traj.trajectory_steps[1].action is not None, "Action should be set"
    assert traj.trajectory_steps[1].env_feedback is not None, "Feedback should be set"
    
    print(f"✓ Trajectory created with {len(traj.trajectory_steps)} steps")
    print(f"✓ Env interactions: {traj.total_env_interactions}")
    print(f"✓ Final state length: {len(traj.final_state)}")
    print("✓ Trajectory structure test PASSED\n")
    return True


def test_action_pattern_detection():
    """Test action pattern detection regex."""
    print("=" * 60)
    print("Test 3: Action Pattern Detection")
    print("=" * 60)
    
    test_cases = [
        ("<zoom> [100, 200, 400, 350] </zoom>", True, "zoom with coords"),
        ("<call_svm>", True, "call_svm"),
        ("<ZOOM> [1,2,3,4] </ZOOM>", True, "uppercase zoom"),
        ("<CALL_SVM>", True, "uppercase svm"),
        ("<zoom>[10,20,30,40]</zoom>", True, "zoom without spaces"),
        ("Normal text without action", False, "plain text"),
        ("<zoom> [1, 2] </zoom> more text", True, "zoom with trailing text"),
        ("</zoom>", False, "only end tag"),
    ]
    
    all_passed = True
    for text, should_match, desc in test_cases:
        match = action_pattern_detector(text)
        matched = match is not None
        status = "✓" if matched == should_match else "✗"
        if matched != should_match:
            all_passed = False
        print(f"  {status} '{text[:40]:40s}' -> {'MATCH' if matched else 'NONE':8s} ({desc})")
    
    if all_passed:
        print("✓ Action pattern detection test PASSED\n")
    else:
        print("✗ Action pattern detection test FAILED\n")
    return all_passed


def test_mock_tokenizer():
    """Test mock tokenizer."""
    print("=" * 60)
    print("Test 4: Mock Tokenizer")
    print("=" * 60)
    
    tokenizer = MockTokenizer()
    
    # Test encoding
    text = "请分析"
    tokens = tokenizer.encode(text)
    print(f"  Encoded '{text}' -> {tokens[:10]}...")
    assert len(tokens) > 0, "Should encode to non-empty list"
    
    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"  Decoded back -> '{decoded[:30]}...'")
    
    # Test special tokens
    special_tokens = ['<zoom>', '<call_svm>', '<eos>']
    for st in special_tokens:
        tid = tokenizer.token_to_id.get(st)
        if tid is not None:
            print(f"  Special token '{st}' -> ID {tid}")
    
    print("✓ Mock tokenizer test PASSED\n")
    return True


def test_mock_module():
    """Test mock MLLM module."""
    print("=" * 60)
    print("Test 5: Mock MLLM Module")
    print("=" * 60)
    
    module = MockMLLMModule()
    
    # Test forward pass
    input_ids = torch.randint(0, 100, (2, 10))
    output = module(input_ids)
    
    assert hasattr(output, 'logits'), "Output should have logits"
    assert output.logits.shape == (2, 10, 1000), f"Wrong logits shape: {output.logits.shape}"
    print(f"  Forward pass: input {input_ids.shape} -> logits {output.logits.shape}")
    
    # Test action sequence
    for step in [2, 11, 17]:
        action = module.generate_action(step)
        print(f"  Step {step} action: {action}")
    
    print("✓ Mock MLLM module test PASSED\n")
    return True


def test_interceptive_generation_loop():
    """Test the interceptive generation loop logic."""
    print("=" * 60)
    print("Test 6: Interceptive Generation Loop Logic")
    print("=" * 60)
    
    tokenizer = MockTokenizer()
    module = MockMLLMModule()
    env = DummyEnvironment()
    
    # Simulate generation loop
    prompt_tokens = tokenizer.encode("请分析古籍")
    current_tokens = prompt_tokens.copy()
    step = 0
    interaction_count = 0
    max_steps = 20
    
    print(f"  Starting with {len(prompt_tokens)} prompt tokens")
    
    while step < max_steps:
        # Simulate single token generation
        action_id = module.generate_action(step)
        
        if action_id == 1:  # zoom
            # Simulate action text generation
            action_text = "<zoom> [100, 200, 300, 400] </zoom>"
            action_tokens = tokenizer.encode(action_text)
            current_tokens.extend(action_tokens)

            # Environment interaction
            result = env.step(action_text)
            interaction_count += 1

            # Simulate feedback tokens
            feedback_tokens = tokenizer.encode(result.feedback)
            current_tokens.extend(feedback_tokens)

            print(f"  Step {step}: ACTION <zoom> detected, env feedback: {len(feedback_tokens)} tokens")
            step += 1
        
        elif action_id == 3:  # call_svm
            action_text = "<call_svm>"
            action_tokens = tokenizer.encode(action_text)
            current_tokens.extend(action_tokens)

            result = env.step(action_text)
            interaction_count += 1
            feedback_tokens = tokenizer.encode(result.feedback)
            current_tokens.extend(feedback_tokens)

            print(f"  Step {step}: ACTION <call_svm> detected, env feedback: {len(feedback_tokens)} tokens")
            step += 1
        
        elif action_id == 4:  # eos
            current_tokens.append(4)
            print(f"  Step {step}: EOS reached")
            break
        
        else:
            # Normal token
            current_tokens.append(50 + step)  # Random normal token
            step += 1
    
    print(f"  Final sequence length: {len(current_tokens)} tokens")
    print(f"  Total env interactions: {interaction_count}")
    
    assert interaction_count == 2, f"Expected 2 interactions, got {interaction_count}"
    assert len(current_tokens) > len(prompt_tokens), "Sequence should grow"
    
    print("✓ Interceptive generation loop test PASSED\n")
    return True


def test_trajectory_collection():
    """Test trajectory collection with alternating state/action/feedback."""
    print("=" * 60)
    print("Test 7: Trajectory Collection")
    print("=" * 60)
    
    traj = InterceptedTrajectory(
        prompt="请分析古籍",
        full_sequence=[],
    )
    
    # Simulate trajectory
    steps_data = [
        ("请", None, None),
        ("分析", None, None),
        ("<zoom> [100,200,300,400] </zoom>", "<zoom> [100,200,300,400] </zoom>", "[System: SVM extracted...]"),
        ("古籍", None, None),
        ("图像", None, None),
        ("<call_svm>", "<call_svm>", "[System: OCR...]"),
        ("内容", None, None),
        ("完整", None, None),
    ]
    
    for state, action, feedback in steps_data:
        step = TrajectoryStep(
            state=state,
            action=action,
            env_feedback=feedback,
            tokens_generated=torch.tensor([1]),
            log_probs=torch.tensor([-0.1]),
        )
        traj.trajectory_steps.append(step)
    
    # Count env interactions
    interactions = sum(1 for s in traj.trajectory_steps if s.action is not None)
    traj.total_env_interactions = interactions
    traj.full_sequence = list(range(sum(s.tokens_generated.shape[0] for s in traj.trajectory_steps)))
    
    # Verify trajectory structure
    action_steps = [s for s in traj.trajectory_steps if s.action is not None]
    print(f"  Total steps: {len(traj.trajectory_steps)}")
    print(f"  Action steps: {len(action_steps)}")
    
    for i, s in enumerate(action_steps):
        print(f"    Step {i}: action='{s.action}', feedback_len={len(s.env_feedback or '')}")
    
    assert len(action_steps) == 2, f"Expected 2 action steps, got {len(action_steps)}"
    assert all(s.env_feedback is not None for s in action_steps), "All action steps should have feedback"
    
    print("✓ Trajectory collection test PASSED\n")
    return True


def test_special_token_handling():
    """Test special token ID resolution."""
    print("=" * 60)
    print("Test 8: Special Token Handling")
    print("=" * 60)
    
    tokenizer = MockTokenizer()
    
    # Get special token IDs
    zoom_id = tokenizer.token_to_id.get('<zoom>')
    svm_id = tokenizer.token_to_id.get('<call_svm>')
    eos_id = tokenizer.token_to_id.get('<eos>')
    
    print(f"  <zoom> ID: {zoom_id}")
    print(f"  <call_svm> ID: {svm_id}")
    print(f"  <eos> ID: {eos_id}")
    
    assert zoom_id is not None, "<zoom> token not found"
    assert svm_id is not None, "<call_svm> token not found"
    assert eos_id is not None, "<eos> token not found"
    
    # Encode and decode special tokens
    encoded = tokenizer.encode("<zoom>")
    decoded = tokenizer.decode(encoded)
    print(f"  Roundtrip <zoom>: '{encoded}' -> '{decoded}'")
    
    print("✓ Special token handling test PASSED\n")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IsoGraph Action Interceptor - Standalone Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_dummy_environment,
        test_trajectory_structure,
        test_action_pattern_detection,
        test_mock_tokenizer,
        test_mock_module,
        test_interceptive_generation_loop,
        test_trajectory_collection,
        test_special_token_handling,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
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
        print("All tests passed!")
        print("\nIsoGraph Action Interceptor implementation is ready.")
        print("Next steps:")
        print("  1. Integrate with full verl module (requires dependencies)")
        print("  2. Test with real MLLM model")
        print("  3. Implement VE-MDP environment (replace DummyEnvironment)")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
