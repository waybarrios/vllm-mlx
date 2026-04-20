"""Tests for ThinkingAwareLogitsProcessor and supporting utilities."""

import os
from typing import Callable

import mlx.core as mx
import pytest

from vllm_mlx.constrained.thinking_processor import (
    BoundedSuffixMatcher,
    Phase,
    ThinkingAwareLogitsProcessor,
)


class TestBoundedSuffixMatcher:
    def test_single_token_match(self):
        m = BoundedSuffixMatcher([42])
        assert m.feed(42) is True

    def test_single_token_no_match(self):
        m = BoundedSuffixMatcher([42])
        assert m.feed(99) is False

    def test_multi_token_sequence(self):
        m = BoundedSuffixMatcher([10, 20, 30])
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(30) is True

    def test_partial_match_then_mismatch_resets(self):
        m = BoundedSuffixMatcher([10, 20, 30])
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(99) is False  # mismatch
        # Must restart; next 10,20,30 should still match
        assert m.feed(10) is False
        assert m.feed(20) is False
        assert m.feed(30) is True

    def test_overlapping_prefix_not_missed(self):
        # Target: [1, 1, 2]. Stream: 1, 1, 1, 2.
        # A naive reset-to-0 matcher would miss this.
        m = BoundedSuffixMatcher([1, 1, 2])
        assert m.feed(1) is False
        assert m.feed(1) is False  # partial: [1, 1]
        assert m.feed(1) is False  # buf=[1, 1, 1] != [1, 1, 2]
        assert m.feed(2) is True  # buf=[1, 1, 2] == target

    def test_back_to_back_matches(self):
        m = BoundedSuffixMatcher([5, 6])
        assert m.feed(5) is False
        assert m.feed(6) is True
        # Second match immediately
        assert m.feed(5) is False
        assert m.feed(6) is True

    def test_empty_target_raises(self):
        with pytest.raises(ValueError):
            BoundedSuffixMatcher([])


# --- ThinkingAwareLogitsProcessor tests ---

# Use small token IDs for test markers.
# <think> = [10, 11], </think> = [20, 21]
_START_IDS = [10, 11]
_END_IDS = [20, 21]
_VOCAB_SIZE = 100


def _make_processor(
    budget: int = 1000,
    inner: Callable | None = None,
    start_ids: list[int] = _START_IDS,
    end_ids: list[int] = _END_IDS,
) -> ThinkingAwareLogitsProcessor:
    return ThinkingAwareLogitsProcessor(
        start_token_ids=start_ids,
        end_token_ids=end_ids,
        thinking_token_budget=budget,
        inner=inner,
        vocab_size=_VOCAB_SIZE,
    )


def _uniform_logits() -> mx.array:
    return mx.zeros((_VOCAB_SIZE,))


def _feed_sequence(
    proc: ThinkingAwareLogitsProcessor,
    token_ids: list[int],
) -> list[mx.array]:
    """Feed a sequence of token IDs one at a time, returning logits after each."""
    results = []
    tokens_so_far = []
    for tid in token_ids:
        tokens_so_far.append(tid)
        logits = proc(mx.array(tokens_so_far), _uniform_logits())
        results.append(logits)
    return results


class TestPhaseTransitions:
    def test_starts_in_idle(self):
        proc = _make_processor()
        assert proc.state == Phase.IDLE

    def test_idle_to_thinking_on_start_sequence(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11])  # <think>
        assert proc.state == Phase.THINKING

    def test_thinking_to_content_on_natural_end(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11, 50, 51, 20, 21])  # <think> tokens </think>
        assert proc.state == Phase.CONTENT

    def test_content_is_terminal_no_reentry(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11, 50, 20, 21])  # reach CONTENT
        assert proc.state == Phase.CONTENT
        _feed_sequence(proc, [10, 11])  # emit start markers again
        assert proc.state == Phase.CONTENT  # no re-entry

    def test_idle_passes_logits_through(self):
        proc = _make_processor()
        logits = _uniform_logits()
        tokens = mx.array([99])
        result = proc(tokens, logits)
        # In IDLE, logits are unchanged
        assert mx.array_equal(result, logits)

    def test_thinking_passes_logits_through(self):
        proc = _make_processor()
        _feed_sequence(proc, [10, 11])  # enter THINKING
        logits = _uniform_logits()
        result = proc(mx.array([10, 11, 50]), logits)
        assert mx.array_equal(result, logits)

    def test_thinking_tokens_counted(self):
        proc = _make_processor(budget=100)
        _feed_sequence(proc, [10, 11, 50, 51, 52])
        # 3 thinking tokens (50, 51, 52), start sequence not counted
        assert proc.thinking_tokens == 3

    def test_prompt_has_think_tag_starts_in_thinking(self):
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=_START_IDS,
            end_token_ids=_END_IDS,
            thinking_token_budget=100,
            vocab_size=_VOCAB_SIZE,
            prompt_has_think_tag=True,
        )
        assert proc.state == Phase.THINKING
        _feed_sequence(proc, [50])
        assert proc.thinking_tokens == 1

    def test_prompt_has_think_tag_budget_zero(self):
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=_START_IDS,
            end_token_ids=_END_IDS,
            thinking_token_budget=0,
            vocab_size=_VOCAB_SIZE,
            prompt_has_think_tag=True,
        )
        assert proc.state == Phase.TRANSITIONING

    def test_prompt_has_think_tag_counts_and_transitions(self):
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=_START_IDS,
            end_token_ids=_END_IDS,
            thinking_token_budget=3,
            vocab_size=_VOCAB_SIZE,
            prompt_has_think_tag=True,
        )
        _feed_sequence(proc, [50, 51, 52])
        assert proc.thinking_tokens == 3
        assert proc.state == Phase.TRANSITIONING


class TestBudgetEnforcement:
    def test_budget_forces_transition(self):
        proc = _make_processor(budget=3)
        # <think>(10,11) + 3 thinking tokens + should transition
        _feed_sequence(proc, [10, 11, 50, 51, 52])
        assert proc.state == Phase.TRANSITIONING
        assert proc.thinking_tokens == 3

    def test_transition_forces_end_sequence(self):
        proc = _make_processor(budget=2)
        # <think>(10,11) + 2 tokens -> transition
        results = _feed_sequence(proc, [10, 11, 50, 51])
        assert proc.state == Phase.TRANSITIONING
        # Now feed through transition: should force end tokens [20, 21]
        tokens_so_far = [10, 11, 50, 51]

        # First transition token
        tokens_so_far.append(0)  # placeholder, processor overrides logits
        logits = proc(mx.array(tokens_so_far), _uniform_logits())
        # Only token 20 should have logit 0, rest -inf
        assert logits[20].item() == 0.0
        assert logits[0].item() == float("-inf")

        # Second transition token
        tokens_so_far.append(0)
        logits = proc(mx.array(tokens_so_far), _uniform_logits())
        assert logits[21].item() == 0.0
        assert proc.state == Phase.CONTENT

    def test_transition_forces_end_sequence_2d_logits(self):
        """Verify forced transition works with (1, vocab) shaped logits."""
        proc = _make_processor(budget=2)
        _feed_sequence(proc, [10, 11, 50, 51])
        assert proc.state == Phase.TRANSITIONING
        tokens_so_far = [10, 11, 50, 51, 0]
        logits_2d = mx.zeros((1, _VOCAB_SIZE))
        result = proc(mx.array(tokens_so_far), logits_2d)
        assert result.shape == (1, _VOCAB_SIZE)
        assert result[0, 20].item() == 0.0
        assert result[0, 0].item() == float("-inf")

    def test_forced_transition_tokens_not_counted(self):
        proc = _make_processor(budget=2)
        _feed_sequence(proc, [10, 11, 50, 51])  # 2 thinking tokens
        assert proc.thinking_tokens == 2
        # Feed through forced transition
        _feed_sequence(proc, [20, 21])  # these are transition control tokens
        assert proc.thinking_tokens == 2  # unchanged

    def test_budget_zero_immediate_transition(self):
        proc = _make_processor(budget=0)
        _feed_sequence(proc, [10, 11])  # <think> detected
        assert proc.state == Phase.TRANSITIONING
        assert proc.thinking_tokens == 0

    def test_natural_end_before_budget(self):
        proc = _make_processor(budget=100)
        _feed_sequence(proc, [10, 11, 50, 20, 21])  # natural </think>
        assert proc.state == Phase.CONTENT
        # tokens 50 and 20 are counted; end detection only fires when the
        # full end sequence (20, 21) is matched, so 20 is counted before
        # the match completes on 21.
        assert proc.thinking_tokens == 2

    def test_budget_boundary_allows_exactly_n(self):
        proc = _make_processor(budget=5)
        _feed_sequence(proc, [10, 11, 50, 51, 52, 53, 54])
        # 5 thinking tokens (50-54), budget=5, next would exceed -> TRANSITIONING
        assert proc.thinking_tokens == 5
        assert proc.state == Phase.TRANSITIONING


class TestInnerDelegation:
    def test_inner_not_called_during_thinking(self):
        calls = []

        def mock_inner(tokens, logits):
            calls.append(len(tokens))
            return logits

        proc = _make_processor(budget=100, inner=mock_inner)
        _feed_sequence(proc, [10, 11, 50, 51])  # IDLE then THINKING
        assert len(calls) == 0

    def test_inner_called_during_content(self):
        calls = []

        def mock_inner(tokens, logits):
            calls.append(len(tokens))
            return logits * 2  # distinguishable modification

        proc = _make_processor(budget=100, inner=mock_inner)
        _feed_sequence(proc, [10, 11, 50, 20, 21])  # reach CONTENT
        assert proc.state == Phase.CONTENT
        # inner is called once during _feed_sequence when the end matcher
        # fires on token 21 and state transitions to CONTENT.
        assert len(calls) == 1

        tokens = mx.array([10, 11, 50, 20, 21, 60])
        logits = _uniform_logits()
        result = proc(tokens, logits)
        assert len(calls) == 2  # second call from explicit invocation
        assert mx.array_equal(result, logits * 2)

    def test_no_inner_passes_through_in_content(self):
        proc = _make_processor(budget=100, inner=None)
        _feed_sequence(proc, [10, 11, 50, 20, 21])  # reach CONTENT
        logits = _uniform_logits()
        tokens = mx.array([10, 11, 50, 20, 21, 60])
        result = proc(tokens, logits)
        assert mx.array_equal(result, logits)


# =============================================================================
# API model and adapter tests
# =============================================================================

from vllm_mlx.api.models import ChatCompletionRequest, Usage
from vllm_mlx.api.anthropic_models import AnthropicRequest, AnthropicUsage
from vllm_mlx.api.anthropic_adapter import anthropic_to_openai


class TestApiModels:
    def test_chat_request_accepts_thinking_token_budget(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking_token_budget=8192,
        )
        assert req.thinking_token_budget == 8192

    def test_chat_request_budget_none_by_default(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.thinking_token_budget is None

    def test_chat_request_budget_zero(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking_token_budget=0,
        )
        assert req.thinking_token_budget == 0

    def test_usage_has_reasoning_tokens(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.reasoning_tokens is None
        u2 = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            reasoning_tokens=5,
        )
        assert u2.reasoning_tokens == 5

    def test_anthropic_request_accepts_thinking_token_budget(self):
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            thinking_token_budget=4096,
        )
        assert req.thinking_token_budget == 4096

    def test_anthropic_usage_has_reasoning_tokens(self):
        u = AnthropicUsage(input_tokens=10, output_tokens=20)
        assert u.reasoning_tokens is None
        u2 = AnthropicUsage(input_tokens=10, output_tokens=20, reasoning_tokens=8)
        assert u2.reasoning_tokens == 8


class TestAnthropicAdapter:
    def test_budget_forwarded_through_adapter(self):
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_token_budget=4096,
        )
        openai_req = anthropic_to_openai(req)
        assert openai_req.thinking_token_budget == 4096

    def test_budget_none_when_not_set(self):
        req = AnthropicRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        openai_req = anthropic_to_openai(req)
        assert openai_req.thinking_token_budget is None


class TestResolver:
    def test_request_value_takes_precedence(self):
        import vllm_mlx.server as srv

        original = srv._default_thinking_token_budget
        try:
            srv._default_thinking_token_budget = 4096
            assert srv._resolve_thinking_token_budget(8192) == 8192
        finally:
            srv._default_thinking_token_budget = original

    def test_server_default_used_when_no_request_value(self):
        import vllm_mlx.server as srv

        original = srv._default_thinking_token_budget
        try:
            srv._default_thinking_token_budget = 4096
            assert srv._resolve_thinking_token_budget(None) == 4096
        finally:
            srv._default_thinking_token_budget = original

    def test_none_when_no_default_and_no_request(self):
        import vllm_mlx.server as srv

        original = srv._default_thinking_token_budget
        try:
            srv._default_thinking_token_budget = None
            assert srv._resolve_thinking_token_budget(None) is None
        finally:
            srv._default_thinking_token_budget = original

    def test_zero_is_valid_budget(self):
        import vllm_mlx.server as srv

        assert srv._resolve_thinking_token_budget(0) == 0


class TestDecisionTree:
    """Test the processor construction decision tree logic."""

    def test_thinking_budget_json_builds_unified(self):
        mock_json = lambda tokens, logits: logits  # noqa: E731
        proc = _make_processor(budget=8192, inner=mock_json)
        assert isinstance(proc, ThinkingAwareLogitsProcessor)
        assert proc._inner is mock_json

    def test_thinking_budget_no_json_builds_unified(self):
        proc = _make_processor(budget=8192, inner=None)
        assert isinstance(proc, ThinkingAwareLogitsProcessor)
        assert proc._inner is None

    def test_thinking_no_budget_json_is_fallback(self):
        """No budget + JSON = server must fall back to forcing thinking off.
        Processor alone cannot enforce this; server-level decision."""
        # Documented for integration test coverage.
        pass

    def test_no_parser_budget_is_noop(self):
        """Budget without a parser is meaningless.
        Server gates on parser availability."""
        # Documented for integration test coverage.
        pass


INTEGRATION = os.environ.get("VLLM_MLX_TEST_THINKING_INTEGRATION", "")


@pytest.mark.skipif(
    not INTEGRATION,
    reason="Set VLLM_MLX_TEST_THINKING_INTEGRATION=1 to run",
)
class TestThinkingIntegration:
    """Integration tests requiring a running Qwen model.

    Run with: VLLM_MLX_TEST_THINKING_INTEGRATION=1 pytest ...
    Requires a Qwen 3.x model server on localhost:8080 with
    --reasoning-parser qwen3.
    """

    def test_budget_caps_reasoning_tokens(self):
        """With budget=100, reasoning_tokens should be <= 100."""
        import httpx

        resp = httpx.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "qwen3.5-27b",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 4096,
                "thinking_token_budget": 100,
                "stream": False,
            },
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        usage = data.get("usage", {})
        assert usage.get("reasoning_tokens") is not None
        assert usage["reasoning_tokens"] <= 100
        # Content should be non-empty
        content = data["choices"][0]["message"]["content"]
        assert content and len(content.strip()) > 0

    def test_budget_with_json_produces_valid_json(self):
        """Thinking + JSON + budget should produce valid JSON content."""
        import httpx
        import json

        resp = httpx.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "qwen3.5-27b",
                "messages": [{"role": "user", "content": 'Return {"answer": 42}'}],
                "max_tokens": 4096,
                "thinking_token_budget": 200,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)  # must be valid JSON
        assert isinstance(parsed, dict)
