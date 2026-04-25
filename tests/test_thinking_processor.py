# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ThinkingAwareLogitsProcessor.

Covers:
* Phase state machine transitions (IDLE -> THINKING -> CONTENT)
* Budget enforcement (THINKING -> TRANSITIONING -> CONTENT)
* Budget-zero edge case (immediate TRANSITIONING)
* Inner processor delegation in CONTENT phase
* Content-phase control token masking
* BoundedSuffixMatcher correctness
* Snapshot/restore for speculative decoding rollback
* prompt_has_think_tag fast-start paths

These tests are pure-logic: they use synthetic token IDs and small
vocab sizes. No model loading required.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_mlx.constrained.thinking_processor import (
    BoundedSuffixMatcher,
    Phase,
    ThinkingAwareLogitsProcessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
# Synthetic token IDs for <think> and </think>
THINK_START = [10]
THINK_END = [11]
# Multi-token end sequence for testing
THINK_END_MULTI = [11, 12]


def _make_logits(vocab_size: int = VOCAB_SIZE) -> mx.array:
    """Return uniform logits (all zeros)."""
    return mx.zeros((vocab_size,))


def _tokens(*ids: int) -> mx.array:
    """Build a 1-D token array from integer IDs."""
    return mx.array(list(ids), dtype=mx.int32)


def _argmax(logits: mx.array) -> int:
    """Return the argmax of a logits vector."""
    return int(logits.argmax().item())


# ---------------------------------------------------------------------------
# BoundedSuffixMatcher
# ---------------------------------------------------------------------------


class TestBoundedSuffixMatcher:
    def test_single_token_match(self):
        m = BoundedSuffixMatcher([5])
        assert not m.feed(4)
        assert m.feed(5)

    def test_multi_token_match(self):
        m = BoundedSuffixMatcher([1, 2, 3])
        assert not m.feed(1)
        assert not m.feed(2)
        assert m.feed(3)

    def test_no_false_positive(self):
        m = BoundedSuffixMatcher([1, 2])
        m.feed(1)
        m.feed(9)  # breaks the sequence
        assert not m.feed(2)  # 9,2 != 1,2

    def test_overlapping_prefix(self):
        """The rolling buffer catches overlapping prefixes."""
        m = BoundedSuffixMatcher([1, 1])
        m.feed(1)
        # Buffer is [1]; next 1 gives [1,1]
        assert m.feed(1)

    def test_snapshot_restore(self):
        m = BoundedSuffixMatcher([1, 2, 3])
        m.feed(1)
        m.feed(2)
        snap = m.snapshot()
        assert not m.feed(9)  # break it
        m.restore(snap)
        assert m.feed(3)  # restored to [1,2], feed 3 -> match

    def test_reset(self):
        m = BoundedSuffixMatcher([5])
        m.feed(5)
        m.reset()
        assert not m.feed(4)
        assert m.feed(5)

    def test_empty_target_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BoundedSuffixMatcher([])


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — basic lifecycle
# ---------------------------------------------------------------------------


class TestThinkingProcessorLifecycle:
    def test_idle_to_thinking_to_content(self):
        """Full natural lifecycle: IDLE -> THINKING -> CONTENT via end tokens."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )
        assert proc.state == Phase.IDLE

        # Feed start token
        logits = proc(_tokens(*THINK_START), _make_logits())
        assert proc.state == Phase.THINKING

        # Feed some thinking tokens
        logits = proc(_tokens(*THINK_START, 5, 6), _make_logits())
        assert proc.state == Phase.THINKING
        assert proc.thinking_tokens == 2

        # Feed end token
        logits = proc(_tokens(*THINK_START, 5, 6, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT

    def test_budget_enforcement(self):
        """Budget exhaustion triggers TRANSITIONING then CONTENT."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=3,
            vocab_size=VOCAB_SIZE,
        )

        # Enter thinking
        proc(_tokens(*THINK_START), _make_logits())
        assert proc.state == Phase.THINKING

        # Consume budget: 3 thinking tokens
        proc(_tokens(*THINK_START, 1, 2, 3), _make_logits())
        assert proc.state == Phase.TRANSITIONING

        # During TRANSITIONING, logits force the end token
        logits = proc(_tokens(*THINK_START, 1, 2, 3), _make_logits())
        assert _argmax(logits) == THINK_END[0]

        # After forced end token is consumed, enter CONTENT
        proc(_tokens(*THINK_START, 1, 2, 3, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT
        assert proc.thinking_tokens == 3

    def test_budget_zero(self):
        """Budget=0 with prompt_has_think_tag starts in TRANSITIONING."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=0,
            vocab_size=VOCAB_SIZE,
            prompt_has_think_tag=True,
        )
        assert proc.state == Phase.TRANSITIONING

        # Force transition on empty tokens
        logits = proc(_tokens(), _make_logits())
        assert _argmax(logits) == THINK_END[0]

    def test_prompt_has_think_tag_starts_thinking(self):
        """prompt_has_think_tag=True with budget>0 starts in THINKING."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
            prompt_has_think_tag=True,
        )
        assert proc.state == Phase.THINKING

    def test_is_retired_no_inner(self):
        """is_retired is True in CONTENT with no inner processor."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )
        assert not proc.is_retired
        # Drive to CONTENT
        proc(_tokens(*THINK_START, 5, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT
        assert proc.is_retired

    def test_is_retired_with_inner(self):
        """is_retired is False in CONTENT when inner processor is present."""
        inner_called = []

        def fake_inner(tokens, logits):
            inner_called.append(True)
            return logits

        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            inner=fake_inner,
            vocab_size=VOCAB_SIZE,
        )
        proc(_tokens(*THINK_START, 5, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT
        assert not proc.is_retired


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — multi-token end sequence
# ---------------------------------------------------------------------------


class TestThinkingProcessorMultiTokenEnd:
    def test_multi_token_transition(self):
        """Budget exhaustion with multi-token end forces each token in sequence."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END_MULTI,
            thinking_token_budget=2,
            vocab_size=VOCAB_SIZE,
        )

        # Enter thinking and exhaust budget
        proc(_tokens(*THINK_START, 1, 2), _make_logits())
        assert proc.state == Phase.TRANSITIONING

        # First forced token
        logits = proc(_tokens(*THINK_START, 1, 2), _make_logits())
        assert _argmax(logits) == THINK_END_MULTI[0]

        # Feed first end token, still transitioning
        proc(_tokens(*THINK_START, 1, 2, THINK_END_MULTI[0]), _make_logits())
        assert proc.state == Phase.TRANSITIONING

        # Second forced token
        logits = proc(_tokens(*THINK_START, 1, 2, THINK_END_MULTI[0]), _make_logits())
        assert _argmax(logits) == THINK_END_MULTI[1]

        # Feed second end token, now CONTENT
        proc(_tokens(*THINK_START, 1, 2, *THINK_END_MULTI), _make_logits())
        assert proc.state == Phase.CONTENT


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — inner processor delegation
# ---------------------------------------------------------------------------


class TestThinkingProcessorInnerDelegation:
    def test_inner_called_in_content_phase(self):
        """Inner processor is called during CONTENT phase."""
        calls = []

        def tracking_inner(tokens, logits):
            calls.append(len(tokens) if hasattr(tokens, "__len__") else tokens.size)
            return logits

        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            inner=tracking_inner,
            vocab_size=VOCAB_SIZE,
        )

        # Drive to CONTENT -- inner is called on this step too since the
        # processor reaches CONTENT and then __call__ delegates.
        proc(_tokens(*THINK_START, 5, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT
        assert len(calls) == 1

        # Another call in CONTENT phase
        proc(_tokens(*THINK_START, 5, *THINK_END, 20), _make_logits())
        assert len(calls) == 2

    def test_inner_not_called_during_thinking(self):
        """Inner processor is NOT called during THINKING phase."""
        calls = []

        def tracking_inner(tokens, logits):
            calls.append(True)
            return logits

        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            inner=tracking_inner,
            vocab_size=VOCAB_SIZE,
        )

        proc(_tokens(*THINK_START, 5, 6), _make_logits())
        assert proc.state == Phase.THINKING
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — content-phase masking
# ---------------------------------------------------------------------------


class TestThinkingProcessorContentMasking:
    def test_control_tokens_masked_in_content(self):
        """Start/end tag tokens are masked to -inf during CONTENT."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )

        # Drive to CONTENT
        proc(_tokens(*THINK_START, 5, *THINK_END), _make_logits())
        logits = proc(_tokens(*THINK_START, 5, *THINK_END, 20), _make_logits())

        # Start and end tokens should be masked
        assert logits[THINK_START[0]].item() == float("-inf")
        assert logits[THINK_END[0]].item() == float("-inf")

    def test_control_tokens_not_masked_during_thinking(self):
        """During THINKING phase, control tokens are NOT masked (pass-through)."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )

        # In THINKING
        proc(_tokens(*THINK_START), _make_logits())
        logits = proc(_tokens(*THINK_START, 5), _make_logits())
        # Should be pass-through (all zeros)
        assert logits[THINK_START[0]].item() == 0.0


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — snapshot/restore (speculative rollback)
# ---------------------------------------------------------------------------


class TestThinkingProcessorRollback:
    def test_rollback_on_prefix_change(self):
        """Sync correctly handles token prefix divergence (speculative rollback)."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )

        # Feed: start + two thinking tokens
        proc(_tokens(*THINK_START, 5, 6), _make_logits())
        assert proc.state == Phase.THINKING
        assert proc.thinking_tokens == 2

        # Now the sequence diverges: start + one thinking token (rollback)
        proc(_tokens(*THINK_START, 5), _make_logits())
        assert proc.state == Phase.THINKING
        assert proc.thinking_tokens == 1

    def test_rollback_across_phases(self):
        """Rollback from CONTENT to THINKING when prefix diverges."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=100,
            vocab_size=VOCAB_SIZE,
        )

        # Drive to CONTENT
        proc(_tokens(*THINK_START, 5, *THINK_END), _make_logits())
        assert proc.state == Phase.CONTENT

        # Diverge at a shorter prefix still in THINKING range
        proc(_tokens(*THINK_START, 5, 7), _make_logits())
        assert proc.state == Phase.THINKING
        assert proc.thinking_tokens == 2


# ---------------------------------------------------------------------------
# ThinkingAwareLogitsProcessor — 2-D logits
# ---------------------------------------------------------------------------


class TestThinkingProcessor2DLogits:
    def test_2d_logits_transition(self):
        """Force transition works with 2-D (1, vocab) logits shape."""
        proc = ThinkingAwareLogitsProcessor(
            start_token_ids=THINK_START,
            end_token_ids=THINK_END,
            thinking_token_budget=1,
            vocab_size=VOCAB_SIZE,
        )

        # Enter thinking, exhaust budget
        proc(_tokens(*THINK_START, 1), _make_logits())
        assert proc.state == Phase.TRANSITIONING

        logits_2d = mx.zeros((1, VOCAB_SIZE))
        result = proc(_tokens(*THINK_START, 1), logits_2d)
        assert result.shape == (1, VOCAB_SIZE)
        assert result[0, THINK_END[0]].item() == 0.0
        # All others should be -inf
        assert result[0, 0].item() == float("-inf")
