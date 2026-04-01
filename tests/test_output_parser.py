# SPDX-License-Identifier: Apache-2.0
"""Tests for ReasoningOutputParser token-based streaming extraction."""

import pytest

from vllm_mlx.reasoning import (
    DEEPSEEK_R1_MARKERS,
    DeltaMessage,
    QWEN3_MARKERS,
    QWEN35_MARKERS,
    ReasoningOutputParser,
    create_qwen3_parser,
    create_qwen35_parser,
)


class TestReasoningOutputParserInit:
    """Test parser initialization."""

    def test_basic_init(self):
        """Test basic initialization without token IDs."""
        parser = ReasoningOutputParser(
            start_token="<｜begin▁of▁thinking▁｜>",
            end_token="<｜end▁of▁thinking▁｜>",
        )
        assert parser.start_token == "<｜begin▁of▁thinking▁｜>"
        assert parser.end_token == "<｜end▁of▁thinking▁｜>"
        assert parser.start_token_id is None
        assert parser.end_token_id is None

    def test_init_with_token_ids(self):
        """Test initialization with explicit token IDs."""
        parser = ReasoningOutputParser(
            start_token="<｜begin▁of▁thinking▁｜>",
            end_token="<｜end▁of▁thinking▁｜>",
            start_token_id=151648,
            end_token_id=151649,
        )
        assert parser.start_token_id == 151648
        assert parser.end_token_id == 151649

    def test_factory_functions(self):
        """Test factory function parsers."""
        qwen3 = create_qwen3_parser()
        assert qwen3.start_token == "<｜begin▁of▁thinking▁｜>"
        assert qwen3.start_token_id == 151648

        qwen35 = create_qwen35_parser()
        assert qwen35.end_token_id == 151649


class TestNonStreamingExtraction:
    """Test extract_reasoning for complete outputs."""

    @pytest.fixture
    def parser(self):
        return create_qwen3_parser()

    def test_both_markers_present(self, parser):
        """Test extraction with both markers."""
        output = "<｜begin▁of▁thinking▁｜>Let me think step by step<｜end▁of▁thinking▁｜>The answer is 42."
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think step by step"
        assert content == "The answer is 42."

    def test_only_end_marker_implicit_mode(self, parser):
        """Test implicit reasoning mode (only end marker)."""
        output = "Thinking without start tag<｜end▁of▁thinking▁｜>Final answer"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Thinking without start tag"
        assert content == "Final answer"

    def test_no_markers_pure_content(self, parser):
        """Test output without any markers."""
        output = "Just regular content here"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is None
        assert content == "Just regular content here"

    def test_multiline_reasoning(self, parser):
        """Test reasoning with multiple lines."""
        output = """<｜begin▁of▁thinking▁｜>
Step 1: Analyze the problem
Step 2: Compute result
<｜end▁of▁thinking▁｜>42"""
        reasoning, content = parser.extract_reasoning(output)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert content == "42"


class TestStreamingTextBased:
    """Test text-based streaming extraction."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_streaming_reasoning_phase(self, parser):
        """Test streaming during reasoning phase."""
        msg = parser.extract_reasoning_streaming("", "Let me think", "Let me think")
        assert msg is not None
        assert msg.reasoning == "Let me think"
        assert msg.content is None

    def test_streaming_content_phase(self, parser):
        """Test streaming after reasoning ends."""
        # First, process end marker
        parser.extract_reasoning_streaming(
            "reasoning<｜end▁of▁thinking▁｜>",
            "reasoning<｜end▁of▁thinking▁｜>content",
            "<｜end▁of▁thinking▁｜>content"
        )
        # Then pure content
        msg = parser.extract_reasoning_streaming(
            "reasoning<｜end▁of▁thinking▁｜>content",
            "reasoning<｜end▁of▁thinking▁｜>content here",
            " here"
        )
        assert msg.content == " here"

    def test_streaming_skip_markers(self, parser):
        """Test that pure marker deltas are skipped."""
        msg = parser.extract_reasoning_streaming("", "<｜begin▁of▁thinking▁｜>", "<｜begin▁of▁thinking▁｜>")
        # Should return None (marker skipped)
        assert msg is None

    def test_streaming_transition(self, parser):
        """Test transition from reasoning to content in one delta."""
        msg = parser.extract_reasoning_streaming(
            "reasoning",
            "reasoning<｜end▁of▁thinking▁｜>answer",
            "<｜end▁of▁thinking▁｜>answer"
        )
        assert msg is not None
        # Transition chunk: reasoning is empty (already in previous), content is "answer"
        assert msg.reasoning is None  # No new reasoning in this delta
        assert msg.content == "answer"


class TestStreamingTokenBased:
    """Test token-based streaming extraction (core feature)."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_token_based_reasoning_phase(self, parser):
        """Test token-based extraction during reasoning."""
        msg = parser.extract_streaming_with_tokens(
            delta_text="Let me think",
            delta_token_ids=[100, 101, 102]  # Random token IDs
        )
        assert msg is not None
        assert msg.reasoning == "Let me think"
        assert parser.end_token_id not in parser._previous_token_ids

    def test_token_based_content_phase_after_end(self, parser):
        """Test that content is returned after end token seen."""
        # Simulate end token in previous tokens
        parser._previous_token_ids = [100, 151649]  # end_token_id present
        
        msg = parser.extract_streaming_with_tokens(
            delta_text="Final answer",
            delta_token_ids=[200, 201]
        )
        assert msg.content == "Final answer"
        assert msg.reasoning is None

    def test_token_based_end_token_in_delta(self, parser):
        """Test end token detection in delta."""
        # Note: 151649 is the end_token_id for Qwen3
        msg = parser.extract_streaming_with_tokens(
            delta_text="reasoning<｜end▁of▁thinking▁｜>content",
            delta_token_ids=[100, 151649, 200]
        )
        assert msg is not None
        # Should split at end marker
        assert "reasoning" in msg.reasoning
        assert "content" in msg.content

    def test_token_based_start_token_stripping(self, parser):
        """Test start token stripping from delta."""
        # Note: 151648 is the start_token_id for Qwen3
        msg = parser.extract_streaming_with_tokens(
            delta_text="<｜begin▁of▁thinking▁｜>reasoning text",
            delta_token_ids=[151648, 100, 101]
        )
        assert msg is not None
        # Start marker should be stripped
        assert "<｜begin▁of▁thinking▁｜>" not in msg.reasoning

    def test_token_based_multi_token_delta(self, parser):
        """Test handling of multi-token deltas."""
        # Simulate a delta with multiple tokens, including end marker
        delta_text = "final reasoning<｜end▁of▁thinking▁｜>answer"
        delta_ids = [100, 101, 151649, 200]
        
        msg = parser.extract_streaming_with_tokens(
            delta_text=delta_text,
            delta_token_ids=delta_ids
        )
        assert msg is not None
        # Should split correctly
        assert "final reasoning" in msg.reasoning
        assert "answer" in msg.content
        # Should mark as content phase
        assert parser._in_content_phase

    def test_token_based_state_tracking(self, parser):
        """Test that previous_token_ids are tracked."""
        parser.extract_streaming_with_tokens("text1", [100, 101])
        assert 100 in parser._previous_token_ids
        assert 101 in parser._previous_token_ids
        
        parser.extract_streaming_with_tokens("text2", [102])
        assert 102 in parser._previous_token_ids
        assert len(parser._previous_token_ids) == 3


class TestResetState:
    """Test state reset functionality."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        # Simulate some state
        parser._previous_token_ids = [1, 2, 3]
        parser._in_content_phase = True
        return parser

    def test_reset_clears_token_ids(self, parser):
        """Test that reset clears previous_token_ids."""
        parser.reset_state()
        assert parser._previous_token_ids == []

    def test_reset_clears_phase_state(self, parser):
        """Test that reset clears in_content_phase."""
        parser.reset_state()
        assert parser._in_content_phase is False


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_empty_delta(self, parser):
        """Test empty delta text."""
        msg = parser.extract_streaming_with_tokens("", [])
        assert msg is None  # Empty delta should return None

    def test_whitespace_only_reasoning(self, parser):
        """Test reasoning with only whitespace."""
        output = "<｜begin▁of▁thinking▁｜>   <｜end▁of▁thinking▁｜>content"
        reasoning, content = parser.extract_reasoning(output)
        # Whitespace-only reasoning should be None
        assert reasoning is None or reasoning.strip() == ""

    def test_marker_at_boundary(self, parser):
        """Test marker at exact boundary positions."""
        # End marker at start of content
        output = "<｜begin▁of▁thinking▁｜>think<｜end▁of▁thinking▁｜>"
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "think"
        assert content is None or content == ""

    def test_no_token_ids_graceful_fallback(self):
        """Test graceful fallback when token IDs not available."""
        parser = ReasoningOutputParser(
            start_token="<｜begin▁of▁thinking▁｜>",
            end_token="<｜end▁of▁thinking▁｜>",
            # No token IDs provided
        )
        parser.reset_state()
        
        # Should still work with text-based approach
        msg = parser.extract_streaming_with_tokens("reasoning text", [100, 101])
        assert msg is not None
        assert msg.reasoning == "reasoning text"


class TestIntegrationWithTextBasedParser:
    """Test integration between token-based and text-based methods."""

    @pytest.fixture
    def parser(self):
        parser = create_qwen3_parser()
        parser.reset_state()
        return parser

    def test_text_based_fallback_for_no_token_ids(self, parser):
        """Test that text-based method still works."""
        # Use text-based streaming (no token IDs)
        msg = parser.extract_reasoning_streaming(
            previous_text="",
            current_text="reasoning",
            delta_text="reasoning"
        )
        assert msg is not None
        assert msg.reasoning == "reasoning"

    def test_both_methods_consistent(self, parser):
        """Test that both methods produce consistent results."""
        # Token-based
        parser.reset_state()
        msg1 = parser.extract_streaming_with_tokens("reasoning", [100])
        
        # Text-based
        parser.reset_state()
        msg2 = parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        
        # Both should mark as reasoning
        assert msg1.reasoning == msg2.reasoning


class TestMarkerConfigurations:
    """Test marker configuration constants."""

    def test_qwen3_markers(self):
        """Test Qwen3 marker config."""
        assert QWEN3_MARKERS["start_token"] == "<｜begin▁of▁thinking▁｜>"
        assert QWEN3_MARKERS["end_token"] == "<｜end▁of▁thinking▁｜>"
        assert QWEN3_MARKERS["start_token_id"] == 151648
        assert QWEN3_MARKERS["end_token_id"] == 151649

    def test_qwen35_markers(self):
        """Test Qwen3.5 marker config."""
        assert QWEN35_MARKERS["start_token"] == "<｜begin▁of▁thinking▁｜>"
        assert QWEN35_MARKERS["end_token_id"] == 151649