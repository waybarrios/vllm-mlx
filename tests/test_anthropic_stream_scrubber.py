# SPDX-License-Identifier: Apache-2.0
"""Tests for _AnthropicStreamScrubber – stateful tag stripping for Anthropic streaming.

Tests the scrubber introduced in commit 6805baf which strips <think>, <tool_call>,
<function=...>, and <parameter=...> markup from streamed text deltas on the
Anthropic /v1/messages endpoint.

These are pure logic tests with no MLX dependency.
"""

import pytest

from vllm_mlx.server import _AnthropicStreamScrubber


# =============================================================================
# Basic Construction / Initial State
# =============================================================================


class TestScrubberInitialState:
    """Test scrubber creation and initial state."""

    def test_initial_mode_is_text(self):
        scrubber = _AnthropicStreamScrubber()
        assert scrubber.mode == "TEXT"

    def test_initial_carry_is_empty(self):
        scrubber = _AnthropicStreamScrubber()
        assert scrubber.carry == ""

    def test_class_constants_exist(self):
        """Verify key class-level constants are defined."""
        assert _AnthropicStreamScrubber.THINK_OPEN == "<think>"
        assert _AnthropicStreamScrubber.THINK_CLOSE == "</think>"
        assert _AnthropicStreamScrubber.TOOL_OPEN == "<tool_call>"
        assert _AnthropicStreamScrubber.TOOL_CLOSE == "</tool_call>"
        assert _AnthropicStreamScrubber.FUNC_CLOSE == "</function>"
        assert _AnthropicStreamScrubber.PARAM_CLOSE == "</parameter>"
        assert _AnthropicStreamScrubber.FUNC_PREFIX == "<function="
        assert _AnthropicStreamScrubber.PARAM_PREFIX == "<parameter="

    def test_carry_n_large_enough(self):
        """CARRY_N must be at least max(len(tag)) - 1."""
        assert _AnthropicStreamScrubber.CARRY_N >= _AnthropicStreamScrubber.MAX_TAG - 1


# =============================================================================
# Plain Text (no tags) – passthrough
# =============================================================================


class TestScrubberPlainText:
    """Scrubber should pass through normal text unchanged."""

    def test_empty_string(self):
        scrubber = _AnthropicStreamScrubber()
        assert scrubber.feed("") == ""

    def test_none_delta(self):
        """feed(None) should not crash."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed(None)
        assert result == ""

    def test_short_text(self):
        scrubber = _AnthropicStreamScrubber()
        # Short text with no '<' should emit immediately (zero carry)
        result = scrubber.feed("Hi")
        assert result == "Hi"
        assert scrubber.carry == ""

    def test_long_plain_text(self):
        """Text longer than CARRY_N should emit most of it immediately."""
        scrubber = _AnthropicStreamScrubber()
        text = "Hello, this is a long sentence with no markup at all, just ordinary text."
        result = scrubber.feed(text)
        flushed = scrubber.flush()
        assert result + flushed == text

    def test_multiple_plain_deltas(self):
        """Multiple consecutive plain-text deltas should reconstruct fully."""
        scrubber = _AnthropicStreamScrubber()
        parts = ["Hello ", "world, ", "how ", "are ", "you?"]
        collected = ""
        for p in parts:
            collected += scrubber.feed(p)
        collected += scrubber.flush()
        assert collected == "".join(parts)

    def test_flush_in_text_mode_returns_carry(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("abc")
        flushed = scrubber.flush()
        # After flush, carry should be empty
        assert scrubber.carry == ""


# =============================================================================
# <think>...</think> suppression
# =============================================================================


class TestScrubberThinkTags:
    """Test suppression of <think>...</think> blocks."""

    def test_think_block_in_single_delta(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Hello <think>internal reasoning</think> world")
        result += scrubber.flush()
        assert "<think>" not in result
        assert "internal reasoning" not in result
        assert "</think>" not in result
        assert "Hello " in result
        assert " world" in result

    def test_think_block_removes_content(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("<think>some reasoning</think>After thought")
        result += scrubber.flush()
        assert "some reasoning" not in result
        assert "After thought" in result

    def test_think_block_split_across_deltas(self):
        """Tag split across multiple feed() calls."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("Before <thi")
        collected += scrubber.feed("nk>secret reasoning here")
        collected += scrubber.feed("</think> After")
        collected += scrubber.flush()
        assert "secret reasoning" not in collected
        assert "<think>" not in collected
        assert "</think>" not in collected
        assert "Before" in collected
        assert "After" in collected

    def test_think_block_close_tag_split(self):
        """Closing </think> tag split across deltas."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("<think>reasoning</th")
        collected += scrubber.feed("ink>visible text")
        collected += scrubber.flush()
        assert "reasoning" not in collected
        assert "visible text" in collected

    def test_multiple_think_blocks(self):
        scrubber = _AnthropicStreamScrubber()
        text = "A<think>r1</think>B<think>r2</think>C"
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "r1" not in result
        assert "r2" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_think_with_newlines(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Hi<think>\nStep 1\nStep 2\n</think> Done")
        result += scrubber.flush()
        assert "Step 1" not in result
        assert "Step 2" not in result
        assert "Hi" in result
        assert "Done" in result

    def test_think_at_end_of_stream_flushed_away(self):
        """If stream ends inside a <think> block, flush discards carry."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("Hello <think>unclosed reasoning")
        flushed = scrubber.flush()
        # In suppression mode, flush returns ""
        assert "unclosed reasoning" not in collected + flushed
        assert "Hello" in collected + flushed


# =============================================================================
# <tool_call>...</tool_call> suppression
# =============================================================================


class TestScrubberToolCallTags:
    """Test suppression of <tool_call>...</tool_call> blocks."""

    def test_tool_call_in_single_delta(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed('Before <tool_call>{"name":"fn"}</tool_call> After')
        result += scrubber.flush()
        assert "<tool_call>" not in result
        assert '{"name":"fn"}' not in result
        assert "</tool_call>" not in result
        assert "Before" in result
        assert "After" in result

    def test_tool_call_split_across_deltas(self):
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("Text <tool_")
        collected += scrubber.feed('call>{"name":"search","args":{}}</tool_call> rest')
        collected += scrubber.flush()
        assert '{"name":"search"' not in collected
        assert "<tool_call>" not in collected
        assert "Text" in collected
        assert "rest" in collected

    def test_tool_call_close_split(self):
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed('<tool_call>data</tool_')
        collected += scrubber.feed("call>visible")
        collected += scrubber.flush()
        assert "data" not in collected
        assert "visible" in collected

    def test_multiple_tool_calls(self):
        scrubber = _AnthropicStreamScrubber()
        text = 'A<tool_call>call1</tool_call>B<tool_call>call2</tool_call>C'
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "call1" not in result
        assert "call2" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_tool_call_at_end_of_stream(self):
        """Unclosed tool_call at end of stream – suppressed by flush."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("Prefix <tool_call>unclosed")
        flushed = scrubber.flush()
        assert "unclosed" not in collected + flushed
        assert "Prefix" in collected + flushed


# =============================================================================
# <function=NAME>...</function> suppression
# =============================================================================


class TestScrubberFunctionTags:
    """Test suppression of <function=name>...</function> (Llama-style)."""

    def test_function_tag_single_delta(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed('Hello <function=search>{"q":"test"}</function> world')
        result += scrubber.flush()
        assert "<function=" not in result
        assert '{"q":"test"}' not in result
        assert "</function>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_function_tag_split_across_deltas(self):
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        collected += scrubber.feed("Text <func")
        collected += scrubber.feed("tion=get_weather")
        collected += scrubber.feed('>{"city":"NYC"}</function> done')
        collected += scrubber.flush()
        assert "get_weather" not in collected
        assert '{"city":"NYC"}' not in collected
        assert "Text" in collected
        assert "done" in collected

    def test_function_tag_with_complex_name(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("<function=my_long_function_name>body</function>after")
        result += scrubber.flush()
        assert "my_long_function_name" not in result
        assert "body" not in result
        assert "after" in result

    def test_stray_function_close_suppressed(self):
        """A stray </function> outside a function block should be suppressed."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("text</function>more")
        result += scrubber.flush()
        assert "</function>" not in result
        assert "text" in result
        assert "more" in result


# =============================================================================
# <parameter=NAME>...</parameter> suppression
# =============================================================================


class TestScrubberParameterTags:
    """Test suppression of <parameter=name>...</parameter> (Llama-style)."""

    def test_parameter_tag_single_delta(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Before <parameter=city>NYC</parameter> After")
        # parameter= maps to IN_FUNCTION mode, closes on </function>
        # But </parameter> is consumed as stray closing tag in TEXT mode...
        # Actually <parameter= opens IN_FUNCTION, and IN_FUNCTION closes on </function>
        # So </parameter> won't close IN_FUNCTION. Let's verify behavior.
        # The close for IN_FUNCTION is </function>, so content after <parameter=city>
        # stays suppressed until </function> is seen or stream ends.
        result += scrubber.flush()
        assert "<parameter=" not in result
        assert "Before" in result

    def test_stray_parameter_close_suppressed(self):
        """A stray </parameter> tag should be stripped."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("text</parameter>more")
        result += scrubber.flush()
        assert "</parameter>" not in result
        assert "text" in result
        assert "more" in result

    def test_parameter_inside_function_block(self):
        """Parameter tags typically appear inside function blocks."""
        scrubber = _AnthropicStreamScrubber()
        text = '<function=search><parameter=query>test</parameter></function>done'
        result = scrubber.feed(text)
        result += scrubber.flush()
        # Everything inside <function=...>...</function> should be suppressed
        assert "query" not in result
        assert "test" not in result
        assert "done" in result


# =============================================================================
# Stray Closing Tags
# =============================================================================


class TestScrubberStrayClosingTags:
    """Test that stray closing tags outside their context are consumed."""

    def test_stray_think_close(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("hello</think>world")
        result += scrubber.flush()
        assert "</think>" not in result
        assert "hello" in result
        assert "world" in result

    def test_stray_tool_call_close(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("hello</tool_call>world")
        result += scrubber.flush()
        assert "</tool_call>" not in result
        assert "hello" in result
        assert "world" in result

    def test_stray_function_close(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("hello</function>world")
        result += scrubber.flush()
        assert "</function>" not in result
        assert "hello" in result
        assert "world" in result

    def test_stray_parameter_close(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("hello</parameter>world")
        result += scrubber.flush()
        assert "</parameter>" not in result
        assert "hello" in result
        assert "world" in result

    def test_multiple_stray_closing_tags(self):
        scrubber = _AnthropicStreamScrubber()
        text = "a</think>b</tool_call>c</function>d</parameter>e"
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "</think>" not in result
        assert "</tool_call>" not in result
        assert "</function>" not in result
        assert "</parameter>" not in result
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result
        assert "e" in result


# =============================================================================
# Mixed Scenarios
# =============================================================================


class TestScrubberMixedContent:
    """Test combinations of tags and text."""

    def test_think_then_tool_call(self):
        scrubber = _AnthropicStreamScrubber()
        text = 'Before<think>reasoning</think>Middle<tool_call>{"fn":"x"}</tool_call>After'
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "reasoning" not in result
        assert '{"fn":"x"}' not in result
        assert "Before" in result
        assert "Middle" in result
        assert "After" in result

    def test_tool_call_then_think(self):
        scrubber = _AnthropicStreamScrubber()
        text = '<tool_call>data</tool_call>text<think>thought</think>end'
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "data" not in result
        assert "thought" not in result
        assert "text" in result
        assert "end" in result

    def test_think_with_function_inside_tool_call(self):
        """Nested-looking tags – only outer suppression matters."""
        scrubber = _AnthropicStreamScrubber()
        text = '<tool_call>outer<function=inner>nested</function></tool_call>visible'
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "outer" not in result
        assert "nested" not in result
        assert "visible" in result

    def test_interleaved_text_and_tags(self):
        scrubber = _AnthropicStreamScrubber()
        parts = [
            "Hello ",
            "<think>",
            "Let me think about this...",
            "</think>",
            " Here's my answer.",
        ]
        collected = ""
        for p in parts:
            collected += scrubber.feed(p)
        collected += scrubber.flush()
        assert "Let me think about this" not in collected
        assert "Hello" in collected
        assert "Here's my answer." in collected

    def test_realistic_streaming_scenario(self):
        """Simulate a realistic token-by-token streaming scenario."""
        scrubber = _AnthropicStreamScrubber()
        # Model outputs: "<think>Let me check</think>The weather is sunny."
        # Split into small token-like deltas
        tokens = [
            "<",
            "think",
            ">",
            "Let",
            " me",
            " check",
            "</",
            "think",
            ">",
            "The",
            " weather",
            " is",
            " sunny",
            ".",
        ]
        collected = ""
        for tok in tokens:
            collected += scrubber.feed(tok)
        collected += scrubber.flush()
        assert "Let me check" not in collected
        assert "<think>" not in collected
        assert "</think>" not in collected
        assert "The weather is sunny." in collected

    def test_realistic_tool_call_streaming(self):
        """Simulate tool-call markup arriving token by token."""
        scrubber = _AnthropicStreamScrubber()
        tokens = [
            "I'll ",
            "search",
            " for ",
            "that.",
            "<tool",
            "_call",
            ">",
            '{"name',
            '":"',
            "search",
            '","',
            "arguments",
            '":{"',
            "q",
            '":"',
            "weather",
            '"}}',
            "</tool",
            "_call",
            ">",
        ]
        collected = ""
        for tok in tokens:
            collected += scrubber.feed(tok)
        collected += scrubber.flush()
        assert "<tool_call>" not in collected
        assert "</tool_call>" not in collected
        assert '"name"' not in collected
        assert "I'll search for that." in collected


# =============================================================================
# Tag Split Across Boundaries (carry buffer tests)
# =============================================================================


class TestScrubberCarryBuffer:
    """Test the carry buffer behavior for tags split across deltas."""

    def test_tag_split_at_every_character(self):
        """Split <think> one char at a time."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        for ch in "before<think>hidden</think>after":
            collected += scrubber.feed(ch)
        collected += scrubber.flush()
        assert "hidden" not in collected
        assert "before" in collected
        assert "after" in collected

    def test_close_tag_split_at_every_character(self):
        """Split </think> one char at a time."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        for ch in "<think>suppressed</think>visible":
            collected += scrubber.feed(ch)
        collected += scrubber.flush()
        assert "suppressed" not in collected
        assert "visible" in collected

    def test_tool_call_tag_split_at_every_character(self):
        """Split <tool_call> one char at a time."""
        scrubber = _AnthropicStreamScrubber()
        collected = ""
        for ch in "pre<tool_call>body</tool_call>post":
            collected += scrubber.feed(ch)
        collected += scrubber.flush()
        assert "body" not in collected
        assert "pre" in collected
        assert "post" in collected

    def test_carry_cleared_after_flush(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("some text")
        scrubber.flush()
        assert scrubber.carry == ""

    def test_carry_cleared_after_full_consumption(self):
        """When all content is consumed, carry should be empty."""
        scrubber = _AnthropicStreamScrubber()
        # Feed a complete tag that consumes everything
        scrubber.feed("<think>x</think>")
        result = scrubber.flush()
        assert scrubber.carry == ""


# =============================================================================
# flush() Behavior
# =============================================================================


class TestScrubberFlush:
    """Test flush() method at end of stream."""

    def test_flush_emits_remaining_text(self):
        scrubber = _AnthropicStreamScrubber()
        # "hi" has no '<' so is emitted immediately by feed().
        result = scrubber.feed("hi")
        assert result == "hi"
        # flush() should return empty since carry is empty.
        flushed = scrubber.flush()
        assert flushed == ""

    def test_flush_emits_carry_with_angle_bracket(self):
        """Text ending with '<' is held in carry; flush emits it."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("text<")
        assert scrubber.carry == "<"
        flushed = scrubber.flush()
        # '<' alone is not a valid tag, flush strips nothing extra
        assert flushed == "<"

    def test_flush_in_think_mode_discards(self):
        """If stream ends while inside <think>, flush returns empty."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<think>unfinished")
        assert scrubber.mode == "IN_THINK"
        flushed = scrubber.flush()
        assert flushed == ""
        assert scrubber.carry == ""

    def test_flush_in_toolcall_mode_discards(self):
        """If stream ends while inside <tool_call>, flush returns empty."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<tool_call>unfinished")
        assert scrubber.mode == "IN_TOOLCALL"
        flushed = scrubber.flush()
        assert flushed == ""

    def test_flush_in_function_mode_discards(self):
        """If stream ends while inside <function=...>, flush returns empty."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<function=test>unfinished")
        assert scrubber.mode == "IN_FUNCTION"
        flushed = scrubber.flush()
        assert flushed == ""

    def test_flush_strips_residual_exact_tags(self):
        """flush() in TEXT mode strips any leftover exact tags from carry."""
        scrubber = _AnthropicStreamScrubber()
        # Manually set carry to simulate leftover tag fragment
        scrubber.carry = "text<think>leftover"
        scrubber.mode = "TEXT"
        flushed = scrubber.flush()
        assert "<think>" not in flushed
        assert "text" in flushed

    def test_flush_strips_residual_function_tags(self):
        """flush() strips residual <function=name> from carry."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.carry = "before<function=test>after"
        scrubber.mode = "TEXT"
        flushed = scrubber.flush()
        assert "<function=" not in flushed
        assert "before" in flushed

    def test_flush_strips_residual_parameter_tags(self):
        """flush() strips residual <parameter=name> from carry."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.carry = "before<parameter=x>after"
        scrubber.mode = "TEXT"
        flushed = scrubber.flush()
        assert "<parameter=" not in flushed
        assert "before" in flushed

    def test_double_flush_returns_empty(self):
        """Calling flush() twice should return empty the second time."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("hello")
        first = scrubber.flush()
        second = scrubber.flush()
        assert second == ""


# =============================================================================
# State Machine Transitions
# =============================================================================


class TestScrubberStateMachine:
    """Test state transitions of the scrubber."""

    def test_text_to_think(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("text<think>")
        assert scrubber.mode == "IN_THINK"

    def test_think_back_to_text(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<think>content</think>")
        # After consuming </think>, should be back in TEXT
        assert scrubber.mode == "TEXT"

    def test_text_to_toolcall(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("text<tool_call>")
        assert scrubber.mode == "IN_TOOLCALL"

    def test_toolcall_back_to_text(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<tool_call>data</tool_call>")
        assert scrubber.mode == "TEXT"

    def test_text_to_function(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("text<function=test>")
        assert scrubber.mode == "IN_FUNCTION"

    def test_function_back_to_text(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("<function=test>body</function>")
        assert scrubber.mode == "TEXT"

    def test_text_to_parameter_enters_function_mode(self):
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("text<parameter=x>")
        assert scrubber.mode == "IN_FUNCTION"

    def test_stray_close_stays_in_text(self):
        """Stray closing tags should not change mode."""
        scrubber = _AnthropicStreamScrubber()
        scrubber.feed("text</think>more")
        assert scrubber.mode == "TEXT"

        scrubber2 = _AnthropicStreamScrubber()
        scrubber2.feed("text</tool_call>more")
        assert scrubber2.mode == "TEXT"


# =============================================================================
# Edge Cases
# =============================================================================


class TestScrubberEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_think_block(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("before<think></think>after")
        result += scrubber.flush()
        assert "before" in result
        assert "after" in result
        assert "<think>" not in result

    def test_empty_tool_call_block(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("before<tool_call></tool_call>after")
        result += scrubber.flush()
        assert "before" in result
        assert "after" in result

    def test_angle_brackets_in_plain_text(self):
        """Plain < and > that aren't tags should eventually be emitted."""
        scrubber = _AnthropicStreamScrubber()
        # These don't form valid tags so should pass through
        result = scrubber.feed("x < y and a > b are normal math expressions here")
        result += scrubber.flush()
        assert "x < y" in result or ("x" in result and "< y" in result)
        assert "a > b" in result or ("a" in result and "> b" in result)

    def test_partial_tag_that_is_not_a_tag(self):
        """Something like '<thinkable>' shouldn't be treated as <think>."""
        scrubber = _AnthropicStreamScrubber()
        # "<think" prefix matches but "able>" is not the same as ">"
        # Actually <think> is an exact match so <thinkable> has <think> inside it
        # The scrubber will find <think> at the start... let's see
        # Actually "<thinkable>" contains "<think" but not "<think>" exactly
        # Let me re-check: "<thinkable>" – scanning for "<think>" won't match
        # because the 7th char 'a' != '>'
        result = scrubber.feed("a<thinkable>test")
        result += scrubber.flush()
        # <thinkable> is not a recognized tag, should pass through
        assert "test" in result

    def test_only_tags_no_text(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("<think>hidden</think>")
        result += scrubber.flush()
        assert result == ""

    def test_consecutive_think_blocks_no_gap(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("<think>aaa</think><think>bbb</think>visible")
        result += scrubber.flush()
        assert "aaa" not in result  # suppressed
        assert "bbb" not in result  # suppressed
        assert "visible" in result

    def test_very_long_suppressed_content(self):
        """Test with large content inside tags."""
        scrubber = _AnthropicStreamScrubber()
        long_content = "x" * 10000
        text = f"before<think>{long_content}</think>after"
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert long_content not in result
        assert "before" in result
        assert "after" in result

    def test_unicode_text_preserved(self):
        """Unicode text outside tags should be preserved."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Héllo wörld 你好 🌍<think>secret</think> done")
        result += scrubber.flush()
        assert "secret" not in result
        assert "Héllo" in result
        assert "done" in result

    def test_newlines_between_tags(self):
        scrubber = _AnthropicStreamScrubber()
        text = "line1\n<think>hidden\n</think>\nline2"
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "hidden" not in result
        assert "line1" in result
        assert "line2" in result

    def test_back_to_back_different_tags(self):
        scrubber = _AnthropicStreamScrubber()
        text = '<think>t</think><tool_call>tc</tool_call><function=f>fc</function>end'
        result = scrubber.feed(text)
        result += scrubber.flush()
        assert "end" in result
        # All tagged content suppressed
        for s in ["<think>", "</think>", "<tool_call>", "</tool_call>",
                   "<function=", "</function>"]:
            assert s not in result


# =============================================================================
# _find_earliest_marker Internal Method
# =============================================================================


class TestFindEarliestMarker:
    """Test the _find_earliest_marker helper directly."""

    def test_no_markers(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("hello world", 0)
        assert result is None

    def test_finds_think_open(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("text<think>more", 0)
        assert result is not None
        pos, marker, consume = result
        assert pos == 4
        assert marker == "<think>"
        assert consume == len("<think>")

    def test_finds_earliest_of_multiple(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("<think>text<tool_call>", 0)
        assert result is not None
        pos, marker, _ = result
        assert pos == 0
        assert marker == "<think>"

    def test_finds_function_prefix(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("text<function=test>more", 0)
        assert result is not None
        pos, marker, consume = result
        assert pos == 4
        assert marker == "<function="
        assert consume == len("<function=test>")

    def test_function_prefix_missing_close_angle(self):
        """If '>' is missing for a prefix tag, consume should be -1."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("text<function=test", 0)
        assert result is not None
        pos, marker, consume = result
        assert pos == 4
        assert marker == "<function="
        assert consume == -1  # truncated

    def test_start_offset(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("<think>more<think>", 7)
        assert result is not None
        pos, _, _ = result
        assert pos == 11  # second <think>

    def test_parameter_prefix(self):
        scrubber = _AnthropicStreamScrubber()
        result = scrubber._find_earliest_marker("<parameter=name>val", 0)
        assert result is not None
        pos, marker, consume = result
        assert pos == 0
        assert marker == "<parameter="
        assert consume == len("<parameter=name>")


# =============================================================================
# Integration with Scrubber in Streaming Context
# =============================================================================


class TestScrubberStreamingIntegration:
    """Simulate real streaming patterns to ensure correctness end-to-end."""

    def _stream_through(self, scrubber, deltas):
        """Feed a list of deltas through the scrubber, return collected output."""
        collected = ""
        for d in deltas:
            collected += scrubber.feed(d)
        collected += scrubber.flush()
        return collected

    def test_clean_text_passthrough(self):
        """Normal text with no tags should come through unchanged."""
        scrubber = _AnthropicStreamScrubber()
        text = "The weather today is sunny and warm."
        words = text.split(" ")
        deltas = [w + " " for w in words[:-1]] + [words[-1]]
        result = self._stream_through(scrubber, deltas)
        assert result == text

    def test_think_then_answer_streaming(self):
        """Model thinks, then answers."""
        scrubber = _AnthropicStreamScrubber()
        deltas = [
            "<think>",
            "Let me reason about this...\n",
            "The user wants weather info.\n",
            "</think>",
            "The weather ",
            "is sunny ",
            "today.",
        ]
        result = self._stream_through(scrubber, deltas)
        assert "reason" not in result
        assert "The weather is sunny today." in result

    def test_tool_call_json_streaming(self):
        """Model emits tool call JSON in small chunks."""
        scrubber = _AnthropicStreamScrubber()
        deltas = [
            "Let me look that up.",
            "<tool_call>",
            '{"',
            'name": "',
            'get_weather',
            '", "arguments',
            '": {"city": "',
            'San Francisco',
            '"}}',
            "</tool_call>",
        ]
        result = self._stream_through(scrubber, deltas)
        assert "Let me look that up." in result
        assert "get_weather" not in result
        assert "San Francisco" not in result

    def test_think_then_tool_call_streaming(self):
        """Model reasons then makes a tool call."""
        scrubber = _AnthropicStreamScrubber()
        deltas = [
            "<think>",
            "I need to ",
            "search for this.",
            "</think>",
            "I'll help with that.",
            "<tool_call>",
            '{"name":"search"}',
            "</tool_call>",
        ]
        result = self._stream_through(scrubber, deltas)
        assert "I need to" not in result
        assert "I'll help with that." in result
        assert "search" not in result or result == "I'll help with that."

    def test_scrubber_reuse_not_recommended(self):
        """After flush, feeding more data should still work (even if atypical)."""
        scrubber = _AnthropicStreamScrubber()
        r1 = scrubber.feed("first")
        r1 += scrubber.flush()
        # Reuse
        r2 = scrubber.feed("second")
        r2 += scrubber.flush()
        assert "first" in r1
        assert "second" in r2


# =============================================================================
# Zero-Latency Carry: plain text should not be held back
# =============================================================================


class TestScrubberZeroLatencyCarry:
    """Verify that the conditional carry buffer doesn't stall plain text."""

    def test_plain_text_emits_immediately(self):
        """No '<' in text → carry should be empty, all text emitted."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Hello world")
        assert result == "Hello world"
        assert scrubber.carry == ""

    def test_plain_deltas_no_carry(self):
        """Multiple plain deltas should each emit fully."""
        scrubber = _AnthropicStreamScrubber()
        for word in ["The ", "quick ", "brown ", "fox."]:
            result = scrubber.feed(word)
            assert result == word
            assert scrubber.carry == ""

    def test_angle_bracket_at_end_triggers_carry(self):
        """A '<' near the end should be held in carry (could be tag start)."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("text<")
        assert "<" not in result
        assert scrubber.carry == "<"

    def test_angle_bracket_resolved_next_delta(self):
        """Carry '<' is resolved when next delta shows it's not a tag."""
        scrubber = _AnthropicStreamScrubber()
        r1 = scrubber.feed("value < ")
        r2 = scrubber.feed("other")
        r2 += scrubber.flush()
        full = r1 + r2
        assert "value < other" in full

    def test_angle_bracket_resolved_as_tag(self):
        """Carry '<' is resolved when next delta completes a tag."""
        scrubber = _AnthropicStreamScrubber()
        r1 = scrubber.feed("before<")
        r2 = scrubber.feed("think>hidden</think>after")
        r2 += scrubber.flush()
        full = r1 + r2
        assert "before" in full
        assert "after" in full
        assert "hidden" not in full

    def test_first_token_emits_immediately(self):
        """The very first token should not be stalled by carry buffer."""
        scrubber = _AnthropicStreamScrubber()
        result = scrubber.feed("Hi")
        assert result == "Hi"

    def test_long_plain_text_no_carry(self):
        """Long text with no '<' should all be emitted, carry empty."""
        scrubber = _AnthropicStreamScrubber()
        text = "A" * 500
        result = scrubber.feed(text)
        assert result == text
        assert scrubber.carry == ""
