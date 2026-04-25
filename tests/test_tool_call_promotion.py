# SPDX-License-Identifier: Apache-2.0
"""Tests for tool call promotion from reasoning to content."""

import logging

import pytest

from vllm_mlx.reasoning import get_parser


@pytest.fixture
def parser():
    cls = get_parser("qwen3")
    return cls()


class TestNonStreamingPromotion:
    """Non-streaming extract_reasoning() promotes <tool_call> from reasoning."""

    def test_closed_tool_call_inside_think_appended(self, parser):
        """Test case 1: closed block appended to content."""
        output = (
            "<think>I should check the weather.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "I should check the weather" in reasoning
        assert "<tool_call>" not in reasoning
        assert content is not None
        assert "<tool_call>" in content
        assert "get_weather" in content

    def test_tool_call_after_think_unchanged(self, parser):
        """Test case 2: tool call in content stays in content."""
        output = (
            "<think>Let me think about this.</think>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think about this."
        assert content is not None
        assert "<tool_call>" in content

    def test_multiple_tool_calls_inside_think(self, parser):
        """Test case 3: multiple closed blocks all appended."""
        output = (
            "<think>I need two lookups.\n"
            "<tool_call>\n"
            "<function=get_weather><parameter=city>Tokyo</parameter></function>\n"
            "</tool_call>\n"
            "Now the second one.\n"
            "<tool_call>\n"
            "<function=get_time><parameter=tz>JST</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "<tool_call>" not in reasoning
        assert "I need two lookups" in reasoning
        assert "Now the second one" in reasoning
        assert content is not None
        assert content.count("<tool_call>") == 2

    def test_truncated_unclosed_tool_call_prepended(self, parser):
        """Test case 4: unclosed tool call in reasoning prepended to content."""
        output = (
            "<think>Let me call the API.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "<tool_call>" in content
        assert "Let me call the API" in (reasoning or "")

    def test_hermes_json_tool_call(self, parser):
        """Test case 5: Hermes JSON format promoted."""
        output = (
            "<think>I will check.\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "<tool_call>" not in reasoning
        assert content is not None
        assert "get_weather" in content

    def test_prose_mention_not_promoted(self, parser):
        """Test case 6: prose mentioning <tool_call> without structure stays."""
        output = (
            "<think>The model should use <tool_call> to invoke functions. "
            "Then it should verify.</think>\n"
            "The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "should use" in reasoning
        assert content == "The answer is 42."

    def test_content_none_handled(self, parser):
        """Test case 7: content=None when reasoning is entire output."""
        output = (
            "<think>Checking.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</tool_call>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "<tool_call>" in content

    def test_tool_call_spanning_think_boundary(self, parser):
        """Test case 8: unclosed portion prepended, reassembles with content."""
        output = (
            "<think>R<tool_call>\n"
            "<function=f>\n"
            "</think>"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "C"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "<tool_call>" in content
        assert "</tool_call>" in content

    def test_closed_appended_preserves_existing_content(self, parser):
        """Closed block appended after existing post-think content."""
        output = (
            "<think>Let me check.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
            "Here is my answer."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Here is my answer." in content
        assert "<tool_call>" in content
        assert content.index("Here is my answer") < content.index("<tool_call>")

    def test_promotion_logs_warning(self, parser, caplog):
        """Promotion should log a warning for operators."""
        with caplog.at_level(logging.WARNING):
            output = (
                "<think>\n"
                "<tool_call>\n"
                "<function=f><parameter=x>1</parameter></function>\n"
                "</tool_call>\n"
                "</think>\n"
            )
            parser.extract_reasoning(output)
        assert any("tool_call" in r.message.lower() for r in caplog.records)

    def test_no_tool_calls_no_warning(self, parser, caplog):
        """No promotion -> no warning logged."""
        with caplog.at_level(logging.WARNING):
            output = "<think>Just reasoning.</think>\nContent."
            parser.extract_reasoning(output)
        assert not any("tool_call" in r.message.lower() for r in caplog.records)


class TestStreamingPromotion:
    """Streaming extract_reasoning_streaming() promotes tool calls.

    Uses full-text or large chunk sizes. The upstream streaming parser
    uses current_text/previous_text tag detection which has imprecision
    when tags span chunk boundaries at very small chunk sizes.
    """

    def _stream(self, parser, text, chunk_size=None):
        """Feed text through streaming parser in chunks."""
        if chunk_size is None:
            chunk_size = len(text)
        parser.reset_state()
        reasoning_parts = []
        content_parts = []
        accumulated = ""
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            previous = accumulated
            accumulated += chunk
            delta = parser.extract_reasoning_streaming(previous, accumulated, chunk)
            if delta:
                if delta.reasoning:
                    reasoning_parts.append(delta.reasoning)
                if delta.content:
                    content_parts.append(delta.content)
        final = parser.finalize_stream()
        if final:
            if final.reasoning:
                reasoning_parts.append(final.reasoning)
            if final.content:
                content_parts.append(final.content)
        return "".join(reasoning_parts) or None, "".join(content_parts) or None

    def test_stream_tool_call_inside_think(self, parser):
        """Tool call promoted as content during streaming."""
        text = (
            "<think>I should check.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
            "Final answer."
        )
        reasoning, content = self._stream(parser, text)
        assert reasoning is not None
        assert "I should check" in reasoning
        assert content is not None
        assert "<tool_call>" in content
        assert "get_weather" in content
        assert "Final answer." in content

    def test_stream_think_ends_while_buffering(self, parser):
        """</think> before </tool_call> flushes as content."""
        text = (
            "<think>Check.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</think>\n"
            "Done."
        )
        reasoning, content = self._stream(parser, text)
        assert content is not None
        assert "<tool_call>" in content

    def test_stream_finalize_with_buffered_tool_call(self, parser):
        """Stream ends mid-tool-call, flushed as content."""
        text = (
            "<think>\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
        )
        reasoning, content = self._stream(parser, text)
        assert content is not None
        assert "<tool_call>" in content

    def test_stream_multiple_tool_calls(self, parser):
        """Each tool call promoted independently."""
        text = (
            "<think>Two calls.\n"
            "<tool_call>\n"
            "<function=a><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "Middle reasoning.\n"
            "<tool_call>\n"
            "<function=b><parameter=y>2</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = self._stream(parser, text)
        assert "Two calls" in (reasoning or "")
        assert "Middle reasoning" in (reasoning or "")
        assert content is not None
        assert content.count("<tool_call>") == 2

    def test_stream_large_chunks(self, parser):
        """Promotion correct at chunk sizes that don't split tags."""
        text = (
            "<think>Check.\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\nDone."
        )
        for cs in [20, 50, len(text)]:
            reasoning, content = self._stream(parser, text, chunk_size=cs)
            assert content is not None, f"chunk_size={cs}"
            assert "<tool_call>" in content, f"chunk_size={cs}"

    def test_stream_no_tool_calls_regression(self, parser):
        """Normal reasoning unchanged when streamed as full text."""
        text = "<think>Just thinking here.</think>\nThe answer is 42."
        reasoning, content = self._stream(parser, text)
        assert "Just thinking here." in (reasoning or "")
        assert content is not None
        assert "The answer is 42." in content


class TestComposition:
    """End-to-end: reasoning parser + tool parser compose correctly."""

    def test_promoted_parsed_by_tool_parser(self, parser):
        """Tool parser finds promoted calls."""
        pytest.importorskip("transformers")
        from vllm_mlx.tool_parsers import ToolParserManager

        output = (
            "<think>Let me look this up.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None

        for parser_name in ["qwen3_xml", "qwen3.5", "qwen", "qwen3"]:
            try:
                tool_cls = ToolParserManager.get_tool_parser(parser_name)
                break
            except KeyError:
                continue
        else:
            pytest.skip("No Qwen tool parser registered")

        tool_parser = tool_cls(None)
        result = tool_parser.extract_tool_calls(content)
        assert result.tools_called
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_promoted_preserves_trailing_content(self, parser):
        """Closed appended after existing content."""
        output = (
            "<think>Checking.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
            "Based on results, here is my answer."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert "Based on results" in content
        assert "<tool_call>" in content

    def test_tool_choice_required_with_promotion(self, parser):
        """Required tool choice + promoted content."""
        output = (
            "<think>User requires a tool call.\n"
            "<tool_call>\n"
            "<function=mandatory_fn><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "mandatory_fn" in content
