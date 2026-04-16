#!/usr/bin/env python3
"""Unit tests for Gemma4 streaming parser edge cases."""

import sys

sys.path.insert(0, "/Users/janhilgard/vllm-mlx-upstream")

from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser


def stream_parse(parser, text, token_list):
    """Feed token_list to parser one-by-one, accumulating reasoning/content."""
    parser.reset_state()
    accumulated = ""
    reasoning_parts = []
    content_parts = []
    for tok in token_list:
        prev = accumulated
        accumulated += tok
        msg = parser.extract_reasoning_streaming(prev, accumulated, tok)
        if msg is None:
            continue
        if msg.reasoning:
            reasoning_parts.append(msg.reasoning)
        if msg.content:
            content_parts.append(msg.content)
    return "".join(reasoning_parts), "".join(content_parts)


def test_channel_marker_split_across_tokens():
    """<|channel> alone should not leak into reasoning if response follows."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "reasoning text",
        "<|channel>",
        "response",
        "\n",
        "content text",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    assert "<|channel>" not in reasoning, f"Marker leaked into reasoning: {reasoning!r}"
    assert "<|channel>" not in content, f"Marker leaked into content: {content!r}"
    assert reasoning == "reasoning text", f"Got: {reasoning!r}"
    assert content == "content text", f"Got: {content!r}"
    print("[PASS] test_channel_marker_split_across_tokens")


def test_leading_newline_after_transition():
    """Leading \\n after <|channel>response should be stripped from content."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "reasoning",
        "<|channel>response",
        "\n",  # marker in one token, \n in next
        "actual content",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    assert content == "actual content", f"Got: {content!r}"
    print("[PASS] test_leading_newline_after_transition")


def test_realistic_deterministic_production_stream():
    """Reproduce realistic production output with many small deltas."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "The",
        " user",
        " said",
        ' "',
        "hi",
        '".',
        " Plan",
        ":",
        " greet",
        " politely",
        ".",
        "\n",
        "<|channel>",
        "response",
        "\n",
        "Hello",
        "!",
        " How",
        " can",
        " I",
        " help",
        " you",
        " today",
        "?",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    assert "<|channel>" not in reasoning, f"Marker leaked: {reasoning!r}"
    assert "<|channel>" not in content, f"Marker leaked: {content!r}"
    assert (
        reasoning == 'The user said "hi". Plan: greet politely.\n'
    ), f"Got: {reasoning!r}"
    assert content == "Hello! How can I help you today?", f"Got: {content!r}"
    print("[PASS] test_realistic_deterministic_production_stream")


def test_standard_format_with_split():
    """Standard <|channel>thought...<channel|>content format, split token."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "reasoning",
        "<channel|>",
        "content",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    assert reasoning == "reasoning", f"Got: {reasoning!r}"
    assert content == "content", f"Got: {content!r}"
    print("[PASS] test_standard_format_with_split")


def test_no_transition_stays_in_thinking():
    """When model never emits transition marker, all goes to reasoning."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "reasoning with",
        " no",
        " transition",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    assert reasoning == "reasoning with no transition", f"Got: {reasoning!r}"
    assert content == "", f"Got: {content!r}"
    print("[PASS] test_no_transition_stays_in_thinking")


def test_finalize_stream_fallback():
    """B: If stream ends in thinking phase, finalize should allow fallback."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "draft answer here",
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    # Should still classify as reasoning (current correct behavior)
    assert reasoning == "draft answer here", f"Got: {reasoning!r}"
    # Finalize hook should return any pending partial marker buffer
    msg = parser.finalize_stream()
    # If parser buffered a partial <|channel> at end of stream, it should flush here
    assert msg is None or msg.reasoning is not None or msg.content is not None
    print("[PASS] test_finalize_stream_fallback")


def test_partial_marker_at_end_flushed():
    """If stream ends with buffered partial marker, finalize emits it."""
    parser = Gemma4ReasoningParser()
    tokens = [
        "<|channel>",
        "thought",
        "\n",
        "text",
        "<|channel>",  # partial — could be transition or just end
    ]
    reasoning, content = stream_parse(parser, None, tokens)
    # At this point, "<|channel>" should be buffered, not emitted
    assert "<|channel>" not in reasoning, f"Leaked: {reasoning!r}"
    # Finalize flushes the buffered marker as reasoning (it wasn't transition)
    msg = parser.finalize_stream()
    if msg and msg.reasoning:
        reasoning += msg.reasoning
    assert reasoning == "text<|channel>", f"Got: {reasoning!r}"
    print("[PASS] test_partial_marker_at_end_flushed")


if __name__ == "__main__":
    test_channel_marker_split_across_tokens()
    test_leading_newline_after_transition()
    test_realistic_deterministic_production_stream()
    test_standard_format_with_split()
    test_no_transition_stays_in_thinking()
    test_finalize_stream_fallback()
    test_partial_marker_at_end_flushed()
    print("\nAll tests passed!")
