# SPDX-License-Identifier: Apache-2.0
"""
Tests for GPT-OSS harmony-format tool call parsing at the server boundary.

Tests exercise _extract_reasoning_and_tool_calls directly: when the model emits
an analysis (reasoning) channel followed by a commentary channel, the analysis
block must be stripped before the text is handed to the tool parser so reasoning
prose is never run through the generic fallback (which can extract spurious tool
calls from JSON-shaped text) and never duplicates into response content.

Usage:
    pytest tests/test_gpt_oss_tool_parsing.py -v
"""

from types import SimpleNamespace


def test_tools_branch_strips_analysis_block_before_parser(monkeypatch):
    import vllm_mlx.server as server

    class FakeReasoningParser:
        def extract_reasoning(self, text):
            return (
                'I will use {"name": "get_weather", "arguments": {"city": "SF"}}',
                None,
            )

    seen = []

    def fake_parse(text, request, **_):
        seen.append(text)
        return text, None

    raw = (
        '<|channel|>analysis<|message|>I will use {"name": "get_weather", '
        '"arguments": {"city": "SF"}}<|end|>'
        "<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city": "S'
    )
    request = SimpleNamespace(tools=[{"type": "function"}])
    monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
    monkeypatch.setattr(server, "_parse_tool_calls_with_parser", fake_parse)

    reasoning, cleaned, tool_calls = server._extract_reasoning_and_tool_calls(
        raw, request
    )

    assert reasoning is not None and reasoning != ""
    assert tool_calls is None
    assert len(seen) == 1
    assert "<|channel|>analysis" not in seen[0]
    assert seen[0] == (
        "<|end|><|channel|>commentary to=functions.get_weather" '<|message|>{"city": "S'
    )


def test_tools_branch_analysis_then_commentary_call_passes_through(monkeypatch):
    """Happy path: analysis channel followed by a commentary channel with a
    <|call|> terminator. The call must still be passed through to the parser
    (commentary block preserved), but the analysis block must be stripped."""
    import vllm_mlx.server as server

    class FakeReasoningParser:
        def extract_reasoning(self, text):
            return "analysis content", None

    seen = []

    def fake_parse(text, request, **_):
        seen.append(text)
        return None, ["call_extracted"]

    raw = (
        "<|channel|>analysis<|message|>thinking..."
        "<|channel|>commentary to=functions.read_file"
        '<|message|>{"path":"/etc/hosts"}<|call|>'
    )
    request = SimpleNamespace(tools=[{"type": "function"}])
    monkeypatch.setattr(server, "_reasoning_parser", FakeReasoningParser())
    monkeypatch.setattr(server, "_parse_tool_calls_with_parser", fake_parse)

    reasoning, cleaned, tool_calls = server._extract_reasoning_and_tool_calls(
        raw, request
    )

    assert reasoning == "analysis content"
    assert tool_calls == ["call_extracted"]
    assert len(seen) == 1
    assert "<|channel|>analysis" not in seen[0]
    assert seen[0] == (
        "<|channel|>commentary to=functions.read_file"
        '<|message|>{"path":"/etc/hosts"}<|call|>'
    )


# ============================================================================
# Streaming gate activation for gpt-oss harmony commentary
# ============================================================================


def test_streaming_marker_possible_for_commentary():
    """The streaming marker gate must activate for a gpt-oss commentary block.

    _STREAMING_TOOL_MARKERS now includes "<|channel|>commentary" and "<|call|>",
    otherwise _streaming_tool_markup_possible would never fire for harmony
    tool calls and the streaming parser path would stay dormant.
    """
    import vllm_mlx.server as server

    commentary = (
        "<|channel|>commentary to=functions.get_weather"
        '<|message|>{"city": "SF"}<|call|>'
    )
    assert server._streaming_tool_markup_possible(commentary) is True

    # A commentary block that starts before the current delta also activates
    # the gate (boundary-window scan).
    assert (
        server._streaming_tool_markup_possible_after_delta(
            "prefix text ", "<|channel|>commentary to=functions.get_weather"
        )
        is True
    )
    # The terminator alone is a marker too (e.g. split across chunks).
    assert (
        server._streaming_tool_markup_possible_after_delta(
            "<|channel|>commentary to=functions.get_weather"
            '<|message|>{"city": "SF"}',
            "<|call|>",
        )
        is True
    )


def test_streaming_marker_false_for_plain_text():
    """Plain content must not activate the harmony tool-markup gate."""
    import vllm_mlx.server as server

    plain = "The weather in San Francisco is 72 degrees and sunny."
    assert server._streaming_tool_markup_possible(plain) is False
    assert server._streaming_tool_markup_possible_after_delta("", "72 degrees") is False
    # Final-channel content (no commentary) is not tool markup either.
    final_only = "<|channel|>final<|message|>The answer is 42.<|return|>"
    assert server._streaming_tool_markup_possible(final_only) is False
