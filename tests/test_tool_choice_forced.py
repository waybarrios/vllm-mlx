# SPDX-License-Identifier: Apache-2.0
"""Tests for forced tool_choice support.

Covers:
- tool_choice={"type":"function","function":{"name":"X"}} (forced specific tool)
- tool_choice="required" (must call some tool)
- QwenToolParser.SUPPORTS_NATIVE_TOOL_FORMAT
- QwenToolParser empty <tool_call> wrapper cleanup
"""

import json

import pytest

from vllm_mlx.server import (
    _apply_forced_tool_choice,
    _get_forced_tool_name,
    _tool_name,
)
from vllm_mlx.tool_parsers import QwenToolParser

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculate math",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


# ---------------------------------------------------------------------------
# _get_forced_tool_name
# ---------------------------------------------------------------------------


class TestGetForcedToolName:
    """Test extraction of forced tool name from tool_choice."""

    def test_dict_with_function_name(self):
        tc = {"type": "function", "function": {"name": "calculate"}}
        assert _get_forced_tool_name(tc) == "calculate"

    def test_dict_missing_function_key(self):
        tc = {"type": "function"}
        assert _get_forced_tool_name(tc) is None

    def test_dict_wrong_type(self):
        tc = {"type": "tool", "function": {"name": "foo"}}
        assert _get_forced_tool_name(tc) is None

    def test_string_auto(self):
        assert _get_forced_tool_name("auto") is None

    def test_string_none(self):
        assert _get_forced_tool_name("none") is None

    def test_string_required(self):
        assert _get_forced_tool_name("required") is None

    def test_none_value(self):
        assert _get_forced_tool_name(None) is None

    def test_empty_dict(self):
        assert _get_forced_tool_name({}) is None


# ---------------------------------------------------------------------------
# _tool_name
# ---------------------------------------------------------------------------


class TestToolName:
    """Test _tool_name helper."""

    def test_extracts_name(self):
        assert _tool_name(WEATHER_TOOL) == "get_weather"

    def test_no_function_key(self):
        assert _tool_name({"type": "function"}) is None

    def test_function_not_dict(self):
        assert _tool_name({"function": "not_a_dict"}) is None


# ---------------------------------------------------------------------------
# _apply_forced_tool_choice
# ---------------------------------------------------------------------------


class TestApplyForcedToolChoice:
    """Test _apply_forced_tool_choice for forced and required modes."""

    def _make_messages(self):
        return [{"role": "user", "content": "Hello"}]

    def test_forced_specific_tool_filters(self):
        """Forced tool_choice should filter tools to only the named one."""
        tools = [WEATHER_TOOL, CALC_TOOL]
        msgs = self._make_messages()
        tc = {"type": "function", "function": {"name": "calculate"}}

        new_tools, new_msgs = _apply_forced_tool_choice(tc, tools, msgs)

        assert len(new_tools) == 1
        assert _tool_name(new_tools[0]) == "calculate"

    def test_forced_specific_tool_injects_instruction(self):
        """Forced tool_choice should inject a system instruction."""
        tools = [WEATHER_TOOL, CALC_TOOL]
        msgs = self._make_messages()
        tc = {"type": "function", "function": {"name": "calculate"}}

        _, new_msgs = _apply_forced_tool_choice(tc, tools, msgs)

        # Should have a system message prepended or appended
        system_msgs = [m for m in new_msgs if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert "calculate" in system_msgs[0]["content"]

    def test_forced_disables_thinking(self):
        """Forced tool_choice should set enable_thinking=False in chat_kwargs."""
        tools = [WEATHER_TOOL]
        msgs = self._make_messages()
        tc = {"type": "function", "function": {"name": "get_weather"}}
        kwargs = {}

        _apply_forced_tool_choice(tc, tools, msgs, kwargs)

        assert kwargs.get("enable_thinking") is False

    def test_forced_unknown_tool_raises(self):
        """Forced tool_choice with unknown function name should raise ValueError."""
        tools = [WEATHER_TOOL, CALC_TOOL]
        msgs = self._make_messages()
        tc = {"type": "function", "function": {"name": "nonexistent"}}

        with pytest.raises(ValueError, match="not found in tools"):
            _apply_forced_tool_choice(tc, tools, msgs)

    def test_required_injects_instruction(self):
        """tool_choice='required' should inject a system instruction."""
        tools = [WEATHER_TOOL, CALC_TOOL]
        msgs = self._make_messages()

        new_tools, new_msgs = _apply_forced_tool_choice("required", tools, msgs)

        # All tools kept
        assert len(new_tools) == 2
        # System instruction injected
        system_msgs = [m for m in new_msgs if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert "MUST" in system_msgs[0]["content"]

    def test_required_does_not_disable_thinking(self):
        """tool_choice='required' should NOT disable thinking."""
        tools = [WEATHER_TOOL]
        msgs = self._make_messages()
        kwargs = {}

        _apply_forced_tool_choice("required", tools, msgs, kwargs)

        assert "enable_thinking" not in kwargs

    def test_auto_is_noop(self):
        """tool_choice='auto' should not modify anything."""
        tools = [WEATHER_TOOL, CALC_TOOL]
        msgs = self._make_messages()

        new_tools, new_msgs = _apply_forced_tool_choice("auto", tools, msgs)

        assert new_tools is tools  # Same object, not modified
        assert new_msgs is msgs

    def test_none_tools_is_noop(self):
        """Empty tools list should be a no-op."""
        msgs = self._make_messages()

        new_tools, new_msgs = _apply_forced_tool_choice("required", None, msgs)

        assert new_tools is None
        assert new_msgs is msgs

    def test_forced_appends_to_existing_system(self):
        """If system message already exists, instruction should be appended."""
        tools = [WEATHER_TOOL]
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        tc = {"type": "function", "function": {"name": "get_weather"}}

        _, new_msgs = _apply_forced_tool_choice(tc, tools, msgs)

        system_content = new_msgs[0]["content"]
        assert system_content.startswith("You are helpful.")
        assert "get_weather" in system_content

    def test_does_not_mutate_original_messages(self):
        """Original messages list should not be mutated."""
        tools = [WEATHER_TOOL]
        msgs = [{"role": "user", "content": "Hello"}]
        original_len = len(msgs)
        tc = {"type": "function", "function": {"name": "get_weather"}}

        _, new_msgs = _apply_forced_tool_choice(tc, tools, msgs)

        assert len(msgs) == original_len  # Original unchanged
        assert len(new_msgs) != original_len  # New one has system msg


# ---------------------------------------------------------------------------
# QwenToolParser — SUPPORTS_NATIVE_TOOL_FORMAT
# ---------------------------------------------------------------------------


class TestQwenNativeToolFormat:
    """Test QwenToolParser native tool format support."""

    def test_supports_native_format_class_attribute(self):
        assert QwenToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True

    def test_supports_native_format_method(self):
        assert QwenToolParser.supports_native_format() is True


# ---------------------------------------------------------------------------
# QwenToolParser — empty <tool_call> cleanup
# ---------------------------------------------------------------------------


class TestQwenToolCallCleanup:
    """Test that empty <tool_call> wrappers are cleaned from content."""

    def _parser(self):
        return QwenToolParser(tokenizer=None)

    def test_function_style_cleans_wrapper(self):
        """<tool_call><function=X>...</function></tool_call> should leave no content."""
        text = (
            "<tool_call>\n<function=get_weather>"
            "<parameter=location>Prague</parameter>"
            "</function>\n</tool_call>"
        )
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        # Content should be None or empty after cleanup
        assert not result.content

    def test_multiple_function_calls_clean_wrappers(self):
        """Multiple <tool_call> wrappers should all be cleaned."""
        text = (
            "<tool_call>\n<function=get_weather>"
            "<parameter=location>Paris</parameter>"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>"
            "<parameter=location>London</parameter>"
            "</function>\n</tool_call>"
        )
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert not result.content

    def test_text_before_tool_call_preserved(self):
        """Text before the tool call block should be preserved."""
        text = (
            "Let me check the weather.\n"
            "<tool_call>\n<function=get_weather>"
            "<parameter=location>Tokyo</parameter>"
            "</function>\n</tool_call>"
        )
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert result.content == "Let me check the weather."

    def test_xml_style_no_double_cleanup(self):
        """XML-style tool calls already clean their own tags."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert not result.content

    def test_function_with_json_args_cleans_wrapper(self):
        """<function=X>{"key":"val"}</function> wrapped in <tool_call> should clean."""
        text = (
            "<tool_call>\n"
            '<function=calculate>{"expression": "2+2"}</function>\n'
            "</tool_call>"
        )
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "calculate"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["expression"] == "2+2"
        assert not result.content

    def test_think_tags_stripped_before_cleanup(self):
        """<think> tags should be stripped, then <tool_call> cleaned."""
        text = (
            "<think>Let me think about this...</think>\n"
            "<tool_call>\n<function=get_weather>"
            "<parameter=location>Rome</parameter>"
            "</function>\n</tool_call>"
        )
        result = self._parser().extract_tool_calls(text)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "get_weather"
        assert not result.content
