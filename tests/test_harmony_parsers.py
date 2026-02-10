# SPDX-License-Identifier: Apache-2.0
"""
Tests for Harmony format parsers (GPT-OSS models).

Tests cover:
- HarmonyToolParser: tool call extraction from commentary channel
- HarmonyReasoningParser: reasoning extraction from analysis channel
- convert_tools_to_typescript: OpenAI JSON Schema to TypeScript conversion

Usage:
    pytest tests/test_harmony_parsers.py -v
"""

import json

import pytest

from vllm_mlx.api.harmony_tools import convert_tools_to_typescript
from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

# ============================================================================
# Tool Parser Tests
# ============================================================================


class TestHarmonyToolParser:
    """Tests for HarmonyToolParser."""

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_registration(self):
        """Parser is registered under harmony and gpt-oss names."""
        assert ToolParserManager.get_tool_parser("harmony") is HarmonyToolParser
        assert ToolParserManager.get_tool_parser("gpt-oss") is HarmonyToolParser

    def test_single_tool_call(self, parser):
        """Parse a single tool call from commentary channel."""
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "San Francisco", "unit": "celsius"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_tool_call_with_analysis_and_final(self, parser):
        """Parse tool call when analysis and final channels are present."""
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>The user wants weather. I should call get_weather.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_final_response_only(self, parser):
        """Parse response with no tool calls (final channel only)."""
        text = (
            "<|start|>\n"
            "<|channel|>final\n"
            "<|message|>The weather in San Francisco is 72F and sunny!\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "The weather in San Francisco is 72F and sunny!"

    def test_multiple_tool_calls(self, parser):
        """Parse multiple tool calls from separate commentary blocks."""
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.get_time\n"
            "<|constrain|>json\n"
            '<|message|>{"timezone": "PST"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_tool_call_without_constrain(self, parser):
        """Parse tool call without <|constrain|>json tag."""
        text = (
            "<|channel|>commentary to=functions.simple_func\n"
            '<|message|>{"arg": "value"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "simple_func"

    def test_malformed_json_arguments(self, parser):
        """Handle malformed JSON gracefully by keeping raw string."""
        text = (
            "<|channel|>commentary to=functions.broken_func\n"
            "<|constrain|>json\n"
            "<|message|>{invalid json here}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "broken_func"
        assert result.tool_calls[0]["arguments"] == "{invalid json here}"

    def test_tool_call_with_final_content(self, parser):
        """Tool calls coexist with final channel content."""
        text = (
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "python"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Here are the results.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Here are the results."

    def test_empty_input(self, parser):
        """Handle empty input."""
        result = parser.extract_tool_calls("")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_plain_text_input(self, parser):
        """Handle plain text with no Harmony tokens."""
        result = parser.extract_tool_calls("Just a regular response.")
        assert not result.tools_called
        assert result.content == "Just a regular response."

    def test_unique_tool_ids(self, parser):
        """Each tool call gets a unique ID."""
        text = (
            "<|channel|>commentary to=functions.func_a\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>\n"
            "<|channel|>commentary to=functions.func_b\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        ids = [tc["id"] for tc in result.tool_calls]
        assert len(set(ids)) == 2
        assert all(id_.startswith("call_") for id_ in ids)

    def test_nested_json_arguments(self, parser):
        """Parse tool call with nested JSON arguments."""
        args = {"filter": {"type": "range", "min": 0, "max": 100}, "sort": "asc"}
        text = (
            "<|channel|>commentary to=functions.query\n"
            "<|constrain|>json\n"
            f"<|message|>{json.dumps(args)}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["filter"]["type"] == "range"

    def test_streaming_no_tool_markers(self, parser):
        """Streaming: plain text passes through as content."""
        result = parser.extract_tool_calls_streaming("", "Hello", "Hello")
        assert result == {"content": "Hello"}

    def test_streaming_tool_call_complete(self, parser):
        """Streaming: emit tool calls when <|call|> appears."""
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls_streaming("", current, "<|call|>")

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "func"

    def test_streaming_building_tool_call(self, parser):
        """Streaming: suppress output while building tool call."""
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a":'
        )
        result = parser.extract_tool_calls_streaming("", current, '{"a":')
        assert result is None


# ============================================================================
# Reasoning Parser Tests
# ============================================================================


class TestHarmonyReasoningParser:
    """Tests for HarmonyReasoningParser."""

    @pytest.fixture()
    def parser(self):
        return HarmonyReasoningParser()

    def test_registration(self):
        """Parser is registered under the harmony name."""
        parser_cls = get_parser("harmony")
        assert parser_cls is HarmonyReasoningParser

    def test_extract_analysis_and_final(self, parser):
        """Extract reasoning from analysis and content from final."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>Let me think step by step.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>The answer is 42.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think step by step."
        assert content == "The answer is 42."

    def test_multiple_analysis_blocks(self, parser):
        """Concatenate multiple analysis blocks."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>First thought.\n"
            "<|end|>\n"
            "<|channel|>analysis\n"
            "<|message|>Second thought.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Result.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert "First thought." in reasoning
        assert "Second thought." in reasoning
        assert content == "Result."

    def test_no_analysis_channel(self, parser):
        """Output with no analysis returns None reasoning."""
        output = "<|channel|>final\n" "<|message|>Direct answer.\n" "<|return|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is None
        assert content == "Direct answer."

    def test_analysis_only_no_final(self, parser):
        """Output with only analysis returns None content."""
        output = "<|channel|>analysis\n" "<|message|>Just thinking...\n" "<|end|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Just thinking..."
        assert content is None

    def test_empty_input(self, parser):
        """Handle empty input."""
        reasoning, content = parser.extract_reasoning("")
        assert reasoning is None
        assert content is None

    def test_analysis_with_commentary_and_final(self, parser):
        """Ignore commentary channel, extract analysis and final."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>Need to call a tool.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Found results.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Need to call a tool."
        assert content == "Found results."

    def test_streaming_analysis_to_final(self, parser):
        """Streaming: emit reasoning for analysis, content for final."""
        parser.reset_state()

        # Channel switch to analysis
        r1 = parser.extract_reasoning_streaming(
            "", "<|channel|>analysis\n", "<|channel|>analysis\n"
        )
        assert r1 is None  # channel switch, no content yet

        # Message start
        r2 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n",
            "<|channel|>analysis\n<|message|>",
            "<|message|>",
        )
        assert r2 is None  # message start token

        # Reasoning content
        r3 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>",
            "<|channel|>analysis\n<|message|>Thinking",
            "Thinking",
        )
        assert r3 is not None
        assert r3.reasoning == "Thinking"
        assert r3.content is None

        # End of analysis
        r4 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking",
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|end|>",
        )
        assert r4 is None  # end token

        # Switch to final
        r5 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            "\n<|channel|>final\n",
        )
        assert r5 is None  # channel switch

        # Final message content
        prev = "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n<|message|>"
        parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            prev,
            "<|message|>",
        )
        r6 = parser.extract_reasoning_streaming(
            prev,
            prev + "Answer",
            "Answer",
        )
        assert r6 is not None
        assert r6.content == "Answer"
        assert r6.reasoning is None

    def test_streaming_reset(self, parser):
        """Reset clears internal state."""
        parser._current_channel = "analysis"
        parser._in_message = True
        parser.reset_state()
        assert parser._current_channel is None
        assert parser._in_message is False

    def test_streaming_commentary_suppressed(self, parser):
        """Streaming: commentary channel output is suppressed."""
        parser.reset_state()

        parser.extract_reasoning_streaming(
            "",
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n",
        )
        parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n<|message|>",
            "<|message|>",
        )
        r = parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n<|message|>",
            '<|channel|>commentary to=functions.f\n<|message|>{"a":1}',
            '{"a":1}',
        )
        assert r is None


# ============================================================================
# TypeScript Tool Converter Tests
# ============================================================================


class TestHarmonyToolDefinitionConverter:
    """Tests for convert_tools_to_typescript."""

    def test_simple_tool(self):
        """Convert a simple tool with required parameters."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "namespace functions" in result
        assert "get_weather" in result
        assert "location: string," in result
        assert "// Get weather for a location" in result

    def test_optional_parameters(self):
        """Optional parameters get ? suffix."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "required_param": {"type": "string"},
                            "optional_param": {"type": "number"},
                        },
                        "required": ["required_param"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "required_param: string," in result
        assert "optional_param?: number," in result

    def test_enum_type(self):
        """Enums become TypeScript union types."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_unit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert '"celsius" | "fahrenheit"' in result

    def test_multiple_tools(self):
        """Multiple tools in one namespace."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func_a",
                    "description": "First function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func_b",
                    "description": "Second function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "func_a" in result
        assert "func_b" in result
        assert "// First function" in result
        assert "// Second function" in result

    def test_no_tools(self):
        """None input returns None."""
        assert convert_tools_to_typescript(None) is None
        assert convert_tools_to_typescript([]) is None

    def test_no_parameters(self):
        """Tool with no parameters uses empty signature."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "type ping = () => any;" in result

    def test_array_type(self):
        """Array types convert to Array<type>."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "Array<string>" in result

    def test_boolean_and_integer_types(self):
        """Boolean and integer map correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "config",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "count": {"type": "integer"},
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "enabled?: boolean," in result
        assert "count?: number," in result

    def test_no_description(self):
        """Tool without description has no comment line."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_desc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "//" not in result
        assert "no_desc" in result

    def test_skips_non_function_types(self):
        """Non-function tools are skipped."""
        tools = [
            {"type": "retrieval"},
            {
                "type": "function",
                "function": {
                    "name": "real_func",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "real_func" in result
        assert "retrieval" not in result


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestHarmonyEdgeCases:
    """Edge case tests for Harmony parsers."""

    def test_tool_parser_incomplete_call(self):
        """Incomplete tool call (missing <|call|>) is not parsed."""
        parser = HarmonyToolParser()
        text = "<|channel|>commentary to=functions.func\n" '<|message|>{"arg": "value"}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_tool_parser_unicode_content(self):
        """Handle unicode in tool arguments."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.translate\n"
            "<|constrain|>json\n"
            '<|message|>{"text": "日本語テスト"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語テスト"

    def test_reasoning_parser_unicode_content(self):
        """Handle unicode in reasoning and content."""
        parser = HarmonyReasoningParser()
        output = (
            "<|channel|>analysis\n"
            "<|message|>让我想想...\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>答案是42。\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "让我想想..."
        assert content == "答案是42。"

    def test_mixed_channels_full_flow(self):
        """Full flow: analysis -> commentary -> analysis -> final."""
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 1.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 2.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Done.\n"
            "<|return|>"
        )

        # Tool parser finds tool calls
        tool_parser = HarmonyToolParser()
        tool_result = tool_parser.extract_tool_calls(text)
        assert tool_result.tools_called
        assert len(tool_result.tool_calls) == 1
        assert tool_result.tool_calls[0]["name"] == "search"
        assert tool_result.content == "Done."

        # Reasoning parser finds both analysis blocks
        reasoning_parser = HarmonyReasoningParser()
        reasoning, content = reasoning_parser.extract_reasoning(text)
        assert "Think 1." in reasoning
        assert "Think 2." in reasoning
        assert content == "Done."

    def test_tool_parser_empty_arguments(self):
        """Tool call with empty JSON arguments."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.ping\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_tool_parser_whitespace_handling(self):
        """Handle extra whitespace in Harmony format."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>  {"key": "value"}  \n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["key"] == "value"
