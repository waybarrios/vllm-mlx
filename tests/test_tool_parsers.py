# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for tool call parsers."""

import json

import pytest

from vllm_mlx.tool_parsers import (
    AutoToolParser,
    DeepSeekToolParser,
    FunctionaryToolParser,
    GraniteToolParser,
    HermesToolParser,
    KimiToolParser,
    LlamaToolParser,
    MistralToolParser,
    NemotronToolParser,
    QwenToolParser,
    ToolParserManager,
    xLAMToolParser,
)


class TestToolParserManager:
    """Test the ToolParserManager registry."""

    def test_list_registered(self):
        """Test that all expected parsers are registered."""
        parsers = ToolParserManager.list_registered()
        expected = [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
        ]
        for p in expected:
            assert p in parsers, f"Parser '{p}' not found"

    def test_get_tool_parser_by_name(self):
        """Test getting parsers by name."""
        test_cases = [
            ("mistral", MistralToolParser),
            ("qwen", QwenToolParser),
            ("qwen3", QwenToolParser),
            ("llama", LlamaToolParser),
            ("llama3", LlamaToolParser),
            ("llama4", LlamaToolParser),
            ("auto", AutoToolParser),
            ("deepseek", DeepSeekToolParser),
            ("deepseek_v3", DeepSeekToolParser),
            ("deepseek_r1", DeepSeekToolParser),
            ("kimi", KimiToolParser),
            ("kimi_k2", KimiToolParser),
            ("moonshot", KimiToolParser),
            ("granite", GraniteToolParser),
            ("granite3", GraniteToolParser),
            ("nemotron", NemotronToolParser),
            ("nemotron3", NemotronToolParser),
            ("xlam", xLAMToolParser),
            ("functionary", FunctionaryToolParser),
            ("meetkai", FunctionaryToolParser),
            ("hermes", HermesToolParser),
            ("nous", HermesToolParser),
        ]
        for name, expected_cls in test_cases:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert parser_cls == expected_cls, f"Parser '{name}' returned wrong class"

    def test_get_unknown_parser_raises(self):
        """Test that unknown parser raises KeyError."""
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("unknown_parser")

    def test_parser_instantiation(self):
        """Test that all parsers can be instantiated without tokenizer."""
        for name in [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            parser = parser_cls()  # Should not raise
            assert parser is not None


class TestMistralToolParser:
    """Test the Mistral tool parser."""

    @pytest.fixture
    def parser(self):
        return MistralToolParser()

    def test_old_format_single(self, parser):
        """Test parsing old Mistral format with single tool call."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"

    def test_old_format_multiple(self, parser):
        """Test parsing old Mistral format with multiple tool calls."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": {"timezone": "UTC"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_new_format(self, parser):
        """Test parsing new Mistral format."""
        text = '[TOOL_CALLS]get_weather{"city": "London"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_no_tool_call(self, parser):
        """Test that regular text is not parsed as tool call."""
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_content_with_tool_call(self, parser):
        """Test content before tool call is preserved."""
        text = 'Let me check the weather for you.[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me check the weather for you."


class TestQwenToolParser:
    """Test the Qwen tool parser."""

    @pytest.fixture
    def parser(self):
        return QwenToolParser()

    def test_xml_format(self, parser):
        """Test parsing Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {"x": 1, "y": 2}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_bracket_format(self, parser):
        """Test parsing Qwen bracket format (Qwen3 style)."""
        text = '[Calling tool: add({"a": 5, "b": 3})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add"

    def test_multiple_xml_calls(self, parser):
        """Test multiple XML tool calls."""
        text = '<tool_call>{"name": "func1", "arguments": {}}</tool_call><tool_call>{"name": "func2", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I can help you with that question."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestLlamaToolParser:
    """Test the Llama tool parser."""

    @pytest.fixture
    def parser(self):
        return LlamaToolParser()

    def test_function_format(self, parser):
        """Test parsing Llama function format."""
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"

    def test_multiple_functions(self, parser):
        """Test parsing multiple function calls."""
        text = '<function=add>{"a": 1}</function><function=multiply>{"x": 3}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[1]["name"] == "multiply"

    def test_content_with_function(self, parser):
        """Test content before function call."""
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Computing result"


class TestHermesToolParser:
    """Test the Hermes tool parser."""

    @pytest.fixture
    def parser(self):
        return HermesToolParser()

    def test_tool_call_format(self, parser):
        """Test parsing Hermes format."""
        text = (
            '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_with_reasoning(self, parser):
        """Test with reasoning block."""
        text = '<tool_call_reasoning>I need to search for this</tool_call_reasoning><tool_call>{"name": "search", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert "Reasoning" in (result.content or "")


class TestDeepSeekToolParser:
    """Test the DeepSeek tool parser."""

    @pytest.fixture
    def parser(self):
        return DeepSeekToolParser()

    def test_deepseek_format(self, parser):
        """Test parsing DeepSeek V3 format."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_multiple_calls(self, parser):
        """Test multiple DeepSeek tool calls."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func1
```json
{"a": 1}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func2
```json
{"b": 2}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_content_before_tools(self, parser):
        """Test content before tool calls is preserved."""
        text = """Let me help you with that.<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me help you with that."

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is my response without any tool calls."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestKimiToolParser:
    """Test the Kimi tool parser."""

    @pytest.fixture
    def parser(self):
        return KimiToolParser()

    def test_kimi_format(self, parser):
        """Test parsing Kimi K2 format."""
        text = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>
<|tool_calls_section_end|>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_simple_function_name(self, parser):
        """Test with simple function name (no functions. prefix)."""
        text = (
            "<|tool_call_begin|>search:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I'll answer your question directly."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestGraniteToolParser:
    """Test the Granite tool parser."""

    @pytest.fixture
    def parser(self):
        return GraniteToolParser()

    def test_granite_30_format(self, parser):
        """Test parsing Granite 3.0 format."""
        text = (
            '<|tool_call|>[{"name": "calculate", "arguments": {"expression": "2+2"}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_granite_31_format(self, parser):
        """Test parsing Granite 3.1 format."""
        text = '<tool_call>[{"name": "search", "arguments": {"query": "test"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_multiple_calls(self, parser):
        """Test multiple tool calls."""
        text = '<|tool_call|>[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "The answer is 42."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestNemotronToolParser:
    """Test the Nemotron tool parser."""

    @pytest.fixture
    def parser(self):
        return NemotronToolParser()

    def test_parameter_format(self, parser):
        """Test parsing Nemotron parameter format."""
        text = "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=units>celsius</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"
        assert args["units"] == "celsius"

    def test_json_format(self, parser):
        """Test parsing Nemotron with JSON arguments."""
        text = '<tool_call><function=calculate>{"expression": "2*3"}</function></tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_multiple_calls(self, parser):
        """Test multiple Nemotron tool calls."""
        text = "<tool_call><function=func1><parameter=a>1</parameter></function></tool_call><tool_call><function=func2><parameter=b>2</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is the information you requested."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestXLAMToolParser:
    """Test the xLAM tool parser."""

    @pytest.fixture
    def parser(self):
        return xLAMToolParser()

    def test_json_array(self, parser):
        """Test parsing JSON array format."""
        text = '[{"name": "search", "arguments": {"query": "AI"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_code_block(self, parser):
        """Test parsing markdown code block."""
        text = '```json\n[{"name": "calculate", "arguments": {"x": 5}}]\n```'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_after_think(self, parser):
        """Test parsing after </think> tag."""
        text = (
            '<think>Let me search for this</think>[{"name": "search", "arguments": {}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_tool_calls_tag(self, parser):
        """Test [TOOL_CALLS] tag format."""
        text = '[TOOL_CALLS][{"name": "func", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I don't need to use any tools for this."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestFunctionaryToolParser:
    """Test the Functionary tool parser."""

    @pytest.fixture
    def parser(self):
        return FunctionaryToolParser()

    def test_recipient_format(self, parser):
        """Test parsing Functionary v3 recipient format."""
        text = '<|from|>assistant\n<|recipient|>get_weather\n<|content|>{"city": "NYC"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_function_format(self, parser):
        """Test parsing function format."""
        text = '<function=search>{"query": "test"}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_json_array(self, parser):
        """Test parsing JSON array."""
        text = '[{"name": "func1", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Let me explain that to you."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestAutoToolParser:
    """Test the auto-detecting tool parser."""

    @pytest.fixture
    def parser(self):
        return AutoToolParser()

    def test_detects_mistral(self, parser):
        """Test auto detection of Mistral format."""
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_qwen_xml(self, parser):
        """Test auto detection of Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_detects_qwen_bracket(self, parser):
        """Test auto detection of Qwen bracket format."""
        text = '[Calling tool: add({"a": 1})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "add"

    def test_detects_llama(self, parser):
        """Test auto detection of Llama format."""
        text = '<function=multiply>{"x": 2}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "multiply"

    def test_detects_nemotron(self, parser):
        """Test auto detection of Nemotron format."""
        text = "<tool_call><function=search><parameter=q>test</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_raw_json(self, parser):
        """Test auto detection of raw JSON format."""
        text = '{"name": "test_func", "arguments": {"key": "value"}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "test_func"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "This is just a regular response."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test with empty input."""
        parsers = [
            MistralToolParser(),
            QwenToolParser(),
            LlamaToolParser(),
            DeepSeekToolParser(),
            AutoToolParser(),
        ]
        for parser in parsers:
            result = parser.extract_tool_calls("")
            assert not result.tools_called

    def test_malformed_json(self):
        """Test with malformed JSON."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func", "arguments": {invalid json}]'
        result = parser.extract_tool_calls(text)
        # Should not crash, may or may not parse

    def test_nested_arguments(self):
        """Test with deeply nested arguments."""
        parser = AutoToolParser()
        args = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        text = f'{{"name": "complex", "arguments": {json.dumps(args)}}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["level1"]["level2"]["level3"] == [1, 2, 3]

    def test_unicode_in_arguments(self):
        """Test with unicode characters in arguments."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "translate", "arguments": {"text": "日本語"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語"

    def test_special_characters_in_name(self):
        """Test function names with special characters."""
        parser = LlamaToolParser()
        text = '<function=get_user_info>{"user_id": 123}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_user_info"

    def test_tool_call_id_uniqueness(self):
        """Test that each tool call gets a unique ID."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"


class TestStreamingParsing:
    """Test streaming tool call parsing."""

    def test_mistral_streaming(self):
        """Test Mistral streaming parsing."""
        parser = MistralToolParser()

        # Simulate streaming
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me",
            delta_text="Let me",
        )
        assert result1 == {"content": "Let me"}

        result2 = parser.extract_tool_calls_streaming(
            previous_text="Let me",
            current_text="Let me[TOOL_CALLS]",
            delta_text="[TOOL_CALLS]",
        )
        # Should start tool call parsing

    def test_auto_streaming(self):
        """Test auto parser streaming."""
        parser = AutoToolParser()

        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello world",
            delta_text="Hello world",
        )
        assert result == {"content": "Hello world"}


class TestThinkTagStripping:
    """Test <think> tag stripping in tool parsers (Issue #26)."""

    def test_strip_think_tags_utility(self):
        """Test the strip_think_tags static method."""
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        # Basic stripping
        text = "<think>Let me analyze this</think>The answer is 42"
        assert ToolParser.strip_think_tags(text) == "The answer is 42"

        # Multi-line thinking
        text = "<think>Step 1\nStep 2\nStep 3</think>Result"
        assert ToolParser.strip_think_tags(text) == "Result"

        # No think tags
        text = "Just regular text"
        assert ToolParser.strip_think_tags(text) == "Just regular text"

        # Empty think tags
        text = "<think></think>Content"
        assert ToolParser.strip_think_tags(text) == "Content"

    def test_hermes_with_think_tags(self):
        """Test Hermes parser strips think tags before parsing tool calls."""
        parser = HermesToolParser()

        # Model output with think tags AND tool call (Ring-Mini-Linear-2.0 style)
        output = """<think>Let me search for that information.</think>
<tool_call>{"name": "search", "arguments": {"query": "weather"}}</tool_call>"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_qwen_with_think_tags(self):
        """Test Qwen parser strips think tags before parsing tool calls."""
        parser = QwenToolParser()

        # Model output with think tags AND tool call
        output = """<think>I need to get the weather data.</think>
[Calling tool: get_weather({"city": "Tokyo"})]"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_think_tags_with_no_tool_call(self):
        """Test that think tags are stripped even when no tool call is present."""
        parser = HermesToolParser()

        output = "<think>Let me think about this</think>The answer is 42."
        result = parser.extract_tool_calls(output)

        assert result.tools_called is False
        assert result.content == "The answer is 42."
