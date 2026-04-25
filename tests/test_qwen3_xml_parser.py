# SPDX-License-Identifier: Apache-2.0
"""Functional tests for Qwen3XMLToolParser parsing logic."""

import json

import pytest

pytest.importorskip("transformers")

from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
    Qwen3XMLToolParser,
    StreamingXMLToolCallParser,
)


@pytest.fixture
def parser():
    return Qwen3XMLToolParser(None)


@pytest.fixture
def streaming_parser():
    return StreamingXMLToolCallParser()


class TestNonStreamingExtraction:
    """Non-streaming extract_tool_calls on complete text."""

    def test_single_tool_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Tokyo"

    def test_multiple_tool_calls(self, parser):
        text = (
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>/a.py</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>/b.py</parameter>\n"
            "<parameter=content>hello</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[1]["name"] == "write_file"

    def test_type_coercion_int(self, parser):
        text = (
            "<tool_call>\n"
            "<function=set_temp>\n"
            "<parameter=value>42</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "set_temp",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                        },
                    },
                }
            ]
        }
        result = parser.extract_tool_calls(text, request=request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["value"] == 42
        assert isinstance(args["value"], int)

    def test_type_coercion_bool(self, parser):
        text = (
            "<tool_call>\n"
            "<function=toggle>\n"
            "<parameter=enabled>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "toggle",
                        "parameters": {
                            "type": "object",
                            "properties": {"enabled": {"type": "boolean"}},
                        },
                    },
                }
            ]
        }
        result = parser.extract_tool_calls(text, request=request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["enabled"] is True

    def test_think_tags_stripped(self, parser):
        text = (
            "<think>I should check the weather.</think>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_no_tool_calls(self, parser):
        text = "Just a normal response with no tool calls."
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_multiline_parameter(self, parser):
        text = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>/src/hello.py</parameter>\n"
            "<parameter=content>def hello():\n"
            "    print('hello')\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert "def hello():" in args["content"]

    def test_special_characters_in_value(self, parser):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>x > 5 & y < 10</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert ">" in args["query"]
        assert "&" in args["query"]


class TestStreamingParser:
    """Streaming incremental parsing via Qwen3XMLToolParser wrapper."""

    def test_streaming_produces_deltas(self, parser):
        """Streaming extract produces tool call deltas."""
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        deltas = []
        accumulated = ""
        for i in range(0, len(text), 10):
            prev = accumulated
            chunk = text[i : i + 10]
            accumulated += chunk
            result = parser.extract_tool_calls_streaming(prev, accumulated, chunk)
            if result is not None:
                deltas.append(result)
        assert len(deltas) > 0


class TestMalformedXML:
    """Edge cases with malformed model output."""

    def test_unclosed_tool_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=f>\n"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "f"

    def test_missing_function_close(self, parser):
        text = (
            "<tool_call>\n"
            "<function=f>\n"
            "<parameter=x>1</parameter>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
