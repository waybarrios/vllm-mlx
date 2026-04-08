# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 tool call parser."""

import json

import pytest

from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


class TestGemma4ToolParserExtract:
    """Test extract_tool_calls on complete model output."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()

    def test_single_tool_call_string_arg(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/foo.py<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "read_file"
        args = json.loads(tc["arguments"])
        assert args == {"path": "/tmp/foo.py"}
        assert result.content is None

    def test_single_tool_call_numeric_arg(self):
        output = "<|tool_call>call:search{limit:10,verbose:false}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"limit": 10, "verbose": False}

    def test_mixed_types(self):
        output = '<|tool_call>call:search{query:<|"|>hello world<|"|>,limit:10,verbose:false}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"query": "hello world", "limit": 10, "verbose": False}

    def test_nested_object(self):
        output = '<|tool_call>call:configure{settings:{enabled:true,name:<|"|>test<|"|>}}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"settings": {"enabled": True, "name": "test"}}

    def test_array_argument(self):
        output = '<|tool_call>call:tag{items:[<|"|>foo<|"|>,<|"|>bar<|"|>]}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"items": ["foo", "bar"]}

    def test_multiple_tool_calls_in_one_block(self):
        output = (
            "<|tool_call>"
            'call:glob{pattern:<|"|>README*.md<|"|>}'
            'call:glob{pattern:<|"|>CONTRIBUTING.md<|"|>}'
            "<tool_call|>"
        )
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0 == {"pattern": "README*.md"}
        assert args1 == {"pattern": "CONTRIBUTING.md"}

    def test_content_before_tool_call(self):
        output = 'Let me read that file for you.\n<|tool_call>call:read_file{path:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert result.content == "Let me read that file for you."
        assert len(result.tool_calls) == 1

    def test_no_tool_calls(self):
        output = "Hello, how can I help you today?"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == output

    def test_empty_tool_call_block(self):
        output = "<|tool_call><tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_tool_call_id_generated(self):
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/a<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        tc = result.tool_calls[0]
        assert "id" in tc
        assert tc["id"].startswith("call_")

    def test_string_with_special_chars(self):
        output = '<|tool_call>call:write{content:<|"|>line1\\nline2<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["content"] == "line1\\nline2"

    def test_deeply_nested_objects(self):
        output = "<|tool_call>call:update{a:{b:{c:1,d:true}}}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"a": {"b": {"c": 1, "d": True}}}

    def test_null_value(self):
        output = "<|tool_call>call:clear{target:null}<tool_call|>"
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"target": None}

    def test_unicode_emoji_in_args(self):
        output = '<|tool_call>call:search{query:<|"|>hello world \U0001f30d \u4f60\u597d<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"query": "hello world \U0001f30d \u4f60\u597d"}

    def test_braces_inside_string_value(self):
        output = '<|tool_call>call:run{code:<|"|>if (x) { return y; }<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"code": "if (x) { return y; }"}

    def test_quoted_keys(self):
        output = '<|tool_call>call:read{<|"|>path<|"|>:<|"|>/tmp/foo<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"path": "/tmp/foo"}

    def test_think_tags_stripped(self):
        output = '<think>Let me think about this...</think><|tool_call>call:search{query:<|"|>test<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
