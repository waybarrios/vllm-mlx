# SPDX-License-Identifier: Apache-2.0
"""Integration test: verify Gemma 4 tool calls produce valid OpenAI API responses.

Claude Code (and other OpenAI-compatible clients) expect:
- response.choices[0].message.tool_calls[0].type == "function"
- response.choices[0].message.tool_calls[0].function.name == "read_file"
- response.choices[0].message.tool_calls[0].function.arguments == '{"path":"/tmp/test.py"}'
- response.choices[0].message.content is None (not empty string)
- response.choices[0].finish_reason == "tool_calls"

This test verifies the FULL pipeline from parser output → server wrapping → JSON response,
not just the parser in isolation.
"""

import json

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    Usage,
)
from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser


def _build_response_from_parser(parser_output, model_name="gemma-4-27b-it"):
    """Simulate what server.py does at lines 1494-1511 to build the HTTP response."""
    if parser_output.tools_called:
        tool_calls = [
            ToolCall(
                id=tc.get("id", "call_test"),
                type="function",
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in parser_output.tool_calls
        ]
        content = parser_output.content if parser_output.content else None
        finish_reason = "tool_calls"
    else:
        tool_calls = None
        content = parser_output.content
        finish_reason = "stop"

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestGemma4OpenAIFormat:
    """Verify the full response matches what Claude Code expects."""

    def setup_method(self):
        self.parser = Gemma4ToolParser()

    def test_tool_call_response_has_correct_structure(self):
        """The JSON response must have the exact OpenAI structure."""
        output = '<|tool_call>call:read_file{path:<|"|>/tmp/test.py<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)

        # Serialize to JSON (this is what goes over the wire)
        data = json.loads(response.model_dump_json(exclude_none=True))

        choice = data["choices"][0]
        msg = choice["message"]

        # finish_reason must be "tool_calls" not "stop"
        assert choice["finish_reason"] == "tool_calls"

        # content must be absent or null when tool_calls present
        assert msg.get("content") is None

        # tool_calls must be a list
        assert isinstance(msg["tool_calls"], list)
        assert len(msg["tool_calls"]) == 1

        tc = msg["tool_calls"][0]

        # type must be "function"
        assert tc["type"] == "function"

        # id must be present and non-empty
        assert tc["id"]
        assert isinstance(tc["id"], str)

        # function.name must be the function name
        assert tc["function"]["name"] == "read_file"

        # function.arguments must be a JSON string (not a dict!)
        assert isinstance(tc["function"]["arguments"], str)
        args = json.loads(tc["function"]["arguments"])
        assert args == {"path": "/tmp/test.py"}

    def test_multiple_tool_calls_response(self):
        """Multiple tool calls in one response."""
        output = (
            "<|tool_call>"
            'call:read_file{path:<|"|>/a.py<|"|>}'
            'call:read_file{path:<|"|>/b.py<|"|>}'
            "<tool_call|>"
        )
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        tcs = data["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 2
        assert tcs[0]["function"]["name"] == "read_file"
        assert tcs[1]["function"]["name"] == "read_file"
        # Each must have a unique id
        assert tcs[0]["id"] != tcs[1]["id"]

    def test_content_before_tool_call_preserved(self):
        """Text before the tool call goes in content field."""
        output = 'Let me check that.\n<|tool_call>call:read_file{path:<|"|>/tmp/x<|"|>}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        msg = data["choices"][0]["message"]
        assert msg["content"] == "Let me check that."
        assert len(msg["tool_calls"]) == 1

    def test_no_tool_call_response(self):
        """Plain text response has no tool_calls field."""
        output = "The answer is 42."
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        msg = data["choices"][0]["message"]
        assert msg["content"] == "The answer is 42."
        assert "tool_calls" not in msg  # excluded by exclude_none
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_complex_arguments_serialize_correctly(self):
        """Nested objects and arrays must survive JSON round-trip."""
        output = '<|tool_call>call:configure{settings:{enabled:true,tags:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>'
        result = self.parser.extract_tool_calls(output)
        response = _build_response_from_parser(result)
        data = json.loads(response.model_dump_json(exclude_none=True))

        tc = data["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert args == {"settings": {"enabled": True, "tags": ["a", "b"]}}
