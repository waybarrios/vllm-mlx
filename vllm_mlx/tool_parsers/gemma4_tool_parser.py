# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's tool calling format:
- <|tool_call>call:function_name{param:<|"|>value<|"|>}<tool_call|>

Supports Gemma 4 26B-A4B and similar models.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Pattern to match Gemma 4 tool call format
# <|tool_call>call:function_name{key:<|"|>value<|"|>, ...}<tool_call|>
TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call>call:([\w.-]+)\{(.*?)\}<tool_call\|>',
    re.DOTALL
)

# Pattern to extract parameters: key:<|"|>value<|"|>
PARAM_PATTERN = re.compile(
    r'(\w+):<\|"\|>(.*?)<\|"\|>',
    re.DOTALL
)

# Also handle the case where quotes are rendered as regular quotes
PARAM_PATTERN_SIMPLE = re.compile(
    r'(\w+):\s*"(.*?)"',
    re.DOTALL
)


@ToolParserManager.register_module(["gemma4", "gemma-4"])
class Gemma4ToolParser(ToolParser):
    """Parser for Gemma 4's tool calling format."""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Gemma 4 output."""
        tool_calls = []

        # Also strip thinking/channel markers
        text = model_output

        matches = TOOL_CALL_PATTERN.findall(text)
        for func_name, params_str in matches:
            # Parse parameters
            args = {}
            for pattern in [PARAM_PATTERN, PARAM_PATTERN_SIMPLE]:
                param_matches = pattern.findall(params_str)
                for key, value in param_matches:
                    args[key] = value

            tool_calls.append({
                "id": generate_tool_id(),
                "name": func_name.strip(),
                "arguments": json.dumps(args, ensure_ascii=False),
            })

        if tool_calls:
            # Remove tool call markup from content
            cleaned = TOOL_CALL_PATTERN.sub("", text).strip()
            # Also remove thinking channel markers
            cleaned = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', cleaned, flags=re.DOTALL).strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned or None,
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming Gemma 4 output."""
        # Check for tool call markers
        if "<|tool_call>" not in current_text:
            return {"content": delta_text}

        # Check if tool call is complete (check current_text, not delta_text,
        # because closing tag may span multiple tokens)
        if "<tool_call|>" in current_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        # Inside tool call markup — suppress output
        return None
