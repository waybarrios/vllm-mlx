# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool calling format:
    <|tool_call>call:function_name{param:<|"|>value<|"|>,param2:42}<tool_call|>

The <|"|> tokens are Gemma 4's escaped string delimiters within tool calls.
Parameters use a custom key:value format (not JSON).
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


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _parse_gemma4_params(param_str: str) -> dict:
    """Parse Gemma 4's key:value parameter format into a dict.

    Handles:
        city:<|"|>San Francisco<|"|>
        count:42
        flag:true
        nested:{a:<|"|>b<|"|>}
    """
    result = {}
    if not param_str or not param_str.strip():
        return result

    # Replace escaped quotes with placeholder for parsing
    PLACEHOLDER = "\x00QUOTE\x00"
    s = param_str.replace('<|"|>', PLACEHOLDER)

    # Split on top-level commas (not inside braces or strings)
    depth = 0
    in_string = False
    current = []
    pairs = []
    for ch in s:
        if ch == PLACEHOLDER[0] and s[len(current):].startswith(PLACEHOLDER):
            in_string = not in_string
            current.append(ch)
        elif ch == '{' and not in_string:
            depth += 1
            current.append(ch)
        elif ch == '}' and not in_string:
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0 and not in_string:
            pairs.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        pairs.append(''.join(current).strip())

    for pair in pairs:
        if ':' not in pair:
            continue
        key, _, val = pair.partition(':')
        key = key.strip()
        val = val.strip()

        # Restore escaped quotes and extract string value
        val = val.replace(PLACEHOLDER, '')
        if not val:
            val = val  # empty string
        # Try numeric/bool conversion
        elif val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False
        elif val.lower() == 'null':
            val = None
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # keep as string

        result[key] = val
    return result


# Pattern: <|tool_call>call:function_name{params}<tool_call|>
GEMMA4_TOOL_PATTERN = re.compile(
    r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)

# Streaming: detect start of tool call
GEMMA4_TOOL_START = "<|tool_call>"


@ToolParserManager.register_module(["gemma4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Handles the native format:
        <|tool_call>call:function_name{key:<|"|>value<|"|>}<tool_call|>
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        tool_calls = []
        cleaned_text = model_output

        for match in GEMMA4_TOOL_PATTERN.finditer(model_output):
            func_name = match.group(1)
            param_str = match.group(2)
            arguments = _parse_gemma4_params(param_str)

            tool_calls.append({
                "id": _generate_tool_id(),
                "name": func_name,
                "arguments": json.dumps(arguments),
            })
            cleaned_text = cleaned_text.replace(match.group(0), "").strip()

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls if tool_calls else None,
            content=cleaned_text if cleaned_text else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        # Check if we have a complete tool call in current_text
        if "<tool_call|>" in current_text:
            return self.extract_tool_calls(current_text, request)

        # If we see the start marker, signal that a tool call is in progress
        if GEMMA4_TOOL_START in current_text:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=None, content=None
            )

        # No tool call markers — pass through as content
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=None, content=delta
        )
