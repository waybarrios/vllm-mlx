# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's tool calling format:
- Tool call: <|tool_call>call:name{key:<|"|>value<|"|>,num:42}<tool_call|>

Gemma 4 uses a custom non-JSON argument format:
- Keys are unquoted
- Strings are delimited by <|"|> (a special token) instead of "
- Booleans are true/false, numbers are bare
- Nested objects use {}, arrays use []
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


# Match: <|tool_call>call:NAME{ARGS}<tool_call|>
# Capture group 1: function name
# Capture group 2: raw arguments (Gemma's custom format)
TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call>call:([\w-]+)\{(.*?)\}<tool_call\|>", re.DOTALL
)


def _gemma_args_to_json(raw: str) -> str:
    """
    Convert Gemma 4's custom argument format to valid JSON.

    Gemma format: key:<|"|>value<|"|>,other_key:42,flag:true
    JSON format:  {"key":"value","other_key":42,"flag":true}

    Uses a character-by-character state machine to correctly handle
    nested structures and string values containing special characters.
    """
    # Replace the escape token with a placeholder that won't conflict
    # with the state machine, then we'll produce proper JSON quotes
    QUOTE = "\x00"  # null byte as temporary placeholder
    text = raw.replace('<|"|>', QUOTE)

    result = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        if ch == QUOTE:
            # Start of a string value — copy until closing QUOTE
            result.append('"')
            i += 1
            while i < length and text[i] != QUOTE:
                c = text[i]
                if c == '"':
                    result.append('\\"')
                elif c == '\\':
                    result.append('\\\\')
                elif c == '\n':
                    result.append('\\n')
                elif c == '\t':
                    result.append('\\t')
                else:
                    result.append(c)
                i += 1
            result.append('"')
            i += 1  # skip closing QUOTE

        elif ch in '{}[],:':
            # Structural characters pass through
            result.append(ch)
            i += 1

        elif ch in ' \t\n\r':
            # Whitespace passes through
            result.append(ch)
            i += 1

        else:
            # Bare token: could be a key, number, boolean, or null
            start = i
            while i < length and text[i] not in (QUOTE + '{}[],:  \t\n\r'):
                i += 1
            token = text[start:i]

            # Look ahead: if next non-whitespace char is ':', this is a key
            j = i
            while j < length and text[j] in ' \t\n\r':
                j += 1

            if j < length and text[j] == ':':
                # It's a key — quote it
                result.append(f'"{token}"')
            else:
                # It's a value — keep bare for numbers/booleans/null
                result.append(token)

    return "".join(result)


@ToolParserManager.register_module(["gemma4", "gemma_4", "gemma4_27b"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma 4 models.

    Supports Gemma 4 tool call format:
    - <|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>

    Gemma 4's chat template uses <|tool> / <tool|> for definitions
    and <|tool_call> / <tool_call|> for calls, with <|"|> as string
    delimiters instead of standard JSON quotes.

    Used when --enable-auto-tool-choice --tool-call-parser gemma4 are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response."""
        # Strip think tags if present (Gemma 4 uses <|channel>...<channel|>
        # for thinking but may also emit standard <think> tags)
        text = self.strip_think_tags(model_output)

        tool_calls = []
        matches = TOOL_CALL_PATTERN.findall(text)

        for name, raw_args in matches:
            try:
                json_str = "{" + _gemma_args_to_json(raw_args) + "}"
                arguments = json.loads(json_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except (json.JSONDecodeError, ValueError):
                # Fall back to raw args string
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": raw_args,
                    }
                )

        # Remove tool call markup from content
        cleaned_text = text
        if matches:
            cleaned_text = TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
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
        """Extract tool calls from streaming Gemma 4 model output."""
        # No tool call started yet — pass through as content
        if "<|tool_call>" not in current_text:
            return {"content": delta_text}

        # Tool call end marker arrived — parse all tool calls
        if "<tool_call|>" in delta_text:
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

        # Inside a tool call — buffer (return None to suppress partial output)
        return None
