# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vllm-mlx.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
- Function style: <function=name><parameter=key>value</parameter></function>
"""

import ast
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


def _parse_param_value(val: str) -> Any:
    """Parse a parameter value, handling JSON literals and plain strings."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        python_val = ast.literal_eval(val)
        if isinstance(python_val, set):
            python_val = sorted(python_val, key=str)
        if isinstance(python_val, (complex, bytes)):
            return val
        json.dumps(python_val)
        return python_val
    except (ValueError, SyntaxError, TypeError):
        return val


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["qwen", "qwen3"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]
    - Function: <function=name><parameter=key>value</parameter></function>

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    # Pattern for function-style: <function=name>...</function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    # Pattern for parameter extraction: <parameter=key>value</parameter>
    PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

    # Pattern for empty <tool_call> wrappers left after function extraction
    EMPTY_TOOL_CALL = re.compile(r"<tool_call>\s*</tool_call>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
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
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        # Try function-style: <function=name><parameter=key>value</parameter></function>
        # Qwen3.5 generates this format natively.
        if not tool_calls:
            func_matches = self.FUNCTION_PATTERN.findall(cleaned_text)
            for name, params_block in func_matches:
                # Try JSON arguments first (e.g. <function=name>{"key": "val"}</function>)
                params_block_stripped = params_block.strip()
                if params_block_stripped.startswith("{"):
                    try:
                        arguments = json.loads(params_block_stripped)
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name.strip(),
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            }
                        )
                        continue
                    except json.JSONDecodeError:
                        pass
                # Parse <parameter=key>value</parameter> tags
                params = self.PARAM_PATTERN.findall(params_block)
                arguments = {}
                for p_name, p_value in params:
                    arguments[p_name.strip()] = _parse_param_value(p_value.strip())
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            if func_matches:
                cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            # Clean up empty <tool_call> wrappers left after function extraction
            cleaned_text = self.EMPTY_TOOL_CALL.sub("", cleaned_text).strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # Partial marker prefixes — when current_text ends with one of these,
    # we suppress output until the next token confirms or denies a tool call.
    # These are long enough to avoid false positives on normal text.
    _PARTIAL_MARKERS = ("<function", "[Calling tool", "<tool_call")

    def _has_partial_marker(self, text: str) -> bool:
        """Check if text ends with an incomplete tool call marker prefix."""
        return self._get_partial_marker_len(text) > 0

    def _get_partial_marker_len(self, text: str) -> int:
        """Return the length of a partial tool call marker suffix at end of text."""
        tail = text[-20:]
        best = 0
        for marker in self._PARTIAL_MARKERS:
            for length in range(len(marker), 0, -1):
                if tail.endswith(marker[:length]) and length > best:
                    best = length
                    break
        return best

    def _was_buffering(self, previous_text: str) -> bool:
        """Check if the previous call was buffering a partial marker."""
        return self._has_partial_marker(previous_text)

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
        """
        Extract tool calls from streaming Qwen model output.
        """
        # Check for complete tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text
            or "[Calling tool:" in current_text
            or "<function=" in current_text
        )

        if not has_tool_marker:
            # Buffer partial markers (e.g. "<function" before "=" arrives).
            # Only the marker suffix is buffered; content before it in the
            # same delta is emitted immediately so no text is lost.
            if self._has_partial_marker(current_text):
                marker_len = self._get_partial_marker_len(current_text)
                marker_start = len(current_text) - marker_len
                safe_chars = marker_start - len(previous_text)
                if safe_chars > 0:
                    return {"content": delta_text[:safe_chars]}
                return None
            # If we were buffering before but the marker didn't complete,
            # emit the buffered marker prefix together with the new delta.
            if self._was_buffering(previous_text):
                for marker in self._PARTIAL_MARKERS:
                    for length in range(len(marker), 0, -1):
                        prefix = marker[:length]
                        if previous_text.endswith(prefix):
                            return {"content": prefix + delta_text}
                return {"content": delta_text}
            return {"content": delta_text}

        # Handle <function=name>...</function> (Qwen3.5 native format)
        if "<function=" in current_text:
            func_close_count = current_text.count("</function>")
            prev_func_close = previous_text.count("</function>")

            if current_text.count("<function=") > func_close_count:
                # Inside an incomplete function block, suppress output
                return None

            if func_close_count > prev_func_close:
                # New function block(s) completed
                result = self.extract_tool_calls(current_text)
                if result.tools_called:
                    new_calls = result.tool_calls[prev_func_close:]
                    if new_calls:
                        return {
                            "tool_calls": [
                                {
                                    "index": prev_func_close + i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for i, tc in enumerate(new_calls)
                            ]
                        }

            return None

        # If we're in a tool call, accumulate and parse at the end.
        # Check current_text (accumulated), not delta_text — closing markers
        # like ")]" or "</tool_call>" often span token boundaries and may
        # never appear within a single delta chunk.
        if "</tool_call>" in current_text or ")]" in current_text:
            # Tool call complete, parse the whole thing
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

        return None
