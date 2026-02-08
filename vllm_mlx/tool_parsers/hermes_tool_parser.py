# SPDX-License-Identifier: Apache-2.0
"""
Hermes/Nous tool call parser for vllm-mlx.

Handles Hermes-style tool calling format used by NousResearch models.
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


@ToolParserManager.register_module(["hermes", "nous"])
class HermesToolParser(ToolParser):
    """
    Tool call parser for Hermes/Nous models.

    Supports Hermes tool call format:
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Sometimes with additional reasoning in <tool_call_reasoning>
    - Fallback: raw JSON {"name": "func", "arguments": {...}} (for models that omit tags)

    Used when --enable-auto-tool-choice --tool-call-parser hermes are set.
    """

    # Standard format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Lenient format: <tool_call or <tool_call> followed by JSON (handles malformed tags)
    TOOL_CALL_LENIENT_PATTERN = re.compile(
        r'<tool_call[^{]*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\})', re.DOTALL
    )
    REASONING_PATTERN = re.compile(
        r"<tool_call_reasoning>(.*?)</tool_call_reasoning>", re.DOTALL
    )
    # Fallback pattern for raw JSON tool calls (without tags)
    RAW_JSON_TOOL_PATTERN = re.compile(
        r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}', re.DOTALL
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Hermes model response.
        """
        tool_calls = []
        cleaned_text = model_output

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Remove reasoning tags first (keep for content)
        reasoning_matches = self.REASONING_PATTERN.findall(cleaned_text)
        cleaned_text = self.REASONING_PATTERN.sub("", cleaned_text)

        # Parse tool calls with <tool_call> tags (primary format)
        matches = self.TOOL_CALL_PATTERN.findall(cleaned_text)
        for match in matches:
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

        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # Fallback 1: try lenient pattern for malformed tags like <tool_call without >
        if not tool_calls:
            lenient_matches = self.TOOL_CALL_LENIENT_PATTERN.findall(cleaned_text)
            for match in lenient_matches[:1]:  # Only first to avoid hallucinations
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
                        cleaned_text = self.TOOL_CALL_LENIENT_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    continue

        # Fallback 2: try raw JSON format if no tagged tool calls found
        # Only parse the FIRST valid tool call to avoid hallucinated multiple calls
        if not tool_calls:
            raw_matches = self.RAW_JSON_TOOL_PATTERN.findall(cleaned_text)
            if raw_matches:
                # Only take the first match to avoid hallucinated tool calls
                name, args_str = raw_matches[0]
                try:
                    arguments = json.loads(args_str)
                    # Validate: only accept if tool name exists in request tools
                    valid_tool = True
                    if request and "tools" in request:
                        tool_names = [
                            t.get("function", {}).get("name", "")
                            for t in request.get("tools", [])
                            if isinstance(t, dict)
                        ]
                        valid_tool = name in tool_names

                    if valid_tool and name:
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            }
                        )
                        # Remove the matched tool call from text
                        cleaned_text = self.RAW_JSON_TOOL_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    pass

        # Include reasoning in content if present
        if reasoning_matches:
            reasoning_text = " ".join(reasoning_matches)
            if cleaned_text:
                cleaned_text = f"{cleaned_text}\n\n(Reasoning: {reasoning_text})"
            else:
                cleaned_text = f"(Reasoning: {reasoning_text})"

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
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
        """
        Extract tool calls from streaming Hermes model output.
        """
        # Check for tagged tool calls
        if "<tool_call>" in current_text:
            if "</tool_call>" in delta_text:
                result = self.extract_tool_calls(current_text, request)
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

        # Fallback: check for raw JSON tool calls (detect closing brace pattern)
        # Look for complete JSON object with "name" and "arguments"
        if '{"name":' in current_text and '"arguments":' in current_text:
            # Check if we have a complete JSON object (ends with }})
            if delta_text.rstrip().endswith("}"):
                result = self.extract_tool_calls(current_text, request)
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

        return {"content": delta_text}
