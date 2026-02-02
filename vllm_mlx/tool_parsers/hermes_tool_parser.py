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

    Used when --enable-auto-tool-choice --tool-call-parser hermes are set.
    """

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    REASONING_PATTERN = re.compile(
        r"<tool_call_reasoning>(.*?)</tool_call_reasoning>", re.DOTALL
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Hermes model response.
        """
        tool_calls = []
        cleaned_text = model_output

        # Remove reasoning tags first (keep for content)
        reasoning_matches = self.REASONING_PATTERN.findall(model_output)
        cleaned_text = self.REASONING_PATTERN.sub("", cleaned_text)

        # Parse tool calls
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
        """
        Extract tool calls from streaming Hermes model output.
        """
        if "<tool_call>" not in current_text:
            return {"content": delta_text}

        if "</tool_call>" in delta_text:
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
