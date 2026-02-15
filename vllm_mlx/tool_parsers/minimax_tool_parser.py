# SPDX-License-Identifier: Apache-2.0
"""
MiniMax M2.1 tool call parser for vllm-mlx.

Handles MiniMax M2.1 style tool calling format with XML tags:
<minimax:tool_call>
<invoke name="func_name">
<parameter name="param">value</parameter>
</invoke>
</minimax:tool_call>

Based on the GLM-4.7 parser pattern.
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


@ToolParserManager.register_module(["minimax", "minimax_m2"])
class MiniMaxToolParser(ToolParser):
    """
    Tool call parser for MiniMax M2.1 models.

    Supports MiniMax tool call format:
    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </minimax:tool_call>

    Multiple invocations can appear in a single <minimax:tool_call> block.

    Used when --enable-auto-tool-choice --tool-call-parser minimax are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Match entire tool call block
    TOOL_CALL_PATTERN = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
    )

    # Match individual invoke blocks with function name
    INVOKE_PATTERN = re.compile(r'<invoke name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL)

    # Match parameter key-value pairs
    PARAM_PATTERN = re.compile(
        r'<parameter name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL
    )

    def _deserialize(self, value: str) -> Any:
        """Convert string value to appropriate Python type.

        Uses json.loads for type coercion, falls back to raw string.
        """
        value = value.strip()

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete MiniMax M2.1 model response.
        """
        tool_calls = []

        # Strip think tags using the base class method
        cleaned_text = self.strip_think_tags(model_output)

        # Get valid tool names for validation
        valid_names = self._get_tool_names(request)

        # Find all <minimax:tool_call> blocks
        tc_blocks = self.TOOL_CALL_PATTERN.findall(cleaned_text)

        for block in tc_blocks:
            # Find all <invoke> elements within the block
            invoke_matches = self.INVOKE_PATTERN.findall(block)

            for func_name, params_section in invoke_matches:
                func_name = func_name.strip()
                if not func_name:
                    continue

                # Validate tool name against available tools if provided
                if valid_names and func_name not in valid_names:
                    continue

                # Parse parameters
                arguments = {}
                param_matches = self.PARAM_PATTERN.findall(params_section)
                for param_name, param_value in param_matches:
                    key = param_name.strip()
                    value = self._deserialize(param_value)
                    if key:
                        arguments[key] = value

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        # When tool calls are found, don't return reasoning text as content
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
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
        Extract tool calls from streaming MiniMax M2.1 model output.
        """
        # Skip thinking content in streaming
        if "<think>" in current_text and "</think>" not in current_text:
            return None

        # Once <minimax:tool_call> is detected, buffer until it closes.
        if "<minimax:tool_call>" in current_text:
            if "</minimax:tool_call>" in delta_text:
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

        # No tool call detected yet; strip think tags and emit content
        clean_delta = self.strip_think_tags(delta_text)
        if clean_delta:
            return {"content": clean_delta}
        return None
