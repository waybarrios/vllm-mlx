# SPDX-License-Identifier: Apache-2.0
"""
Mistral tool call parser for vllm-mlx.

Handles Mistral's tool calling format:
- Format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
- Or newer: [TOOL_CALLS]func_name{"arg": "value"}
- Or newest (Ministral 3, Devstral Small 2, Dec 2025 tokenizers):
  [TOOL_CALLS]func_name[ARGS]{"arg": "value"}
  Confirmed directly in these models' chat_template.jinja:
  {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}

Used with models like Mistral-7B-Instruct, Devstral, Ministral 3, etc.
"""

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

ALPHANUMERIC = ascii_letters + digits


def generate_mistral_tool_id() -> str:
    """
    Generate a random Mistral-compatible tool call ID.

    Mistral Tool Call IDs must be alphanumeric with a length of 9.
    """
    return "".join(choices(ALPHANUMERIC, k=9))


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models.

    Supports both old and new Mistral tool call formats:
    - Old (< v11): [TOOL_CALLS] [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    - New (>= v11): [TOOL_CALLS]add{"a": 1, "b": 2}

    Used when --enable-auto-tool-choice --tool-call-parser mistral are set.
    """

    # Mistral chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    BOT_TOKEN = "[TOOL_CALLS]"
    ARGS_TOKEN = "[ARGS]"
    TOOL_CALL_REGEX = re.compile(r"\[{.*}\]", re.DOTALL)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.bot_token_id = self.vocab.get(self.BOT_TOKEN) if self.vocab else None
        # Streaming state for the name/arguments boundary within the current
        # tool call. See _parse_streaming_tool_delta.
        self._args_started: bool = False
        self._name_buffer: str = ""

    def reset(self) -> None:
        super().reset()
        self._args_started = False
        self._name_buffer = ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Mistral model response.

        Args:
            model_output: The complete model output string
            request: Optional request context

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        # If the tool call token is not present, return as text response
        if self.BOT_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_and_raw_tool_calls = model_output.split(self.BOT_TOKEN)
        content = content_and_raw_tool_calls[0].strip()
        raw_tool_calls = content_and_raw_tool_calls[1:]

        tool_calls = []

        for raw_tool_call in raw_tool_calls:
            raw_tool_call = raw_tool_call.strip()
            if not raw_tool_call:
                continue

            # Try newest format first: func_name[ARGS]{"arg": "value"}
            if not raw_tool_call.startswith("[") and self.ARGS_TOKEN in raw_tool_call:
                tool_name, _, args_str = raw_tool_call.partition(self.ARGS_TOKEN)
                tool_name = tool_name.strip()

                if tool_name:
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    )
                continue

            # Try new format: func_name{"arg": "value"}
            if not raw_tool_call.startswith("[") and "{" in raw_tool_call:
                end_name = raw_tool_call.find("{")
                tool_name = raw_tool_call[:end_name].strip()
                args_str = raw_tool_call[end_name:]

                if tool_name:
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    )
                continue

            # Try old format: [{"name": "func", "arguments": {...}}]
            try:
                parsed = json.loads(raw_tool_call)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            tool_calls.append(
                                {
                                    "id": generate_mistral_tool_id(),
                                    "name": item["name"],
                                    "arguments": (
                                        json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict)
                                        else str(args)
                                    ),
                                }
                            )
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: try regex to extract JSON array
            try:
                match = self.TOOL_CALL_REGEX.search(raw_tool_call)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_mistral_tool_id(),
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
            except (json.JSONDecodeError, AttributeError):
                # If all parsing fails, treat as content
                if raw_tool_call:
                    content = (
                        (content + " " + raw_tool_call).strip()
                        if content
                        else raw_tool_call
                    )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
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
        Extract tool calls from streaming Mistral model output.

        For streaming, we detect when [TOOL_CALLS] appears and start
        accumulating tool call data.
        """
        # Check if tool call token is in current output
        if self.BOT_TOKEN not in current_text:
            # Not a tool call yet, return content delta
            return {"content": delta_text}

        # Tool call detected
        if self.BOT_TOKEN in delta_text:
            # This delta contains the start of tool calls
            parts = delta_text.split(self.BOT_TOKEN)
            content_part = parts[0]
            tool_part = self.BOT_TOKEN.join(parts[1:])

            result: dict[str, Any] = {}
            if content_part:
                result["content"] = content_part

            # Start tracking tool call
            self.current_tool_id += 1
            self._args_started = False
            self._name_buffer = ""

            if tool_part:
                # Try to parse the tool part
                tool_delta = self._parse_streaming_tool_delta(tool_part)
                if tool_delta:
                    result["tool_calls"] = [
                        {
                            "index": self.current_tool_id,
                            "id": generate_mistral_tool_id(),
                            "type": "function",
                            "function": tool_delta,
                        }
                    ]

            return result if result else None

        # We're in the middle of a tool call
        if self.current_tool_id >= 0:
            tool_delta = self._parse_streaming_tool_delta(delta_text)
            if tool_delta:
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_id,
                            "type": "function",
                            "function": tool_delta,
                        }
                    ]
                }

        return None

    def _parse_streaming_tool_delta(self, text: str) -> dict[str, str] | None:
        """Parse a streaming delta for tool call information.

        Once the name/arguments boundary (the `[ARGS]` marker, or a bare `{`
        for older checkpoints) has been seen for the current tool call, every
        subsequent delta is argument text and is never re-classified — JSON
        string content (bare keys/values like `city` or `Paris`) has no
        distinguishing leading punctuation, so re-evaluating each delta in
        isolation (the previous approach) misclassified mid-argument
        fragments as more of the function name.
        """
        if not text:
            return None

        if self._args_started:
            return {"arguments": text}

        # Buffer until we can find the boundary marker — it may itself be
        # split across two deltas (e.g. "...[AR" / "GS]...").
        self._name_buffer += text
        for marker in (self.ARGS_TOKEN, "{"):
            idx = self._name_buffer.find(marker)
            if idx == -1:
                continue
            name = self._name_buffer[:idx].strip()
            args_start = idx + len(marker) if marker == self.ARGS_TOKEN else idx
            args = self._name_buffer[args_start:]
            self._args_started = True
            result: dict[str, str] = {}
            if name:
                result["name"] = name
            if args:
                result["arguments"] = args
            return result if result else None

        # Marker not seen yet — withhold rather than guess.
        return None
