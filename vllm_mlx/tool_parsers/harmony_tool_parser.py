# SPDX-License-Identifier: Apache-2.0
"""
Harmony tool call parser for GPT-OSS models.

Harmony uses control tokens and channels for tool calling:

    <|channel|>commentary to=functions.get_weather
    <|constrain|>json
    <|message|>{"location": "San Francisco"}
    <|call|>

The final response is in the 'final' channel:

    <|channel|>final
    <|message|>The weather is 72F.
    <|return|>
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
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Terminator captured as named group: \Z (end-of-string) and a following
# <|channel|> are treated more strictly than explicit terminators — truncated
# mid-block output, or args that would span into a later channel, must not
# become a structured tool call with malformed/glued args.
_COMMENTARY_BLOCK_PATTERN = re.compile(
    r"<\|channel\|>commentary\s+to=functions\.(\w+)"
    r"(?:\s*<\|constrain\|>\w+)?"
    r"\s*<\|message\|>((?:(?!<\|channel\|>).)*?)"
    r"(?P<terminator><\|call\|>|<\|end\|>|<\|return\|>|<\|start\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)

# Pattern: <|channel|>final ... <|return|>
_FINAL_BLOCK_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


@ToolParserManager.register_module(["harmony", "gpt-oss"])
class HarmonyToolParser(ToolParser):
    """
    Tool call parser for GPT-OSS models using Harmony format.

    Harmony uses control tokens and 3 channels:
    - analysis: internal reasoning (handled by reasoning parser)
    - commentary: tool calls addressed with to=functions.{name}
    - final: user-facing response

    Used when --enable-auto-tool-choice --tool-call-parser harmony are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Harmony model response.

        Parses commentary channel blocks for tool calls and the final
        channel for the user-facing content.
        """
        tool_calls = []

        # Extract tool calls from commentary channel blocks
        for match in _COMMENTARY_BLOCK_PATTERN.finditer(model_output):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            terminator = match.group("terminator")
            # End-of-string and a following channel boundary mean the args
            # did not stop at an explicit call terminator: truncated or moved
            # on to another channel, never a call.
            strict = terminator in ("", "<|channel|>")

            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                if strict:
                    # No explicit terminator + invalid JSON: treat as
                    # truncated, not a tool call.
                    continue
                # Explicit terminator + invalid JSON: keep raw-args fallback.
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": args_str,
                    }
                )

        # Extract final channel content
        final_match = _FINAL_BLOCK_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        # No tool calls: return all text as content
        # If there's a final channel, use that; otherwise return the raw output
        # stripped of control tokens
        if content is None:
            content = _strip_control_tokens(model_output)

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content,
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
        Extract tool calls from streaming Harmony model output.

        A commentary block completes when an explicit terminator arrives
        (<|call|>, <|end|>, <|return|>, <|start|>) or when the model moves on
        to the <|channel|>final block; the completed call is emitted once
        (deduplicated by name + arguments). Final-channel content is emitted
        as regular content deltas and plain text passes through unchanged.
        """
        if not hasattr(self, "_emitted_streaming_signatures"):
            self._emitted_streaming_signatures = set()

        # No harmony channel at all: plain text passes through
        if "<|channel|>" not in current_text:
            return {"content": delta_text}

        # A commentary block completed: an explicit terminator arrived in this
        # delta, or the model moved on to the final channel.
        block_completed = any(
            tok in delta_text
            for tok in ("<|call|>", "<|end|>", "<|return|>", "<|start|>")
        ) or (
            "<|channel|>final" in current_text
            and "<|channel|>final" not in previous_text
        )

        if block_completed:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                emitted = []
                for i, tc in enumerate(result.tool_calls):
                    signature = (tc["name"], tc["arguments"])
                    if signature not in self._emitted_streaming_signatures:
                        self._emitted_streaming_signatures.add(signature)
                        emitted.append(
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                        )
                if emitted:
                    return {"tool_calls": emitted}

        # In the final channel, emit content
        if "<|channel|>final" in current_text and "<|call|>" not in current_text:
            # Only emit content after <|message|> in the final channel
            if "<|message|>" in current_text:
                final_start = current_text.rfind("<|channel|>final")
                msg_start = current_text.find("<|message|>", final_start)
                if msg_start >= 0:
                    msg_content = current_text[msg_start + len("<|message|>") :]
                    # Strip trailing control tokens
                    msg_content = msg_content.replace("<|return|>", "").strip()
                    if msg_content and not _is_control_token(delta_text):
                        return {"content": delta_text}

        # Building tool call or in analysis channel, suppress output
        return None

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._emitted_streaming_signatures = set()


def _strip_control_tokens(text: str) -> str:
    """Remove Harmony control tokens from text."""
    tokens = [
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    ]
    result = text
    for token in tokens:
        result = result.replace(token, "")
    # Clean up channel names and constrain values
    result = re.sub(r"(?:analysis|commentary|final)\s*", "", result)
    result = re.sub(r"to=functions\.\w+\s*", "", result)
    result = re.sub(r"json\s*", "", result)
    return result.strip()


def _is_control_token(text: str) -> bool:
    """Check if text is a Harmony control token."""
    return text.strip() in {
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    }
