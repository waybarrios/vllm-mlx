# SPDX-License-Identifier: Apache-2.0
"""
ATEM tool call parser for Muse Glimmer models.

A call is a channel addressed to the tool:

    <|start|>assistant to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Paris</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

Live generation is constrained to one call per turn, but the chat template
also renders whatever ``tool_calls`` a replayed history entry carries as
separate channels joined by ``<|eom|>``, so multiple channels are handled
defensively either way.

Pair with the ``muse_glimmer`` reasoning parser, which removes the reasoning
channel and the first channel header. Parameter values are JSON when they parse
as JSON and plain strings otherwise, matching mlx-vlm's ATEM parser.
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

_TOOL_CALLS_START = "<atem:function_calls>"
_TOOL_CALLS_END = "</atem:function_calls>"
_FRAMING_RE = re.compile(r"<\|start\|>|<\|message\|>")
# A block takes its channel header and the following <|eom|> separator with
# it, so nothing is left for /v1/responses' second reasoning pass to misread
# (``to=user_lookup`` as the user channel plus ``_lookup``) even when a
# replayed history entry renders more than one channel back to back.
_BLOCK_RE = re.compile(
    r"(?:(?:assistant )?to=[\w.\-]+\s*)?"
    r"<atem:function_calls>.*?</atem:function_calls>(?:\s*<\|eom\|>)?",
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>'
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _held_len(text: str) -> int:
    """Length of the trailing run of ``text`` that could still open a block."""
    return next(
        (
            n
            for n in range(len(_TOOL_CALLS_START) - 1, 0, -1)
            if text.endswith(_TOOL_CALLS_START[:n])
        ),
        0,
    )


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


@ToolParserManager.register_module(["muse_glimmer"])
class MuseGlimmerToolParser(ToolParser):
    """Tool call parser for Muse Glimmer's ATEM function calls."""

    STREAMING_MARKERS = (_TOOL_CALLS_START,)
    # <atem:function_calls> spans several tokens, so the routing layer cannot
    # wait for the complete marker before handing deltas to this parser.
    REQUIRES_EAGER_STREAMING = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Whether the first block has opened; reset per request via reset().
        self._in_calls = False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        text = _FRAMING_RE.sub("", model_output)
        tool_calls = [
            {
                "id": _generate_tool_id(),
                "name": invoke.group("name"),
                "arguments": json.dumps(
                    {
                        param.group("name"): _parse_value(param.group("value"))
                        for param in _PARAMETER_RE.finditer(invoke.group("body"))
                    },
                    ensure_ascii=False,
                ),
            }
            for invoke in _INVOKE_RE.finditer(text)
        ]
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        content = _BLOCK_RE.sub("", text).strip()
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content or None
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
        Stream text before the first ``<atem:function_calls>`` as content,
        withholding a trailing partial marker, then buffer each block and emit
        its calls once it closes.
        """
        # Search only the new text plus enough overlap to catch a marker that
        # straddles the previous delta, so each delta costs O(delta).
        result: dict[str, Any] = {}
        if not self._in_calls:
            sent = len(previous_text) - _held_len(previous_text)
            overlap = max(0, len(previous_text) - len(_TOOL_CALLS_START) + 1)
            start = current_text.find(_TOOL_CALLS_START, overlap)
            if start == -1:
                end = len(current_text) - _held_len(current_text)
                return {"content": current_text[sent:end]} if end > sent else None
            self._in_calls = True
            if start > sent:
                result["content"] = current_text[sent:start]
        overlap = max(0, len(previous_text) - len(_TOOL_CALLS_END) + 1)
        if current_text.find(_TOOL_CALLS_END, overlap) != -1:
            tool_calls = self.extract_tool_calls(current_text, request).tool_calls
            emitted = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(tool_calls)
                if i > self.current_tool_id
            ]
            if emitted:
                self.current_tool_id = len(tool_calls) - 1
                result["tool_calls"] = emitted
        return result or None

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._in_calls = False

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Release a partial marker withheld when generation ended on it."""
        held = 0 if _TOOL_CALLS_START in current_text else _held_len(current_text)
        return {"content": current_text[-held:]} if held else None
