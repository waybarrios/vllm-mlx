# SPDX-License-Identifier: Apache-2.0
"""
Tool call parser for DeepSeek-V4 (Pro/Flash) DSML markup.

DeepSeek-V4 does not emit JSON function calls like V3/R1 — it uses its own
markup language, DSML::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Prague</｜DSML｜parameter>
    <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

The ``string`` attribute carries the type: ``"true"`` means the value is a raw
string, ``"false"`` means it is JSON (number, bool, array or object). That
distinction is why this cannot be a regex over ``name=value`` pairs — a string
parameter may legitimately contain ``"``, ``<`` or a JSON-looking payload.

``DeepSeekToolParser`` in ``deepseek_tool_parser.py`` handles the V3/R1 format
(``<｜tool▁calls▁begin｜>`` plus fenced JSON) and shares nothing with this one.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..utils.deepseek_v4_encoding import (
    DSML_TOKEN,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
    partial_marker_len,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

INVOKE_START = f"<{DSML_TOKEN}invoke"
INVOKE_END = f"</{DSML_TOKEN}invoke>"
PARAM_START = f"<{DSML_TOKEN}parameter"
PARAM_END = f"</{DSML_TOKEN}parameter>"

THINKING_END = "</think>"

# Attribute headers are bounded and well-formed; only the *values* are
# free-form, so a regex is safe here and a scanner is used for the rest.
_INVOKE_HEADER_RE = re.compile(r'\s*name="(?P<name>[^"]*)"\s*>')
_PARAM_HEADER_RE = re.compile(
    r'\s*name="(?P<name>[^"]*)"\s+string="(?P<is_str>true|false)"\s*>'
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["deepseek_v4", "dsml"])
class DeepSeekV4ToolParser(ToolParser):
    """Parse DeepSeek-V4 DSML tool calls.

    Example::

        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="search">
        <｜DSML｜parameter name="q" string="true">mlx</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>

    Malformed markup is never fatal: whatever parses becomes a tool call and the
    remainder is returned as content, because a half-generated call must not
    take the server down.
    """

    # The encoder consumes role="tool" messages and assistant tool_calls
    # directly — folding results into <tool_result> blocks and rendering calls
    # back as DSML. Declaring this False would make the server flatten them to
    # "[Tool Result (id)]: ..." and "[Calling tool: name(...)]" first
    # (api/utils.py), so the model would see a shape it was never trained on
    # and the encoder's own handling would never run.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Trailing text withheld because it might be the start of the opening
        # marker; reset per request via reset().
        self._pending: str = ""
        # How much of the accumulated text has already been streamed as
        # content, so the text preceding a tool block is emitted exactly once.
        self._emitted_len: int = 0

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract DSML tool calls from a complete response."""
        start = model_output.find(TOOL_CALLS_START)
        if start == -1:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self.strip_think_tags(model_output) or None,
            )

        content = self._content_before(model_output, start)
        block_start = start + len(TOOL_CALLS_START)
        end = model_output.find(TOOL_CALLS_END, block_start)
        block = model_output[block_start : end if end != -1 else len(model_output)]

        tool_calls = self._parse_invokes(block)
        if not tool_calls:
            # Marker present but nothing parsed — surface the raw text rather
            # than silently dropping the model's output.
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self.strip_think_tags(model_output) or None,
            )

        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

    def _content_before(self, text: str, tool_start: int) -> str | None:
        """Return the assistant content that precedes the tool call block.

        A tool call inside a reasoning block implicitly ends it, so anything up
        to and including ``</think>`` belongs to the reasoning parser, not here.
        """
        head = text[:tool_start]
        think_end = head.rfind(THINKING_END)
        if think_end != -1:
            head = head[think_end + len(THINKING_END) :]
        return head.strip() or None

    def _parse_invokes(self, block: str) -> list[dict[str, Any]]:
        """Parse every ``invoke`` element inside a tool_calls block."""
        tool_calls: list[dict[str, Any]] = []
        pos = 0

        while True:
            inv = block.find(INVOKE_START, pos)
            if inv == -1:
                break

            header = _INVOKE_HEADER_RE.match(block, inv + len(INVOKE_START))
            if header is None:
                # Unparseable header: skip this marker and keep looking.
                pos = inv + len(INVOKE_START)
                continue

            name = header.group("name")
            body_start = header.end()
            body_end = block.find(INVOKE_END, body_start)
            body = block[body_start : body_end if body_end != -1 else len(block)]

            arguments = self._parse_parameters(body)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "type": "function",
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

            if body_end == -1:
                break
            pos = body_end + len(INVOKE_END)

        return tool_calls

    def _parse_parameters(self, body: str) -> dict[str, Any]:
        """Parse ``parameter`` elements into a plain argument dict.

        ``string="true"`` keeps the value verbatim; ``string="false"`` is JSON
        and gets decoded, falling back to the raw string if the model emitted
        something that does not parse.
        """
        arguments: dict[str, Any] = {}
        pos = 0

        while True:
            par = body.find(PARAM_START, pos)
            if par == -1:
                break

            header = _PARAM_HEADER_RE.match(body, par + len(PARAM_START))
            if header is None:
                pos = par + len(PARAM_START)
                continue

            value_start = header.end()
            value_end = body.find(PARAM_END, value_start)
            if value_end == -1:
                # Truncated parameter — drop it rather than guess its value.
                break

            raw = body[value_start:value_end]
            if header.group("is_str") == "true":
                value: Any = raw
            else:
                try:
                    value = json.loads(raw)
                except (ValueError, json.JSONDecodeError):
                    value = raw

            arguments[header.group("name")] = value
            pos = value_end + len(PARAM_END)

        return arguments

    def _format_streaming(self, result: ExtractedToolCallInformation) -> dict[str, Any]:
        """Render extracted tool calls into the streaming delta shape."""
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

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._pending = ""
        self._emitted_len = 0

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
        """Stream DSML output.

        DSML exposes no usable partial state — a parameter's type is only known
        once its closing tag arrives — so once the block opens everything is
        buffered and the calls are emitted in one delta when the block closes.

        Before that, content passes through, except for a tail that could still
        grow into the opening marker. That tail is held in ``_pending`` and
        released as soon as the next delta proves it was ordinary text.
        """
        block_start = current_text.find(TOOL_CALLS_START)
        if block_start != -1:
            closed = (
                TOOL_CALLS_END in current_text and TOOL_CALLS_END not in previous_text
            )
            # Flush any text preceding the block that has not been streamed
            # yet. When the block spans several deltas this goes out on its
            # own, ahead of the calls. When the whole response arrives as one
            # delta there is no later delta to flush into, so the text rides
            # along with the calls rather than being dropped — losing it made
            # the response depend on how the output happened to be chunked.
            head = ""
            if self._emitted_len < block_start:
                head = current_text[self._emitted_len : block_start]
                self._emitted_len = block_start
                self._pending = ""
            if head and not closed:
                return {"content": head}

            # The closing marker frequently straddles delta boundaries, so
            # completion is detected against the accumulated text, not the
            # delta. Comparing with previous_text makes it fire exactly once.
            if closed:
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    self._pending = ""
                    self._emitted_len = len(current_text)
                    formatted = self._format_streaming(result)
                    if head:
                        formatted = {**formatted, "content": head}
                    return formatted
            return None

        text = self._pending + delta_text
        hold = partial_marker_len(text)
        self._pending = text[len(text) - hold :] if hold else ""
        emit = text[: len(text) - hold] if hold else text
        self._emitted_len += len(emit)
        return {"content": emit} if emit else None
