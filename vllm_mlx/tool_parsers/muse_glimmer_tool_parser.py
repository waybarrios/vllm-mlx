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
channel and the first channel header. Parameters declared ``"string"`` in the
request's tool schema are kept verbatim (whitespace included); undeclared
parameters are JSON when they parse as JSON and plain strings otherwise,
matching mlx-vlm's ATEM parser. A value that would serialize to a non-finite
float -- ``NaN``/``Infinity`` literals, or an overflow such as ``1e400`` --
keeps its raw text as a string instead, so ``arguments`` is always strict JSON
(a nested container with a non-finite leaf is kept whole, as its raw text).

Only an ``<atem:invoke>`` inside a recognized ``<atem:function_calls>`` block
is a call -- a bare tag elsewhere in the turn (an XML example in the reply) is
prose, matching streaming, which only ever waits for the block wrapper.

Visible content is always the reply text only: everything from the first
``<atem:function_calls>`` marker on belongs to tool channels, and an
unfinished block (max_tokens cut inside it) is suppressed entirely -- never
leaked as content, in streaming and non-streaming alike.
"""

import json
import math
import re
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

_TOOL_CALLS_START = "<atem:function_calls>"
_TOOL_CALLS_END = "</atem:function_calls>"
_EOM = "<|eom|>"
_FRAMING_RE = re.compile(r"<\|start\|>|<\|message\|>")
# Local duplicate of the reasoning module's bare-header pattern (tool parsers
# must not import from ``vllm_mlx.reasoning``): a ``to=<name>`` channel header
# that is NOT followed by a tool-call marker, i.e. the reasoning channel.
_SELF_HEADER_RE = re.compile(r"to=self(?![\w.\-]*\s*<atem:function_calls>)")
_CHANNEL_HEADER_RE = re.compile(r"(?:assistant\s+)?to=[\w.\-]+")
_INVOKE_RE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>'
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _drop_reasoning_channel(text: str) -> str:
    """Text left after a leading ``to=self`` reasoning channel, else unchanged.

    The tool parser runs before the reasoning parser in ``/v1/responses``
    (``_run_responses_request``) and receives server-redirected reasoning in
    chat streaming, so an ``<atem:function_calls>`` *example* inside a
    ``to=self`` channel must never be scanned as a call. If the channel never
    closed (``<|eom|>`` absent), nothing after it can be a call either.
    """
    stripped = text.lstrip()
    if not _SELF_HEADER_RE.match(stripped):
        return text
    _, sep, rest = stripped.partition(_EOM)
    return rest if sep else ""


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


def _iter_call_regions(text: str) -> Iterator[str]:
    """Yield the body of each recognized call block, in order.

    A region runs from an ``<atem:function_calls>`` opening to its close, or to
    the end of ``text`` when the block was cut off (max_tokens) -- matching the
    unit streaming waits for. An ``<atem:invoke>`` outside every region is an
    XML example in prose, never a call.
    """
    pos = 0
    while (start := text.find(_TOOL_CALLS_START, pos)) != -1:
        body_start = start + len(_TOOL_CALLS_START)
        end = text.find(_TOOL_CALLS_END, body_start)
        if end == -1:
            yield text[body_start:]
            return
        yield text[body_start:end]
        pos = end + len(_TOOL_CALLS_END)


def _reject_constant(name: str) -> Any:
    """Refuse JSON's non-standard ``NaN``/``Infinity``/``-Infinity`` literals."""
    raise ValueError(f"non-finite JSON constant: {name}")


def _has_nonfinite(value: Any) -> bool:
    """True when ``value`` holds a non-finite float, at any depth."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return False


def _parse_value(value: str, *, is_string: bool = False) -> Any:
    if is_string:
        # Declared string parameter: keep raw, whitespace included.
        return value
    try:
        # ``parse_constant`` rejects bare NaN/Infinity; the nonfinite sweep
        # also catches overflow (``1e400`` -> inf). Either way the raw text is
        # kept as a string so the emitted arguments stay strict JSON.
        parsed = json.loads(value, parse_constant=_reject_constant)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    return value if _has_nonfinite(parsed) else parsed


@ToolParserManager.register_module(["muse_glimmer"])
class MuseGlimmerToolParser(ToolParser):
    """Tool call parser for Muse Glimmer's ATEM function calls."""

    STREAMING_MARKERS = (_TOOL_CALLS_START,)
    # <atem:function_calls> spans several tokens, so the routing layer cannot
    # wait for the complete marker before handing deltas to this parser.
    REQUIRES_EAGER_STREAMING = True
    # Every call is its own to=<tool> channel: an <atem:function_calls>
    # example inside a to=self reasoning channel is prose, never a call, so
    # the server must not redirect reasoning into the tool stream.
    TOOL_MARKUP_IN_REASONING = False
    # The chat template renders role="tool" messages and tool_calls channels
    # natively, so replayed history needs no text conversion.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    @staticmethod
    def _string_argument_names(
        request: dict[str, Any] | None, tool_name: str
    ) -> set[str]:
        if request is None:
            return set()
        for tool in request.get("tools", []):
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            )
            return {
                name
                for name, schema in properties.items()
                if isinstance(schema, dict) and schema.get("type") == "string"
            }
        return set()

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Whether the first block has opened; reset per request via reset().
        self._in_calls = False
        # Stashed by extract_tool_calls_streaming so finalize_streaming
        # (whose signature has no request) can keep schema-aware parsing.
        self._request: dict[str, Any] | None = None

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        text = _drop_reasoning_channel(_FRAMING_RE.sub("", model_output))
        marker = text.find(_TOOL_CALLS_START)
        if marker == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=text.strip() or None
            )
        # Scan only inside recognized ``<atem:function_calls>`` regions, the
        # same unit streaming waits for: a bare ``<atem:invoke>`` elsewhere in
        # the turn (e.g. an XML example in the reply) is prose, not a call.
        tool_calls = []
        for region in _iter_call_regions(text):
            for invoke in _INVOKE_RE.finditer(region):
                # Schema lookup once per call, not once per parameter.
                string_args = self._string_argument_names(request, invoke.group("name"))
                arguments = {
                    param.group("name"): _parse_value(
                        param.group("value"),
                        is_string=param.group("name") in string_args,
                    )
                    for param in _PARAMETER_RE.finditer(invoke.group("body"))
                }
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": invoke.group("name"),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
        if tool_calls:
            content = text[:marker].strip()
            if not content or _CHANNEL_HEADER_RE.fullmatch(content) or _EOM in content:
                content = ""
        else:
            # Unfinished block (generation cut inside it): suppress entirely,
            # matching streaming, which never releases partial markers.
            content = ""
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content or None,
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
        self._request = request
        result: dict[str, Any] = {}
        if not self._in_calls:
            sent = len(previous_text) - _held_len(previous_text)
            overlap = max(0, len(previous_text) - len(_TOOL_CALLS_START) + 1)
            start = current_text.find(_TOOL_CALLS_START, overlap)
            if start == -1:
                end = len(current_text) - _held_len(current_text)
                piece = current_text[sent:end]
                # A marker appearing anywhere in this turn means everything
                # before it is channel/redirect text, not reply (the reply is
                # its own to=user channel); never stream it as content.
                if _TOOL_CALLS_START in current_text:
                    return None
                return {"content": piece} if piece else None
            self._in_calls = True
            if start > sent:
                pre = current_text[sent:start]
                # Emit pre-marker text only when it follows a reasoning-channel
                # close; redirected reasoning (or an earlier channel's header)
                # in front of a block is never reply text in this format.
                if _EOM in pre:
                    result["content"] = pre[pre.rindex(_EOM) + len(_EOM) :]
        overlap = max(0, len(previous_text) - len(_TOOL_CALLS_END) + 1)
        if current_text.find(_TOOL_CALLS_END, overlap) != -1:
            tool_calls = self.extract_tool_calls(current_text, request).tool_calls
            if emitted := self._new_streaming_tool_calls(tool_calls):
                result["tool_calls"] = emitted["tool_calls"]
        return result or None

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._in_calls = False
        self._request = None

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Resolve streaming state at end of generation.

        Two cases: release a partial marker withheld from content, and recover
        calls from a block that closed its ``<atem:invoke>`` but never its
        ``</atem:function_calls>`` wrapper (non-streaming extracts those; the
        close-triggered emit in ``extract_tool_calls_streaming`` never fired).
        """
        if _TOOL_CALLS_START in current_text:
            final = self.extract_tool_calls(current_text, self._request)
            return self._new_streaming_tool_calls(final.tool_calls)
        held = _held_len(current_text)
        return {"content": current_text[-held:]} if held else None
