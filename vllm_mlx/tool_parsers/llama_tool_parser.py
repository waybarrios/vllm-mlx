# SPDX-License-Identifier: Apache-2.0
"""
Llama tool call parser for vllm-mlx.

Handles Llama's tool calling formats:
- JSON with python tag: <|python_tag|>{"name": "fn", "parameters": {...}}
    (canonical Llama-3.1+ / Llama-4 format per Meta's model card)
- Bare JSON: {"type": "function", "name": "fn", "parameters": {...}}
    (Llama-3.3 format without a leading marker)
- XML style: <function=name>{"arg": "value"}</function>
    (older Llama 3.0 / Code Llama format; kept for backward compat)
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


@ToolParserManager.register_module(["llama", "llama3", "llama4"])
class LlamaToolParser(ToolParser):
    """
    Tool call parser for Llama models.

    Supports tagged JSON, bare JSON, and legacy XML tool calls.

    Used when --enable-auto-tool-choice --tool-call-parser llama are set.
    """

    # Llama 3+ chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Pattern for older Llama-style: <function=name>{"json"}</function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(\{.*?\})</function>", re.DOTALL)

    # Python-tag marker that Llama 3.1+ / Llama 4 emit immediately before a
    # JSON object of the form {"name": ..., "parameters": {...}}. This is the
    # canonical format per Meta's Llama-3.1 model card.
    PYTHON_TAG = "<|python_tag|>"

    # Bare Llama JSON is ambiguous until the first key is readable. Keep only
    # prefixes that can still become a canonical `name`- or `type`-first tool
    # envelope buffered; ordinary JSON is flushed back as content.
    BARE_JSON_MARKER = re.compile(r'^\s*\{\s*"(?:name|type)"\s*:')
    BARE_JSON_PARTIAL = re.compile(
        r'^\s*\{\s*"?(?:n(?:a(?:m(?:e)?)?)?|t(?:y(?:p(?:e)?)?)?)?"?\s*:?\s*$'
    )

    # The server uses these capabilities to keep Llama-only heuristics from
    # changing the behavior of other configured parsers such as ``auto``.
    STREAMING_MARKERS = (PYTHON_TAG,)
    SUPPORTS_BARE_JSON_STREAMING = True

    @staticmethod
    def _tool_call_from_object(obj: object) -> dict | None:
        """Convert a Llama JSON envelope into the internal call shape."""
        if not (
            isinstance(obj, dict)
            and "name" in obj
            and ("parameters" in obj or "arguments" in obj)
        ):
            return None
        arguments = obj.get("parameters", obj.get("arguments", {}))
        return {
            "id": generate_tool_id(),
            "name": str(obj["name"]).strip(),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    @staticmethod
    def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
        """Remove non-overlapping source spans without changing other bytes."""
        if not spans:
            return text
        parts: list[str] = []
        cursor = 0
        for start, end in sorted(spans):
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    @classmethod
    def _extract_python_tag_data(
        cls, text: str
    ) -> tuple[list[tuple[int, dict]], list[tuple[int, int]], int | None]:
        """Return ordered calls, consumed spans, and an incomplete suffix."""
        records: list[tuple[int, dict]] = []
        spans: list[tuple[int, int]] = []
        pending_start: int | None = None
        decoder = json.JSONDecoder()
        cursor = 0

        while True:
            marker_start = text.find(cls.PYTHON_TAG, cursor)
            if marker_start < 0:
                break
            scan = marker_start + len(cls.PYTHON_TAG)
            while scan < len(text) and text[scan].isspace():
                scan += 1

            try:
                obj, length = decoder.raw_decode(text[scan:])
            except json.JSONDecodeError:
                pending_start = marker_start
                break

            call = cls._tool_call_from_object(obj)
            if call is None:
                cursor = marker_start + len(cls.PYTHON_TAG)
                continue

            records.append((scan, call))
            consumed_end = scan + length

            while True:
                separator_start = consumed_end
                next_start = consumed_end
                while next_start < len(text) and text[next_start].isspace():
                    next_start += 1
                if next_start >= len(text) or text[next_start] != ";":
                    break
                next_start += 1
                while next_start < len(text) and text[next_start].isspace():
                    next_start += 1
                if next_start >= len(text):
                    pending_start = separator_start
                    break
                try:
                    next_obj, next_length = decoder.raw_decode(text[next_start:])
                except json.JSONDecodeError:
                    pending_start = separator_start
                    break
                next_call = cls._tool_call_from_object(next_obj)
                if next_call is None:
                    break
                records.append((next_start, next_call))
                consumed_end = next_start + next_length

            spans.append((marker_start, consumed_end))
            cursor = consumed_end
            if pending_start is not None:
                break

        return records, spans, pending_start

    @classmethod
    def _extract_bare_json_data(
        cls, text: str
    ) -> tuple[list[tuple[int, dict]], list[tuple[int, int]], int | None]:
        """Parse a top-level sequence of Llama call envelopes."""
        decoder = json.JSONDecoder()
        records: list[tuple[int, dict]] = []
        spans: list[tuple[int, int]] = []
        scan = len(text) - len(text.lstrip())
        if scan >= len(text) or text[scan] != "{":
            return records, spans, None

        while scan < len(text):
            object_start = scan
            try:
                obj, length = decoder.raw_decode(text[object_start:])
            except json.JSONDecodeError:
                pending = (
                    0 if cls._bare_json_prefix_is_viable(text) or records else None
                )
                return records, spans, pending

            call = cls._tool_call_from_object(obj)
            if call is None:
                return [], [], None

            object_end = object_start + length
            records.append((object_start, call))
            spans.append((object_start, object_end))
            scan = object_end
            while scan < len(text) and text[scan].isspace():
                scan += 1
            if scan >= len(text):
                return records, spans, None
            if text[scan] != ";":
                return records, spans, None

            separator_start = scan
            scan += 1
            while scan < len(text) and text[scan].isspace():
                scan += 1
            spans[-1] = (spans[-1][0], scan)
            if scan >= len(text):
                return records, spans, separator_start

        return records, spans, None

    @classmethod
    def _bare_json_prefix_is_viable(cls, text: str) -> bool:
        """Return whether an incomplete prefix can still be a bare tool call."""
        return bool(
            cls.BARE_JSON_MARKER.search(text) or cls.BARE_JSON_PARTIAL.search(text)
        )

    @classmethod
    def _starts_with_complete_non_tool_json(cls, text: str) -> bool:
        """Return whether the leading JSON value is ordinary assistant data."""
        stripped = text.lstrip()
        if not stripped.startswith("{"):
            return False
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            return False
        return cls._tool_call_from_object(obj) is None

    @classmethod
    def _extract_xml_data(
        cls, text: str
    ) -> tuple[list[tuple[int, dict]], list[tuple[int, int]], int | None]:
        """Return ordered legacy XML calls and their source spans."""
        records: list[tuple[int, dict]] = []
        spans: list[tuple[int, int]] = []
        for match in cls.FUNCTION_PATTERN.finditer(text):
            name, args_text = match.groups()
            try:
                arguments = json.loads(args_text)
                serialized = json.dumps(arguments, ensure_ascii=False)
            except json.JSONDecodeError:
                serialized = args_text
            records.append(
                (
                    match.start(),
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": serialized,
                    },
                )
            )
            spans.append(match.span())

        pending_start = None
        last_open = text.rfind("<function=")
        last_close = text.rfind("</function>")
        if last_open > last_close:
            pending_start = last_open
        return records, spans, pending_start

    @classmethod
    def _stream_state(cls, text: str) -> tuple[list[dict], str, bool]:
        """Return parsed calls, currently visible content, and pending state."""
        python_records, python_spans, python_pending = cls._extract_python_tag_data(
            text
        )
        xml_records, xml_spans, xml_pending = cls._extract_xml_data(text)
        records = sorted(python_records + xml_records, key=lambda item: item[0])
        spans = python_spans + xml_spans
        pending_candidates = [
            start for start in (python_pending, xml_pending) if start is not None
        ]

        if not records and not spans and not pending_candidates:
            bare_records, bare_spans, bare_pending = cls._extract_bare_json_data(text)
            if bare_records or bare_pending is not None:
                records = bare_records
                spans = bare_spans
                if bare_pending is not None:
                    pending_candidates.append(bare_pending)
            elif not cls._starts_with_complete_non_tool_json(
                text
            ) and cls._bare_json_prefix_is_viable(text):
                pending_candidates.append(0)

        visible_limit = min(pending_candidates) if pending_candidates else len(text)
        visible_spans = [
            (start, min(end, visible_limit))
            for start, end in spans
            if start < visible_limit
        ]
        visible = cls._remove_spans(text[:visible_limit], visible_spans)
        return [call for _, call in records], visible, bool(pending_candidates)

    @staticmethod
    def _visible_delta(previous: str, current: str) -> str:
        """Return content newly made visible by the current parser state."""
        if current.startswith(previous):
            return current[len(previous) :]
        return current

    @staticmethod
    def _format_streaming_tool_calls(
        calls: list[dict], start_index: int = 0
    ) -> dict[str, Any]:
        """Format newly completed calls for an OpenAI streaming delta."""
        return {
            "tool_calls": [
                {
                    "index": start_index + index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for index, call in enumerate(calls)
            ]
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Llama model response.

        Tries three formats in order:
        1. ``<|python_tag|>{...}`` JSON (Llama 3.1+ / 4 canonical per Meta's
           model card)
        2. Bare top-level JSON envelope ``{"name": ..., "parameters": {...}}``
           (Llama 3.3, same payload structure with no marker)
        3. Legacy ``<function=name>{...}</function>`` XML (older Llama 3.0 /
           Code Llama)

        All three can in principle coexist in the same response.
        """
        python_records, python_spans, _ = self._extract_python_tag_data(model_output)
        xml_records, xml_spans, _ = self._extract_xml_data(model_output)
        records = sorted(python_records + xml_records, key=lambda item: item[0])
        spans = python_spans + xml_spans

        if not records:
            bare_records, bare_spans, _ = self._extract_bare_json_data(model_output)
            records = bare_records
            spans = bare_spans

        tool_calls = [call for _, call in records]
        cleaned_text = self._remove_spans(model_output, spans).strip()

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
        Extract tool calls from streaming Llama model output.

        Recognises all three formats handled by ``extract_tool_calls``:
        - legacy XML ``<function=name>{...}</function>``
        - python-tag JSON ``<|python_tag|>{"name":...,"parameters":...}``
        - bare JSON envelope ``{"name":...,"parameters":...}`` (Llama 3.3)

        While the response is shaping up like a tool call the parser
        buffers (returns ``None``); once the call(s) parse end-to-end the
        result is emitted in one shot. Plain assistant content streams
        through as ``{"content": delta_text}`` per chunk, matching the
        existing behaviour for non-tool responses.
        """
        previous_calls, previous_visible, _ = self._stream_state(previous_text)
        current_calls, current_visible, pending = self._stream_state(current_text)
        content = self._visible_delta(previous_visible, current_visible)

        if len(current_calls) > len(previous_calls):
            result = self._format_streaming_tool_calls(
                current_calls[len(previous_calls) :], len(previous_calls)
            )
            if content:
                result["content"] = content
            return result

        if content:
            return {"content": content}
        if pending:
            return None
        return {"content": delta_text}

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Resolve an incomplete Llama prefix when generation has ended."""
        calls, _, pending = self._stream_state(current_text)
        if not pending:
            return None
        if not calls and current_text.lstrip().startswith("{"):
            return {"content": current_text}
        return {"content": ""}
