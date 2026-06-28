# SPDX-License-Identifier: Apache-2.0
"""
Llama tool call parser for vllm-mlx.

Handles Llama's tool calling formats:
- JSON with python tag: <|python_tag|>{"name": "fn", "parameters": {...}}
    (canonical Llama-3.1+ / Llama-4 format per Meta's model card)
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

    Supports Llama tool call format:
    - <function=name>{"arg": "value"}</function>

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

    @staticmethod
    def _extract_python_tag_calls(text: str) -> tuple[list[dict], str]:
        """Find `<|python_tag|>` markers and decode each following JSON object.

        Returns (tool_calls, cleaned_text). cleaned_text has the tag+JSON
        sliced out so downstream code can treat the remainder as user-visible
        content.
        """
        calls: list[dict] = []
        cleaned = text
        cursor = 0
        while True:
            idx = cleaned.find(LlamaToolParser.PYTHON_TAG, cursor)
            if idx < 0:
                break
            json_start = idx + len(LlamaToolParser.PYTHON_TAG)
            try:
                obj, end = json.JSONDecoder().raw_decode(cleaned[json_start:].lstrip())
            except json.JSONDecodeError:
                # Malformed payload — advance past the tag so we don't infinite-loop
                cursor = json_start
                continue
            # Account for any leading whitespace that lstrip() consumed
            ws_len = len(cleaned[json_start:]) - len(cleaned[json_start:].lstrip())
            abs_end = json_start + ws_len + end
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("parameters", obj.get("arguments", {}))
                calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": str(obj["name"]).strip(),
                        "arguments": (
                            json.dumps(args, ensure_ascii=False)
                            if isinstance(args, (dict, list))
                            else str(args)
                        ),
                    }
                )
            cleaned = cleaned[:idx] + cleaned[abs_end:]
            cursor = idx  # check for more tags from this spot
        return calls, cleaned.strip()

    @staticmethod
    def _extract_bare_json_call(text: str) -> tuple[list[dict], str]:
        """Decode a bare JSON tool-call envelope. Llama 3.3 emits the tool call
        as a top-level JSON object with no surrounding tag:
            {"type": "function", "name": "fn", "parameters": {...}}
        We only treat the response as a tool call if (a) the stripped content
        parses as a single JSON object and (b) it has a ``name`` field with a
        ``parameters`` or ``arguments`` payload. This is intentionally strict
        to avoid mis-parsing user-visible JSON that the model emits for other
        reasons.
        """
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return [], text
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return [], text
        if not (
            isinstance(obj, dict)
            and "name" in obj
            and ("parameters" in obj or "arguments" in obj)
        ):
            return [], text
        args = obj.get("parameters", obj.get("arguments", {}))
        call = {
            "id": generate_tool_id(),
            "name": str(obj["name"]).strip(),
            "arguments": (
                json.dumps(args, ensure_ascii=False)
                if isinstance(args, (dict, list))
                else str(args)
            ),
        }
        return [call], ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Llama model response.

        Tries three formats in order:
        1. ``<|python_tag|>{...}`` JSON (Llama 3.1+ / 4 canonical per Meta's
           model card)
        2. Bare top-level JSON envelope ``{"name": ..., "parameters": {...}}``
           (Llama 3.3 — same payload structure, no marker)
        3. Legacy ``<function=name>{...}</function>`` XML (older Llama 3.0 /
           Code Llama)

        All three can in principle coexist in the same response.
        """
        tool_calls: list[dict] = []
        cleaned_text = model_output

        # 1. python-tag JSON
        py_calls, cleaned_text = self._extract_python_tag_calls(cleaned_text)
        tool_calls.extend(py_calls)

        # 2. Bare JSON envelope (no marker) — only if we haven't already
        #    extracted python-tag calls (avoids double-matching when both
        #    present).
        if not tool_calls:
            bare_calls, cleaned_text = self._extract_bare_json_call(cleaned_text)
            tool_calls.extend(bare_calls)

        # 2. <function=...>{...}</function> XML (legacy)
        matches = self.FUNCTION_PATTERN.findall(cleaned_text)
        for name, args_str in matches:
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
                # Keep the raw arguments string
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": args_str,
                    }
                )

        if matches:
            cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

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
        has_xml_marker = "<function=" in current_text
        has_python_tag = self.PYTHON_TAG in current_text
        # Bare-JSON discriminator: opening brace at the start of the
        # buffered text. Mirrors the existing XML behaviour — once we see
        # the opening marker we buffer until the call parses, accepting
        # that arbitrary user-output JSON also gets buffered (the same
        # trade-off the XML path makes when `<function=` lacks a close).
        looks_like_bare_envelope = current_text.lstrip().startswith("{")

        if not (has_xml_marker or has_python_tag or looks_like_bare_envelope):
            # Plain content; stream as-is.
            return {"content": delta_text}

        # Looks like (or is partway through) a tool call. Try the full
        # extractor — it succeeds only when at least one call parses
        # end-to-end, so we naturally buffer until then.
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

        # Still in flight — wait for more deltas before emitting.
        return None
