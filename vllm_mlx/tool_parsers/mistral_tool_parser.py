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

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def generate_mistral_tool_id() -> str:
    """
    Generate a random Mistral-compatible tool call ID.

    Mistral Tool Call IDs must be alphanumeric with a length of 9.
    """
    return "".join(choices(ALPHANUMERIC, k=9))


def _is_plain_tool_name(name: str) -> bool:
    """Return True for names that are safe to dispatch as function calls."""
    return bool(_TOOL_NAME_PATTERN.match(name))


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
    _NAME_BUFFER_LIMIT = 256

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.bot_token_id = self.vocab.get(self.BOT_TOKEN) if self.vocab else None
        # Streaming state for the name/arguments boundary within the current
        # tool call. See _parse_streaming_tool_delta.
        self._args_started: bool = False
        # Quote state carried across argument deltas, used to tell a
        # [TOOL_CALLS] marker inside a JSON string (argument data) from a
        # marker between two calls (a new index). See
        # _scan_args_for_new_call.
        self._args_in_string: bool = False
        self._args_escaped: bool = False
        self._name_buffer: str = ""
        # Set when the boundary marker never arrives and the withheld text
        # was flushed as content; subsequent deltas pass through as content.
        self._name_buffer_overflow: bool = False
        # One id per active tool call, generated when the call starts and
        # attached to whichever delta is the first to carry real content
        # (name and/or arguments) — that may not be the delta containing
        # BOT_TOKEN itself, since the name can still be buffered pending the
        # [ARGS]/`{` boundary. See _parse_streaming_tool_delta.
        self._current_tool_call_id: str | None = None
        self._tool_call_id_emitted: bool = False

    def reset(self) -> None:
        super().reset()
        self._args_started = False
        self._args_in_string = False
        self._args_escaped = False
        self._name_buffer = ""
        self._name_buffer_overflow = False
        self._current_tool_call_id = None
        self._tool_call_id_emitted = False

    def _start_new_tool_call(self) -> None:
        """Begin a new streaming tool call: bump the index and reset the
        per-call name/arguments and id state."""
        self.current_tool_id += 1
        self._args_started = False
        self._args_in_string = False
        self._args_escaped = False
        self._name_buffer = ""
        self._name_buffer_overflow = False
        self._current_tool_call_id = generate_mistral_tool_id()
        self._tool_call_id_emitted = False

    def _scan_args_for_new_call(self, text: str) -> int:
        """Scan an argument delta, updating the persistent JSON string state,
        and return the position of the first [TOOL_CALLS] marker that sits
        outside a string (a new call), or -1 when there is none.

        Quote state is carried across deltas so a marker inside a quoted
        value (e.g. ``{"city": "[TOOL_CALLS]rm"}``) stays argument data while
        a marker between two calls opens the next index.
        """
        i = 0
        while i < len(text):
            ch = text[i]
            if self._args_escaped:
                self._args_escaped = False
                i += 1
                continue
            if self._args_in_string:
                if ch == "\\":
                    self._args_escaped = True
                elif ch == '"':
                    self._args_in_string = False
                i += 1
                continue
            if ch == '"':
                self._args_in_string = True
                i += 1
                continue
            if text.startswith(self.BOT_TOKEN, i):
                return i
            i += 1
        return -1

    def _split_on_tool_call_markers(self, text: str) -> list[str]:
        """Split on [TOOL_CALLS] occurrences that are outside JSON strings.

        A marker appearing inside a quoted string value is argument data,
        not a new call — splitting there would let untrusted model output
        forge a second dispatchable call.

        The quote-state scan starts at the first marker, not at index 0: the
        text before the first marker is prose, not JSON, so an odd number of
        double quotes there must not leave ``in_string`` set when the marker
        arrives (that would hide the call entirely).
        """
        token = self.BOT_TOKEN
        first = text.find(token)
        if first == -1:
            return [text]

        parts: list[str] = [text[:first]]
        start = first + len(token)
        in_string = False
        escaped = False
        i = start
        while i < len(text):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if text.startswith(token, i):
                parts.append(text[start:i])
                start = i + len(token)
                i += len(token)
                continue
            i += 1
        parts.append(text[start:])
        return parts

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

        content_and_raw_tool_calls = self._split_on_tool_call_markers(model_output)
        content = content_and_raw_tool_calls[0].strip()
        raw_tool_calls = content_and_raw_tool_calls[1:]

        tool_calls = []

        for raw_tool_call in raw_tool_calls:
            raw_tool_call = raw_tool_call.strip()
            if not raw_tool_call:
                continue

            # Try newest format first: func_name[ARGS]{"arg": "value"}.
            # The marker is the boundary only when it comes before the first
            # `{` — a legacy call whose JSON arguments contain the literal
            # "[ARGS]" substring must keep the `{` boundary.
            args_idx = raw_tool_call.find(self.ARGS_TOKEN)
            brace_idx = raw_tool_call.find("{")
            if (
                not raw_tool_call.startswith("[")
                and args_idx != -1
                and (brace_idx == -1 or args_idx < brace_idx)
            ):
                tool_name = raw_tool_call[:args_idx].strip()
                args_str = raw_tool_call[args_idx + len(self.ARGS_TOKEN) :]

                if tool_name and _is_plain_tool_name(tool_name):
                    try:
                        json.loads(args_str)
                    except json.JSONDecodeError:
                        # Malformed arguments — reject rather than emit a
                        # corrupted or forged call.
                        continue
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
        # Everything after the name/arguments boundary is arguments text.
        # A [TOOL_CALLS] marker inside a quoted JSON string value is argument
        # data, not a new call; a marker outside a string starts the next
        # call (its own index). The end-of-stream non-streaming re-parse
        # recovers calls that arrive inside a shared delta.
        if self._args_started:
            new_call_pos = self._scan_args_for_new_call(delta_text)
            if new_call_pos == -1:
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_id,
                            "type": "function",
                            "function": {"arguments": delta_text},
                        }
                    ]
                }
            # A new call begins inside this delta: close the current call with
            # the pre-marker text, then start the next one.
            result: dict[str, Any] = {}
            pre = delta_text[:new_call_pos]
            if pre:
                result["tool_calls"] = [
                    {
                        "index": self.current_tool_id,
                        "type": "function",
                        "function": {"arguments": pre},
                    }
                ]
            self._start_new_tool_call()
            tool_delta = self._parse_streaming_tool_delta(
                delta_text[new_call_pos + len(self.BOT_TOKEN) :]
            )
            if tool_delta:
                tool_call: dict[str, Any] = {
                    "index": self.current_tool_id,
                    "type": "function",
                    "function": tool_delta,
                }
                tool_call["id"] = self._current_tool_call_id
                self._tool_call_id_emitted = True
                result["tool_calls"] = (result.get("tool_calls") or []) + [tool_call]
            return result if result else None

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
            self._start_new_tool_call()

            if tool_part:
                # Try to parse the tool part
                tool_delta = self._parse_streaming_tool_delta(tool_part)
                if tool_delta:
                    tool_call: dict[str, Any] = {
                        "index": self.current_tool_id,
                        "type": "function",
                        "function": tool_delta,
                    }
                    tool_call["id"] = self._current_tool_call_id
                    self._tool_call_id_emitted = True
                    result["tool_calls"] = [tool_call]

            return result if result else None

        # We're in the middle of a tool call
        if self.current_tool_id >= 0:
            if self._name_buffer_overflow:
                # The boundary never arrived; everything is plain text now.
                return {"content": delta_text}
            tool_delta = self._parse_streaming_tool_delta(delta_text)
            if tool_delta:
                if self._name_buffer_overflow:
                    # The withheld name text was flushed as content instead
                    # of a tool call — pass it through unlabeled.
                    return {"content": tool_delta["content"]}
                tool_call = {
                    "index": self.current_tool_id,
                    "type": "function",
                    "function": tool_delta,
                }
                # The name may still have been buffered pending the
                # [ARGS]/`{` boundary when the BOT_TOKEN delta arrived, so
                # this may be the first delta with real content — attach the
                # id exactly once, whichever delta that turns out to be.
                if not self._tool_call_id_emitted:
                    tool_call["id"] = self._current_tool_call_id
                    self._tool_call_id_emitted = True
                return {"tool_calls": [tool_call]}

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
        args_idx = self._name_buffer.find(self.ARGS_TOKEN)
        brace_idx = self._name_buffer.find("{")
        if args_idx != -1 and (brace_idx == -1 or args_idx < brace_idx):
            # The [ARGS] marker is the boundary when it precedes the first
            # `{`. A legacy call whose JSON arguments contain the literal
            # "[ARGS]" substring must keep the `{` boundary.
            boundary = self.ARGS_TOKEN
            idx = args_idx
            args_start = idx + len(boundary)
        elif brace_idx != -1:
            boundary = "{"
            idx = brace_idx
            args_start = idx
        else:
            boundary = None
            idx = -1
            args_start = -1

        if boundary is not None:
            name = self._name_buffer[:idx].strip()
            args = self._name_buffer[args_start:]
            self._args_started = True
            result: dict[str, str] = {}
            if name:
                result["name"] = name
            if args:
                result["arguments"] = args
            return result if result else None

        # Marker not seen yet — withhold rather than guess. If it never
        # arrives (truncation, or the model deviating into prose), flush the
        # withheld text as content so the response is not silently lost.
        if len(self._name_buffer) > self._NAME_BUFFER_LIMIT:
            overflowed = self._name_buffer
            self._name_buffer = ""
            self._name_buffer_overflow = True
            return {"content": overflowed}
        return None
