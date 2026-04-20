# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool call format:
  <|tool_call>call:func_name{<|"|>key<|"|>: <|"|>value<|"|>, num: 42}<tool_call|>

Gemma 4 uses special tokens instead of JSON:
- <|tool_call> / <tool_call|> delimit tool call blocks
- <|"|> replaces " for string values
- Keys are unquoted bare identifiers
- Multiple call:name{...} can appear in a single block

Reference: mlx-lm PR #1105, vllm PR #38837
"""

import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

# Delimiters
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"

# Placeholder token used during <|"|> extraction. Matches \x00 + digits + \x00.
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

# Pattern to extract <|"|>-delimited strings (non-greedy, supports multiline)
_STRING_DELIM_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)

# Pattern to match call:name followed by a { (we extract balanced braces manually)
_CALL_PREFIX = re.compile(r"call:(\w+)\s*\{")

# Pattern to quote bare keys: word followed by : at start or after , or {
_BARE_KEY = re.compile(r"(?<=[{,])\s*(\w+)\s*:")

# Max arg block length to prevent runaway parsing on malformed input (1 MB)
_MAX_ARG_BLOCK_LEN = 1_048_576


def _find_balanced_brace(text: str, start: int) -> int:
    """Find the index of the closing } that balances the { at `start`.

    Before counting braces, <|"|>-delimited strings are conceptually opaque --
    we skip over <|"|>...<|"|> regions so that braces inside string values
    (e.g. code snippets) don't affect depth counting.

    Args:
        text: The string to search (may contain <|"|> tokens)
        start: Index of the opening {

    Returns:
        Index of the matching } in the ORIGINAL text, or -1 if not found
    """
    if len(text) - start > _MAX_ARG_BLOCK_LEN:
        return -1

    depth = 0
    i = start
    in_string = False
    while i < len(text):
        if text.startswith('<|"|>', i):
            in_string = not in_string
            i += 5
            continue
        if not in_string:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 tool call args to valid JSON.

    Three-step conversion (ORDER MATTERS):
    1. Extract <|"|>-delimited strings into numbered \\x00N\\x00 placeholders.
       This protects string contents from step 2's bare-key quoting -- without
       this, a string value like "key: value" would be corrupted.
    2. Quote bare keys (word: -> "word":) now that strings are safe.
    3. Restore placeholders as properly JSON-escaped strings via json.dumps().
       Uses a single re.sub pass (O(len(text))) instead of per-placeholder replace.
    """
    strings: list[str] = []

    def _capture(m: re.Match) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    # Step 1: Extract <|"|>-delimited strings
    text = _STRING_DELIM_RE.sub(_capture, text)

    # Step 2: Quote bare keys
    text = _BARE_KEY.sub(r'"\1":', text)

    # Step 3: Restore captured strings as properly escaped JSON strings
    def _restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return json.dumps(strings[idx]) if idx < len(strings) else m.group(0)

    text = _PLACEHOLDER_RE.sub(_restore, text)

    return text


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module("gemma4")
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Parses: <|tool_call>call:func{<|"|>key<|"|>: <|"|>val<|"|>}<tool_call|>

    Used when --enable-auto-tool-choice --tool-call-parser gemma4 are set.

    Native tool format is supported end-to-end: Gemma 4's chat template
    renders ``role: tool`` messages via a forward-scan inside the
    preceding assistant turn (``<|tool_response>response:name{...}<tool_response|>``)
    and expects tool_call ``arguments`` as a dict (not a JSON-string) so
    its mapping-render branch can emit ``key:<|"|>value<|"|>`` pairs
    instead of leaking OpenAI's ``{"key":"value"}`` JSON verbatim —
    which produces the double-brace ``{{"k":"v"}}`` pathology.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Adapt OpenAI-style ``role: tool`` messages to Gemma 4's
        expected dict-shape tool-response content.

        OpenAI's Chat Completions API has ``role: tool`` messages
        carrying string ``content`` (e.g. file contents, shell stdout).
        Gemma 4's chat template's string-response branch wraps that as
        ``response:name{value:<|"|>...<|"|>}``, which the model wasn't
        trained on — it was trained on dict-shape responses matching
        the tool's declared output schema (``response:name{field1:val1,
        field2:val2}``). When the model sees the ``{value:...}`` synthetic
        key, it doesn't recognize the tool call as complete, silently
        ignores the response, and emits a fresh identical ``<|tool_call>``
        on the next turn — infinite tool-call loop.

        Wrapping the string as ``{"content": string}`` routes through
        the template's mapping branch and produces
        ``response:name{content:<|"|>string<|"|>}``. ``content`` matches
        OpenAI's own field name for tool message bodies, and the model
        recognizes the dict shape as a completed response.

        Verified empirically: without the wrap, opencode + Gemma 4 loops
        on every tool call; with the wrap, the model answers referencing
        the tool output as expected.
        """
        adapted: list[dict[str, Any]] = []
        for msg in messages:
            if (
                msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
            ):
                new_msg = dict(msg)
                new_msg["content"] = {"content": msg["content"]}
                adapted.append(new_msg)
            else:
                adapted.append(msg)
        return adapted

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response.

        Handles both layouts the wild can produce:

        1. Single-block, multi-call (per the tool-parser docstring /
           mlx-lm PR #1105): one ``<|tool_call>`` ... ``<tool_call|>``
           enclosing several ``call:name{...}`` entries.
        2. Multi-block (what the chat_template.jinja currently emits for
           parallel tool calls): one ``<|tool_call>call:A{...}<tool_call|>``
           immediately followed by ``<|tool_call>call:B{...}<tool_call|>``,
           etc.

        Iterating every delimited block means whichever shape the model
        produces, we don't silently drop calls 2..N.
        """
        cleaned = self.strip_think_tags(model_output)

        first_start = cleaned.find(TOOL_CALL_START)
        if first_start == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_before = cleaned[:first_start].strip() or None
        tool_calls: list[dict[str, Any]] = []

        cursor = first_start
        while cursor < len(cleaned):
            start_idx = cleaned.find(TOOL_CALL_START, cursor)
            if start_idx == -1:
                break
            block_start = start_idx + len(TOOL_CALL_START)
            end_idx = cleaned.find(TOOL_CALL_END, block_start)
            if end_idx == -1:
                # Unterminated block at the tail — parse what we have
                # and stop iterating.
                block = cleaned[block_start:]
                cursor = len(cleaned)
            else:
                block = cleaned[block_start:end_idx]
                cursor = end_idx + len(TOOL_CALL_END)
            self._extract_calls_from_block(block, tool_calls)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def _extract_calls_from_block(
        self, block: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        """Parse every ``call:name{...}`` sub-section inside a single
        ``<|tool_call>...<tool_call|>`` block and append them to
        ``tool_calls``.
        """
        pos = 0
        while pos < len(block):
            m = _CALL_PREFIX.search(block, pos)
            if not m:
                break

            func_name = m.group(1)
            brace_start = m.end() - 1

            brace_end = _find_balanced_brace(block, brace_start)
            if brace_end == -1:
                pos = m.end()
                continue

            args_raw = block[brace_start : brace_end + 1]
            try:
                args_json = _gemma4_args_to_json(args_raw)
                json.loads(args_json)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": args_json,
                    }
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Gemma 4 tool parser: failed to parse args for "
                    f"call:{func_name}: {e}"
                )

            pos = brace_end + 1

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
        """Extract tool calls from streaming Gemma 4 model output.

        OpenAI delta semantics: each streamed ``tool_calls`` entry gets
        a stable ``id`` that the client concatenates deltas into, keyed
        by ``index``. This implementation emits newly-completed tool
        call blocks once, in order, and uses ``self.prev_tool_call_arr``
        (reset per stream by the base class) to avoid re-emitting
        earlier blocks — which would otherwise regenerate fresh UUIDs
        every time a later ``<tool_call|>`` arrived.
        """
        if TOOL_CALL_START not in current_text:
            return {"content": delta_text}

        # Only emit when at least one block has completed in this delta.
        if TOOL_CALL_END not in delta_text:
            return None

        result = self.extract_tool_calls(current_text)
        if not result.tools_called:
            return None

        already_emitted = len(self.prev_tool_call_arr)
        new_calls = result.tool_calls[already_emitted:]
        if not new_calls:
            return None

        self.prev_tool_call_arr.extend(new_calls)

        return {
            "tool_calls": [
                {
                    "index": already_emitted + i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for i, tc in enumerate(new_calls)
            ]
        }
