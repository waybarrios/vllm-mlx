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

# Pattern to quote bare string VALUES that the template omitted <|"|> around.
# This fires when a schema uses a nullable type like ["string","null"] or an
# enum field without explicit "type": the template takes the non-STRING branch
# and emits the value raw. Preceded by a value-position separator (: [ ,), a
# word starting with a letter, followed by ,/}/]. JSON literals true/false/null
# are filtered out inside the substitution. Ref: llama.cpp PR #21327.
_BARE_VALUE = re.compile(r"(?<=[:\[,])(\s*)([A-Za-z_][\w\-]*)(?=\s*[,}\]])")
_JSON_LITERALS = frozenset({"true", "false", "null"})

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


def _quote_bare_value(m: re.Match) -> str:
    """Substitution callback for _BARE_VALUE — quotes bare identifiers that
    are not JSON literals (true/false/null)."""
    ws, word = m.group(1), m.group(2)
    if word in _JSON_LITERALS:
        return m.group(0)
    return f'{ws}"{word}"'


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 tool call args to valid JSON.

    Four-step conversion (ORDER MATTERS):
    1. Extract <|"|>-delimited strings into numbered \\x00N\\x00 placeholders.
       This protects string contents from step 2's bare-key quoting -- without
       this, a string value like "key: value" would be corrupted.
    2. Quote bare keys (word: -> "word":) now that strings are safe.
    3. Quote bare string VALUES that the template emitted without <|"|>
       wrappers. Happens with nullable/enum schemas where the STRING branch
       of the template isn't taken.
    4. Restore placeholders as properly JSON-escaped strings via json.dumps().
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

    # Step 3: Quote bare string values (nullable / enum-without-type schemas)
    text = _BARE_VALUE.sub(_quote_bare_value, text)

    # Step 4: Restore captured strings as properly escaped JSON strings
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
    """

    # The chat template renders <|tool_response> (token 50) when the assistant
    # emits a tool call without its own tool_responses block — it's the signal
    # that it's the runtime's turn, not the model's. Treat it as EOG so the
    # model doesn't keep generating past the tool call.
    # Ref: llama.cpp PR #21418.
    extra_stop_tokens = ["<|tool_response>"]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response."""
        cleaned = self.strip_think_tags(model_output)

        start_idx = cleaned.find(TOOL_CALL_START)
        if start_idx == -1:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_before = cleaned[:start_idx].strip() or None

        block_start = start_idx + len(TOOL_CALL_START)
        end_idx = cleaned.find(TOOL_CALL_END, block_start)
        if end_idx == -1:
            block = cleaned[block_start:]
        else:
            block = cleaned[block_start:end_idx]

        tool_calls: list[dict[str, Any]] = []

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

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
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
        """Extract tool calls from streaming Gemma 4 model output."""
        has_start = TOOL_CALL_START in current_text

        if not has_start:
            return {"content": delta_text}

        if TOOL_CALL_END in delta_text:
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

        return None
