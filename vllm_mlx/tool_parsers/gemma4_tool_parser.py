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

    Native tool format: Gemma 4's template expects tool_call ``arguments``
    as a dict (not a JSON string) so its mapping branch emits
    ``key:<|"|>value<|"|>`` pairs — raw JSON leaks as ``{{"k":"v"}}``.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # The chat template renders <|tool_response> (token 50) when the assistant
    # emits a tool call without its own tool_responses block — it's the signal
    # that it's the runtime's turn, not the model's. Treat it as EOG so the
    # model doesn't keep generating past the tool call (ref: llama.cpp #21418).
    extra_stop_tokens = ["<|tool_response>"]

    def prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Coalesce OpenAI ``role:tool`` messages into the preceding
        assistant's ``tool_responses`` field — the Gemma-native layout.

        OpenAI shape::

            [assistant{tool_calls:[A,B]}, tool{tool_call_id:A.id, content},
             tool{tool_call_id:B.id, content}, ...]

        Gemma 4 native::

            [assistant{tool_calls:[A,B],
                       tool_responses:[{name:A.name, response:{content:...}},
                                       {name:B.name, response:{content:...}}]}]

        Routes through the template's ``is mapping`` branch, producing
        ``response:name{content:<|"|>str<|"|>}`` which matches training.
        """
        import json

        adapted: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            # Pass-through for anything that isn't an assistant with
            # tool_calls — the template already renders these correctly.
            if role != "assistant" or not msg.get("tool_calls"):
                adapted.append(msg)
                i += 1
                continue

            tool_calls = msg.get("tool_calls") or []
            # Template rendering requires dict arguments; parse JSON strings.
            normalized_calls = []
            id_to_name: dict[str, str] = {}
            for tc in tool_calls:
                tc_copy = dict(tc)
                func = dict(tc_copy.get("function") or {})
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                tc_copy["function"] = func
                normalized_calls.append(tc_copy)
                if tc_copy.get("id") and func.get("name"):
                    id_to_name[tc_copy["id"]] = func["name"]

            # Forward-scan consecutive role:tool messages into
            # tool_responses entries keyed by matching tool_call name.
            tool_responses: list[dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_msg = messages[j]
                name = (
                    id_to_name.get(tool_msg.get("tool_call_id") or "")
                    or tool_msg.get("name")
                    or "unknown"
                )
                content = tool_msg.get("content")
                if isinstance(content, str):
                    response = {"content": content}
                elif isinstance(content, dict):
                    response = content
                elif isinstance(content, list):
                    # Multimodal tool content — coalesce text parts into
                    # ``content``; non-text parts go to ``parts`` preserving order.
                    text_parts: list[str] = []
                    extras: list[dict[str, Any]] = []
                    for part in content:
                        if hasattr(part, "model_dump"):
                            part = part.model_dump(exclude_none=True)
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        else:
                            extras.append(part)
                    response = {"content": "".join(text_parts)}
                    if extras:
                        response["parts"] = extras
                else:
                    response = {"content": "" if content is None else str(content)}

                tool_responses.append({"name": name, "response": response})
                j += 1

            # Rebuild the assistant message with the coalesced responses.
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            new_msg["tool_calls"] = normalized_calls
            if tool_responses:
                new_msg["tool_responses"] = tool_responses
            adapted.append(new_msg)
            i = j  # skip past the consumed tool messages

        return adapted

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response.

        Handles both single-block (multi-call in one ``<|tool_call>...
        <tool_call|>``) and multi-block (back-to-back blocks) layouts.
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
        """Parse every ``call:name{...}`` inside a single tool-call block."""
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
        """Extract tool calls from streaming Gemma 4 output.
        Emits each newly-completed block once with a stable ``id``,
        using ``self.prev_tool_call_arr`` to track what was already sent.
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
