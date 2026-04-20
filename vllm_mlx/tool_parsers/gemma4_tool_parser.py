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
        """Coalesce OpenAI ``role:tool`` messages into the preceding
        assistant's ``tool_responses`` field — the Gemma-native layout
        the chat template's legacy branch handles correctly.

        OpenAI's Chat Completions shape:

            [assistant{tool_calls:[A,B]}, tool{tool_call_id:A.id, content},
             tool{tool_call_id:B.id, content}, ...]

        Gemma 4 native shape (same data, different layout):

            [assistant{tool_calls:[A,B],
                       tool_responses:[{name:A.name, response:{content:"..."}},
                                       {name:B.name, response:{content:"..."}}]}]

        Why this matters: the chat template has two paths for tool
        responses. The OpenAI forward-scan path (``elif tool_calls``)
        reads each ``role:tool`` follow-up's ``content`` and dispatches
        on type — and Jinja2's ``is sequence`` returns True for dicts,
        so the dict-content branch falls into the sequence loop that
        calls ``.get('type')`` on each dict *key* (string), which
        crashes. The legacy ``if tool_responses`` path calls
        ``format_tool_response_block`` directly with the dict response,
        which routes through that macro's ``is mapping`` branch and
        renders the Gemma-native ``response:name{field:value,...}``.

        Wrapping the raw string as ``{"content": str}`` and placing the
        pair inside ``tool_responses`` lets the template's legacy branch
        do its job — the model sees
        ``response:name{content:<|"|>str<|"|>}``, which matches its
        training data and unblocks the tool loop. No template edits.

        Verified empirically: a raw /v1/completions with this exact
        shape produces coherent ``The README says ...`` output against
        the same weights that loop on the template's default string
        wrapper (``response:name{value:<|"|>...<|"|>}``).
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
            # Parse arg strings into dicts (no-op when already dict).
            # The template's tool-call rendering only works on dicts;
            # server.py does the same for native-format parsers but
            # this keeps the parser's prepare_messages self-contained
            # for callers that skip the server-layer conversion.
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

            # Forward-scan consecutive role:tool messages and convert
            # each into a tool_responses entry keyed to the matching
            # tool_call's name (so the rendered response:<name>{...}
            # matches the call:<name>{...} above it).
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
                    # Multimodal tool content (e.g. screenshot return). The
                    # template's dict branch iterates keys, so coalesce any
                    # text parts into a single string under ``content`` and
                    # pass image/etc parts through under type-keyed fields
                    # preserving order. For the common text-only case this
                    # matches the string branch output.
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
