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

Fallback forms (issue #80): under long system prompts + multi-turn + several
tools, Gemma 4 frequently abandons the canonical brace form and instead emits
its call as plain text in `content`, using Python-style call syntax:

  e4b: <|tool_call>call:radarr_get_movies(search="Dune")
  e2b: ```tool_code
       radarr_get_movies(search="Dune")
       ```
  e2b: tool_code = radarr_get_movies(search="Dune")
       print(tool_code)

These are parsed by a fallback layer (ast-based) when the canonical parse finds
no calls, so the host can still dispatch the tool.

Reference: mlx-lm PR #1105, vllm PR #38837
"""

import ast
import json
import logging
import re
import textwrap
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

# --- Fallback forms (issue #80) ---------------------------------------------
# Parenthesized form: `call:fn(...)` — the `call:` prefix is required outside a
# code fence so we never mistake ordinary prose like "see foo()" for a call.
_CALL_PAREN_RE = re.compile(r"call:(\w+)\s*\(")
# ```tool_code ... ``` fenced block (Gemma's code-call convention).
_TOOL_CODE_FENCE_RE = re.compile(r"```tool_code\b[^\n]*\n(.*?)```", re.DOTALL)
# Unfenced `tool_code = fn(...)` assignment (e2b omits the fence, issue #83).
# The literal `tool_code =` anchor prevents ordinary assignments like
# `x = foo()` from being mistaken for a tool call.
_TOOL_CODE_ASSIGN_RE = re.compile(r"(?m)^[ \t]*tool_code\s*=\s*(\w+)\s*\(")
# Cheap marker used by the streaming path to know a fallback call may be forming.
_FALLBACK_MARKER_RE = re.compile(
    r"```tool_code\b|call:\w+\s*\(|tool_code\s*=\s*\w+\s*\("
)


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


def _find_balanced_paren(text: str, start: int) -> int:
    """Find the index of the closing ) that balances the ( at `start`.

    Python string literals ('...'/"...") are treated as opaque so that parens
    inside string argument values don't affect depth counting.

    Returns the index of the matching ) in `text`, or -1 if not found.
    """
    if len(text) - start > _MAX_ARG_BLOCK_LEN:
        return -1

    depth = 0
    i = start
    quote: str | None = None
    while i < len(text):
        c = text[i]
        if quote is not None:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
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


def _call_node_to_tool(call: ast.Call) -> tuple[str, dict[str, Any]] | None:
    """Map a Python `ast.Call` node to (function_name, kwargs_dict).

    Only keyword arguments are mapped (Gemma emits its tool calls as kwargs);
    positional args are ignored because the parameter names aren't recoverable.
    Returns None if the name or any argument value isn't a plain literal.
    """
    func = call.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr  # e.g. module.fn -> fn
    else:
        return None

    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat — can't represent
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return name, args


def _parse_python_call(src: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a single `fn(...)` Python call expression into (name, kwargs)."""
    try:
        node = ast.parse(src.strip(), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(node.body, ast.Call):
        return None
    return _call_node_to_tool(node.body)


def _parse_calls_from_code(code: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse every top-level `fn(...)` call statement in a code-fence body."""
    results: list[tuple[str, dict[str, Any]]] = []
    try:
        module = ast.parse(textwrap.dedent(code).strip(), mode="exec")
    except SyntaxError:
        return results
    for stmt in module.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            parsed = _call_node_to_tool(stmt.value)
            if parsed is not None:
                results.append(parsed)
    return results


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given [start, end) spans from `text` (handles overlaps)."""
    if not spans:
        return text
    out: list[str] = []
    last = 0
    for start, end in sorted(spans):
        if start < last:  # overlapping / contained — extend the cut
            last = max(last, end)
            continue
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out)


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

        # 1. Canonical <|"|>-delimited brace form.
        tool_calls, content_before = self._extract_canonical(cleaned)
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_before,
            )

        # 2. Fallback: Gemma often emits the call as plain content using Python
        #    call syntax — `call:fn(...)` or a ```tool_code``` block — instead of
        #    the brace form. Recover those so the host can dispatch. Ref: #80.
        fallback = self._extract_fallback(cleaned)
        if fallback is not None:
            return fallback

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def _extract_canonical(
        self, cleaned: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Parse the canonical <|tool_call>call:fn{...}<tool_call|> form.

        Returns (tool_calls, content_before). tool_calls is empty when the
        canonical markers/braces aren't present.
        """
        start_idx = cleaned.find(TOOL_CALL_START)
        if start_idx == -1:
            return [], None

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

        return tool_calls, content_before

    def _extract_fallback(self, cleaned: str) -> ExtractedToolCallInformation | None:
        """Parse the Python-style fallback forms (issue #80).

        Handles ```tool_code``` blocks (bare `fn(...)` calls) and the
        parenthesized `call:fn(...)` form. Returns None if neither is present.
        """
        tool_calls: list[dict[str, Any]] = []
        spans: list[tuple[int, int]] = []

        # ```tool_code``` fenced blocks — may contain one or more bare calls.
        for m in _TOOL_CODE_FENCE_RE.finditer(cleaned):
            for name, args in _parse_calls_from_code(m.group(1)):
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                )
            spans.append((m.start(), m.end()))

        # Parenthesized `call:fn(...)` form (outside any code fence).
        for m in _CALL_PAREN_RE.finditer(cleaned):
            if any(s <= m.start() < e for s, e in spans):
                continue  # already covered by a code-fence span
            paren_open = m.end() - 1
            paren_close = _find_balanced_paren(cleaned, paren_open)
            if paren_close == -1:
                continue
            call_src = m.group(1) + cleaned[paren_open : paren_close + 1]
            parsed = _parse_python_call(call_src)
            if parsed is None:
                continue
            name, args = parsed
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
            spans.append((m.start(), paren_close + 1))

        # Unfenced `tool_code = fn(...)` assignment form (e2b, issue #83).
        # Only handle lines NOT already captured by a fence or call: span.
        for m in _TOOL_CODE_ASSIGN_RE.finditer(cleaned):
            if any(s <= m.start() < e for s, e in spans):
                continue  # inside an already-captured block
            paren_open = m.end() - 1
            paren_close = _find_balanced_paren(cleaned, paren_open)
            if paren_close == -1:
                continue
            call_src = m.group(1) + cleaned[paren_open : paren_close + 1]
            parsed = _parse_python_call(call_src)
            if parsed is None:
                continue
            name, args = parsed
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
            spans.append((m.start(), paren_close + 1))

        if not tool_calls:
            return None

        content = _strip_spans(cleaned, spans)
        # Drop any stray Gemma tool-call markers left in the surrounding text.
        content = content.replace(TOOL_CALL_START, "").replace(TOOL_CALL_END, "")
        # Drop residual `print(tool_code)` lines left by the unfenced assignment
        # convention — they are bookkeeping noise, not reply content.
        content = re.sub(r"(?m)^[ \t]*print\([^\n]*\)[ \t]*$", "", content)
        content = content.strip() or None

        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

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
        has_canonical = TOOL_CALL_START in current_text
        has_fallback = bool(_FALLBACK_MARKER_RE.search(current_text))

        if not has_canonical and not has_fallback:
            return {"content": delta_text}

        # Canonical brace form: emit when the end delimiter arrives in this delta.
        if has_canonical and TOOL_CALL_END in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return self._format_streaming(result)
            return None

        # Fallback forms (`call:fn(...)` / ```tool_code```) have no end delimiter.
        # Emit once, on the delta that first makes the call parseable.
        if has_fallback and not self.extract_tool_calls(previous_text).tools_called:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return self._format_streaming(result)

        return None
