# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Supports parsing tool calls from multiple model formats:
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Llama: <function=name>{"arg": "value"}</function>

Also includes structured output (JSON Schema) utilities:
- parse_json_output: Extract JSON from model output
- validate_json_schema: Validate JSON against a schema
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import validate, ValidationError

from .models import FunctionCall, ResponseFormat, ToolCall


def _looks_like_tool_call(obj: Any) -> bool:
    """
    Heuristic: decide whether a parsed JSON object really represents a tool
    call as opposed to user data that happens to carry a ``"name"`` field.

    The OpenAI tool-call wire format ALWAYS has both ``"name"`` and
    ``"arguments"``. Accepting bare ``{"name": ...}`` (previous behaviour)
    caused ``response_format={"type": "json_schema"}`` payloads with a
    ``name`` field to be hijacked as fake tool calls (observed on
    MiniMax-M2: ``{"name": "John", "age": 25}`` -> ``function.name="John"``).

    Args:
        obj: Parsed JSON object.

    Returns:
        True if obj looks like a tool call, False otherwise.
    """
    if not isinstance(obj, dict):
        return False
    if "name" not in obj or "arguments" not in obj:
        return False
    if not isinstance(obj["name"], str) or not obj["name"]:
        return False
    # ``arguments`` must be a JSON-encoded string or a dict per OpenAI spec.
    args = obj["arguments"]
    return isinstance(args, (dict, str))


def _parse_raw_json_tool_calls(text: str) -> Optional[List[dict]]:
    """
    Parse raw JSON tool calls from model output.

    Handles:
    - Single JSON object: {"name": "func", "arguments": {...}}
    - Multiple objects separated by commas: {...}, {...}
    - JSON array: [{...}, {...}]

    Only accepts objects that carry both ``name`` AND ``arguments`` fields
    to avoid hijacking user data emitted via ``response_format``.

    Args:
        text: Raw model output text

    Returns:
        List of tool call dicts with 'name' and 'arguments', or None if no valid tool calls found
    """
    if not text:
        return None

    text = text.strip()

    # Try JSON array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if (
                isinstance(parsed, list)
                and parsed
                and all(_looks_like_tool_call(item) for item in parsed)
            ):
                return [
                    {"name": item["name"], "arguments": item["arguments"]}
                    for item in parsed
                ]
        except json.JSONDecodeError:
            pass

    # Find JSON objects with balanced braces
    tool_calls = []
    depth = 0
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                json_str = text[start : i + 1]
                try:
                    obj = json.loads(json_str)
                    if _looks_like_tool_call(obj):
                        tool_calls.append(
                            {"name": obj["name"], "arguments": obj["arguments"]}
                        )
                except json.JSONDecodeError:
                    pass
                start = None

    return tool_calls if tool_calls else None


def parse_tool_calls(
    text: str, request: dict[str, Any] | None = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Supports multiple formats:
    - MiniMax: <minimax:tool_call><invoke name="..."><parameter name="p">v</parameter></invoke></minimax:tool_call>
    - Qwen3 bracket: [Calling tool: function_name({"arg": "value"})]
    - Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Llama: <function=name>{"arg": "value"}</function>
    - Nemotron: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    - Raw JSON: {"name": "...", "arguments": {...}} (single or multiple)

    Args:
        text: Raw model output text

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    tool_calls = []
    cleaned_text = text

    # Pattern for MiniMax-style: <minimax:tool_call><invoke name="fn"><parameter name="p">v</parameter></invoke></minimax:tool_call>
    minimax_pattern = r"<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>"
    minimax_matches = re.findall(minimax_pattern, text, re.DOTALL)

    for invoke_block in minimax_matches:
        # Parse <invoke name="..."> blocks within the tool_call
        invoke_pattern = r'<invoke\s+name="([^"]+)">(.*?)</invoke>'
        invoke_matches = re.findall(invoke_pattern, invoke_block, re.DOTALL)

        for name, params_block in invoke_matches:
            # Parse <parameter name="...">value</parameter> pairs
            param_pattern = r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>'
            params = re.findall(param_pattern, params_block, re.DOTALL)
            arguments = {}
            for p_name, p_value in params:
                # Try to parse value as JSON (for nested objects/arrays/numbers)
                try:
                    arguments[p_name] = json.loads(p_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[p_name] = p_value

            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=json.dumps(arguments),
                    ),
                )
            )

    # Remove MiniMax tool call tags from cleaned text
    if minimax_matches:
        cleaned_text = re.sub(
            r"<minimax:tool_call>\s*.*?\s*</minimax:tool_call>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        ).strip()

    # Pattern for Qwen3 bracket-style: [Calling tool: function_name({...})]
    bracket_pattern = r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"
    bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)

    for name, args_str in bracket_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove bracket tool calls from cleaned text
    if bracket_matches:
        cleaned_text = re.sub(
            r"\[Calling tool:\s*\w+\(\{.*?\}\)\]", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Nemotron-style: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    nemotron_pattern = (
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"
    )
    nemotron_matches = re.findall(nemotron_pattern, text, re.DOTALL)

    for name, params_block in nemotron_matches:
        # Parse parameters from <parameter=name>value</parameter> format
        param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
        params = re.findall(param_pattern, params_block, re.DOTALL)
        arguments = {}
        for p_name, p_value in params:
            val = p_value.strip()
            try:
                arguments[p_name.strip()] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[p_name.strip()] = val

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(), arguments=json.dumps(arguments)
                ),
            )
        )

    # Remove Nemotron tool call tags from cleaned text
    if nemotron_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*<function=[^>]+>.*?</function>\s*</tool_call>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    # Pattern for Qwen-style tool calls: <tool_call>{"json"}</tool_call>
    qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    qwen_matches = re.findall(qwen_pattern, cleaned_text, re.DOTALL)

    for match in qwen_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove Qwen tool call tags from cleaned text
    if qwen_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Llama-style: <function=name>{"json"}</function>
    llama_pattern = r"<function=([^>]+)>(\{.*?\})</function>"
    llama_matches = re.findall(llama_pattern, cleaned_text, re.DOTALL)

    for name, args_str in llama_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    if llama_matches:
        cleaned_text = re.sub(
            r"<function=[^>]+>\{.*?\}</function>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Note: We keep <think>...</think> tags for reasoning models
    # The user may want to see the model's reasoning process

    # Fallback: Raw JSON tool calls (lowest priority)
    # Only try if no other formats matched AND the caller actually asked for
    # tools. When ``tools`` is not set, a bare JSON response is user data
    # (e.g. from ``response_format``), not a tool invocation — the fallback
    # would otherwise hijack ``{"name": "John", ...}`` into a fake
    # ``function.name="John"`` tool call (observed on MiniMax-M2).
    tools_requested = bool(request and request.get("tools"))
    if not tool_calls and tools_requested:
        raw_json_calls = _parse_raw_json_tool_calls(cleaned_text)
        if raw_json_calls:
            for call_data in raw_json_calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=call_data["name"],
                            arguments=(
                                json.dumps(call_data["arguments"])
                                if isinstance(call_data["arguments"], dict)
                                else str(call_data["arguments"])
                            ),
                        ),
                    )
                )
            # Clean the JSON from text since we parsed it as tool calls
            cleaned_text = ""

    return cleaned_text, tool_calls if tool_calls else None


def convert_tools_for_template(tools: Optional[List]) -> Optional[List[dict]]:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)

        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get(
                    "parameters", {"type": "object", "properties": {}}
                )
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(
                    tool_func, "parameters", {"type": "object", "properties": {}}
                )

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
                    },
                }
            )

    return converted if converted else None


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(
    data: Any, schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def _scan_balanced_json(text: str, start: int) -> Optional[str]:
    """
    Walk forward from ``start`` (which must point at ``{`` or ``[``) and
    return the substring that represents the first balanced JSON value,
    respecting strings and escapes. Returns ``None`` if the opening bracket
    is never closed (truncated output).
    """
    if start < 0 or start >= len(text):
        return None
    opener = text[start]
    if opener not in "{[":
        return None
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_truncated_json(fragment: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a JSON fragment whose closing brackets were cut off
    (e.g. because the model hit ``max_tokens`` mid-object).

    Strategy: scan once to determine the open-bracket stack and whether we
    ended mid-string, then try a handful of repair candidates in order of
    likelihood:

      1. Close unterminated string, close brackets.
      2. Also strip a dangling ``,`` / ``:`` before closing.
      3. Also drop a dangling key (``"k":`` or bare ``"k"``) before closing.
      4. Drop a dangling partial token (number / true / fals / nul) before
         closing.

    Returns the first candidate that ``json.loads`` accepts, or ``None``.
    """
    if not fragment:
        return None
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in fragment:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and (
                (ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")
            ):
                stack.pop()
    if not stack and not in_string:
        return None  # Nothing to repair — caller already tried json.loads.

    def _close(text: str) -> str:
        for opener in reversed(stack):
            text += "}" if opener == "{" else "]"
        return text

    # Base: close unterminated string if any.
    base = fragment
    if in_string:
        if escape:
            base = base[:-1]  # drop trailing backslash so closing quote sticks
        base += '"'

    candidates: list[str] = []

    # 1. Close brackets directly.
    candidates.append(_close(base))

    # 2. Strip a dangling separator (trailing ``,`` / ``:`` / whitespace).
    stripped_sep = re.sub(r"[,:\s]+$", "", base)
    if stripped_sep != base:
        candidates.append(_close(stripped_sep))

    # 3. Drop a dangling key (``"k":`` or bare ``"k"``) inside an object.
    if stack and stack[-1] == "{":
        no_key = re.sub(r',?\s*"[^"]*"\s*:?\s*$', "", stripped_sep)
        if no_key != stripped_sep:
            candidates.append(_close(no_key))

    # 4. Drop a dangling partial scalar (number / keyword / literal).
    no_scalar = re.sub(
        r",?\s*(?:-?\d+(?:\.\d*)?(?:[eE][+-]?\d*)?|t|tr|tru|f|fa|fal|fals|n|nu|nul)$",
        "",
        stripped_sep,
    )
    if no_scalar != stripped_sep:
        candidates.append(_close(no_scalar))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output text.

    Tries multiple strategies, in order of specificity:

    1. Parse entire text as JSON
    2. Extract JSON from complete markdown code blocks (``` ... ```)
    3. Extract JSON from an unterminated markdown code block (``` json\\n{ ... )
       — handles the common "chatty + truncation" failure mode where the
       model starts a ```json fence, never closes it, then hits max_tokens.
    4. Balanced-brace scan for the first ``{`` or ``[`` in the text
    5. Repair truncated JSON by closing unclosed brackets/strings

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from complete markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Unterminated markdown fence — take everything after the
    # last ``` opener. Common truncation case for chatty models.
    unterminated_fence = re.search(r"```(?:json)?\s*\n?([\s\S]*)$", text)
    fenced_candidate: Optional[str] = None
    if unterminated_fence:
        fenced_candidate = unterminated_fence.group(1).strip()
        # Drop a trailing ``` if it slipped through the greedy match.
        if fenced_candidate.endswith("```"):
            fenced_candidate = fenced_candidate[:-3].strip()
        try:
            return json.loads(fenced_candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Balanced-brace scan for the first JSON value anywhere.
    # Preserves correctness better than a greedy regex (which would grab
    # everything between the first ``{`` and the last ``}``).
    for opener in ("{", "["):
        idx = text.find(opener)
        while idx != -1:
            candidate = _scan_balanced_json(text, idx)
            if candidate is not None:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            idx = text.find(opener, idx + 1)

    # Strategy 5: Repair a truncated JSON prefix. Try the fenced candidate
    # first (usually starts right at ``{``), then fall back to the earliest
    # ``{``/``[`` in the full text.
    candidates = []
    if fenced_candidate:
        candidates.append(fenced_candidate)
    for opener in ("{", "["):
        idx = text.find(opener)
        if idx != -1:
            candidates.append(text[idx:])
    for fragment in candidates:
        repaired = _repair_truncated_json(fragment)
        if repaired is not None:
            return repaired

    return None


class StreamingJsonFenceStripper:
    """Strip markdown code fences from streamed content when response_format is set.

    Without guided decoding, chat models often wrap their JSON output in markdown
    fences (```json ... ```) even when the system prompt says not to. The non-
    streaming path strips those via ``extract_json_from_text`` / ``parse_json_output``,
    but the streaming path used to emit the raw deltas, so clients got
    ``"```json{...}```"`` instead of ``"{...}"``.

    This filter buffers just enough text to detect:
      * a leading fence like ``"```"``, ``"```json"``, ``"```\\n"`` or
        ``"```json\\n"`` (with optional leading whitespace), possibly split
        across SSE deltas, and
      * a trailing fence like ``"```"`` or ``"\\n```\\n"`` on stream end.

    Leading-whitespace and leading fences are consumed; trailing fences are
    dropped in :meth:`finalize`. Non-fenced content passes through with at most
    a ``_TAIL_HOLDBACK``-char delay.
    """

    # Opening fence forms to strip, longest first (longest match wins).
    _OPENINGS = ("```json\n", "```json", "```\n", "```")
    # Characters held back at the tail to detect a trailing fence across deltas.
    # 5 covers ``"\n```\n"``.
    _TAIL_HOLDBACK = 5

    def __init__(self) -> None:
        self._buf: str = ""
        self._past_opening: bool = False

    def feed(self, delta: str) -> str:
        """Append a content delta and return the portion safe to emit now."""
        if not delta:
            return ""

        self._buf += delta

        if not self._past_opening:
            ls = self._buf.lstrip()
            if not ls:
                # Still only whitespace — wait for content.
                return ""

            # If ``ls`` is a strict prefix of any opening (shorter than the
            # opening), the rest of the fence might still arrive in the next
            # delta — keep buffering.
            for opening in self._OPENINGS:
                if len(ls) < len(opening) and opening.startswith(ls):
                    return ""

            # Try to match a complete opening fence (longest first).
            matched: Optional[str] = None
            for opening in self._OPENINGS:
                if ls.startswith(opening):
                    matched = opening
                    break

            if matched is not None:
                # Drop the fence and any immediate whitespace that followed.
                self._buf = ls[len(matched) :].lstrip()
            else:
                # Not a fence — keep the stripped text.
                self._buf = ls
            self._past_opening = True

        # Dynamic holdback: walk backwards across any trailing whitespace and
        # backticks so a closing fence cannot straddle the emit boundary, and
        # additionally keep at least ``_TAIL_HOLDBACK`` chars to absorb fences
        # that span delta boundaries.
        buf = self._buf
        i = len(buf)
        while i > 0 and (buf[i - 1] == "`" or buf[i - 1].isspace()):
            i -= 1
        safe_end = min(i, len(buf) - self._TAIL_HOLDBACK)
        if safe_end <= 0:
            return ""

        to_emit = buf[:safe_end]
        self._buf = buf[safe_end:]
        return to_emit

    def finalize(self) -> str:
        """Flush the remaining buffer, dropping any trailing fence."""
        tail = self._buf
        self._buf = ""
        if not tail:
            return ""

        if not self._past_opening:
            # Stream ended before we ever transitioned past the (potential)
            # opening — either drop a strict-prefix fence that never finished
            # arriving, or strip a complete leading fence.
            ls = tail.lstrip()
            # Check strict-prefix FIRST: ``"```js"`` is a prefix of
            # ``"```json\n"`` and should be dropped entirely rather than
            # partially matched by the bare ``"```"`` opening.
            is_partial_prefix = False
            for opening in self._OPENINGS:
                if len(ls) < len(opening) and opening.startswith(ls):
                    is_partial_prefix = True
                    break
            if is_partial_prefix:
                ls = ""
            else:
                matched: Optional[str] = None
                for opening in self._OPENINGS:
                    if ls.startswith(opening):
                        matched = opening
                        break
                if matched is not None:
                    ls = ls[len(matched) :].lstrip()
            tail = ls
            self._past_opening = True

        stripped = tail.rstrip()
        for closing in ("\n```", "```"):
            if stripped.endswith(closing):
                return stripped[: -len(closing)].rstrip()
        return tail


def parse_json_output(
    text: str, response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
) -> Tuple[str, Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    strict_rules = (
        "Output rules (STRICT):\n"
        "- Your first character MUST be `{` (or `[`).\n"
        "- Your last character MUST be `}` (or `]`).\n"
        "- Do NOT wrap the JSON in a markdown code block (no ``` fences).\n"
        "- Do NOT prepend any preamble such as "
        '"Here is the JSON" or "Let me format this".\n'
        "- Do NOT include comments, trailing explanations, or chain-of-thought.\n"
    )

    if format_type == "json_object":
        return (
            "You must respond with a single valid JSON value only.\n\n" + strict_rules
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")

        prompt = f"You must respond with a single valid JSON value matching the '{name}' schema."
        if description:
            prompt += f" {description}"
        prompt += f"\n\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        prompt += strict_rules
        return prompt

    return None
