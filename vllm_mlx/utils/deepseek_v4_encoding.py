# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding for DeepSeek-V4 (Pro/Flash).

DeepSeek-V4 ships no Jinja ``chat_template`` — its ``tokenizer_config.json``
carries only BOS/EOS/pad. The prompt is built programmatically instead, which is
what this module does. It is a port of the reference ``encoding_dsv4.py``
published alongside the model weights; upstream vLLM solves the same problem the
same way in ``vllm/tokenizers/deepseek_v4_encoding.py``.

Format in brief::

    <｜begin▁of▁sentence｜>{system}<｜User｜>{question}<｜Assistant｜><think>

The system message is bare text with no wrapper — roles are delimited solely by
``<｜User｜>`` and ``<｜Assistant｜>``. A turn ends with ``<think>`` in thinking
mode (the model then reasons) or ``</think>`` in chat mode (reasoning
suppressed). There is no ``tool`` role: tool results are merged into the
preceding user turn as ``<tool_result>`` blocks.
"""

import copy
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

# Special tokens for DeepSeek-internal classification tasks. Not reachable
# through the OpenAI API surface, but render_message() honours them so the
# encoder stays a faithful port.
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS)

TOOL_CALLS_BLOCK_NAME = "tool_calls"

# Markers delimiting a tool call block. Defined here, with the rest of the wire
# format, so the encoder and both parsers cannot drift apart.
TOOL_CALLS_START = f"<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"
TOOL_CALLS_END = f"</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"


def partial_marker_len(text: str, *markers: str) -> int:
    """Length of the trailing run of ``text`` that is still a prefix of one of
    ``markers`` (the tool-call start marker by default).

    Streaming deltas are token-sized. ``<｜DSML｜tool_calls>`` is not a single
    token — only the bare ``｜DSML｜`` is — so the marker always arrives in
    pieces. A parser that emits those pieces as they come leaks markup into the
    user-visible stream and then repeats the whole marker once it recognises
    it. Withholding this many trailing characters avoids both.

    Returns 0 when the tail cannot grow into any of the markers.
    """
    markers = markers or (TOOL_CALLS_START,)
    longest = max(len(m) for m in markers)
    for size in range(min(len(text), longest - 1), 0, -1):
        tail = text[-size:]
        if any(m.startswith(tail) for m in markers):
            return size
    return 0


ASSISTANT_MSG_TEMPLATE = "{reasoning}{content}{tool_calls}" + EOS_TOKEN
ASSISTANT_MSG_WO_EOS_TEMPLATE = "{reasoning}{content}{tool_calls}"
TOOL_CALL_TEMPLATE = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
TOOL_CALLS_TEMPLATE = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
TOOL_OUTPUT_TEMPLATE = "<tool_result>{content}</tool_result>"
RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\nYou MUST strictly adhere to the following "
    "schema to reply:\n{schema}"
)

# Reasoning effort is a plain text prefix prepended to the whole conversation,
# not a token and not a sampling parameter. "low" is the default and adds
# nothing; the prefix only applies in thinking mode.
REASONING_EFFORT_PROMPTS: dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def to_json(value: Any) -> str:
    """Serialize to JSON, falling back to ASCII escaping if needed."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools: list[dict]) -> list[dict]:
    """Strip the OpenAI ``{"type": "function", "function": {...}}`` wrapper."""
    return [tool["function"] if "function" in tool else tool for tool in tools]


def tool_calls_from_openai_format(tool_calls: list[dict]) -> list[dict]:
    return [
        {
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        }
        for tc in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: dict[str, str]) -> str:
    """Render one tool call's arguments as DSML ``parameter`` elements.

    ``string="true"`` marks a raw string value; anything else is JSON-encoded
    and marked ``string="false"``. Arguments that do not parse as JSON are
    wrapped under an ``arguments`` key rather than dropped.

    Accepts ``arguments`` either as the JSON string the OpenAI wire format uses
    or as an already-decoded mapping: ``api/utils.py`` decodes it in place when
    native tool format is preserved, and json-loading that again would collapse
    every parameter into one bogus ``arguments`` entry.
    """
    template = (
        '<{dsml_token}parameter name="{key}" string="{is_str}">'
        "{value}</{dsml_token}parameter>"
    )

    raw = tool_call["arguments"]
    if isinstance(raw, (dict, list)):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {"arguments": raw}

    if not isinstance(arguments, dict):
        arguments = {"arguments": arguments}

    parts = []
    for key, value in arguments.items():
        is_str = isinstance(value, str)
        parts.append(
            template.format(
                dsml_token=DSML_TOKEN,
                key=key,
                is_str="true" if is_str else "false",
                value=value if is_str else to_json(value),
            )
        )
    return "\n".join(parts)


def render_tools(tools: list[dict[str, Any]]) -> str:
    """Render tool schemas into the block that goes into the system message."""
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(to_json(t) for t in tools),
        dsml_token=DSML_TOKEN,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=THINKING_END_TOKEN,
    )


def find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Index of the last user/developer message, or -1 if there is none."""
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ("user", "developer"):
            return idx
    return -1


def render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Render a single message into its encoded form.

    Args:
        index: Position of the message to render.
        messages: The full conversation (needed for look-ahead and for
            locating the last user turn).
        thinking_mode: ``"thinking"`` or ``"chat"``.
        drop_thinking: Drop reasoning content from turns before the last user
            message.
        reasoning_effort: ``"low"`` (default), ``"high"`` or ``"max"``. Only
            applied at index 0 and only in thinking mode.
    """
    if not 0 <= index < len(messages):
        raise IndexError(f"index {index} out of range for {len(messages)} messages")
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    if reasoning_effort not in REASONING_EFFORT_PROMPTS:
        raise ValueError(
            f"Invalid reasoning effort: {reasoning_effort}, expected one of "
            f"{list(REASONING_EFFORT_PROMPTS)}"
        )
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]

    if role == "system":
        prompt += content or ""
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        if not content:
            raise ValueError(f"Invalid message for role `{role}`: {msg}")
        prompt += USER_SP_TOKEN + content
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(
                schema=to_json(response_format)
            )

    elif role == "user":
        prompt += USER_SP_TOKEN
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            prompt += "\n\n".join(_render_content_blocks(content_blocks))
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + (content or "")

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v4 has no tool role; preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            rendered = [
                TOOL_CALL_TEMPLATE.format(
                    dsml_token=DSML_TOKEN,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            tc_content = "\n\n" + TOOL_CALLS_TEMPLATE.format(
                dsml_token=DSML_TOKEN,
                tool_calls="\n".join(rendered),
                tc_block_name=TOOL_CALLS_BLOCK_NAME,
            )

        # A message following a task carries the task's output, which has no
        # thinking block of its own.
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (reasoning_content or "") + THINKING_END_TOKEN

        template = ASSISTANT_MSG_WO_EOS_TEMPLATE if wo_eos else ASSISTANT_MSG_TEMPLATE
        prompt += template.format(
            reasoning=thinking_part,
            content=content or "",
            tool_calls=tc_content,
        )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # Transition tokens are only appended when this message is the last one, or
    # when an assistant turn follows it.
    next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
    if next_role is not None and next_role not in ("assistant", "latest_reminder"):
        return prompt

    task = msg.get("task")
    if task is not None:
        if task not in VALID_TASKS:
            raise ValueError(
                f"Invalid task: '{task}'. Valid tasks are: {sorted(VALID_TASKS)}"
            )
        if task != "action":
            prompt += DS_TASK_SP_TOKENS[task]
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                THINKING_START_TOKEN
                if thinking_mode == "thinking"
                else THINKING_END_TOKEN
            )
            prompt += DS_TASK_SP_TOKENS[task]

    elif role in ("user", "developer"):
        prompt += ASSISTANT_SP_TOKEN
        if thinking_mode == "thinking" and (
            not drop_thinking or index >= last_user_idx
        ):
            prompt += THINKING_START_TOKEN
        else:
            prompt += THINKING_END_TOKEN

    return prompt


def _render_content_blocks(content_blocks: list[dict[str, Any]]) -> list[str]:
    """Render user content blocks (interleaved text and tool results)."""
    parts = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                text_parts = [
                    (
                        b.get("text", "")
                        if b.get("type") == "text"
                        else f"[Unsupported {b.get('type')}]"
                    )
                    for b in tool_content
                ]
                tool_content = "\n\n".join(text_parts)
            parts.append(TOOL_OUTPUT_TEMPLATE.format(content=tool_content))
        else:
            parts.append(f"[Unsupported {block_type}]")
    return parts


def merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold ``role: tool`` messages into the preceding user turn.

    DeepSeek-V4 has no standalone tool role — results are carried as
    ``<tool_result>`` blocks inside a user message. Consecutive tool results
    accumulate into one user turn.
    """
    merged: list[dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            block = {"type": "text", "text": msg.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(block)
            else:
                new_msg: dict[str, Any] = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reorder tool results to match the order of the calls that produced them.

    Clients may return results out of order; the model was trained on results
    that follow call order.
    """
    call_order: dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and call_order:
                ordered = sorted(
                    tool_blocks,
                    key=lambda b: call_order.get(b.get("tool_use_id", ""), 0),
                )
                it = iter(ordered)
                msg["content_blocks"] = [
                    next(it) if b.get("type") == "tool_result" else b
                    for b in msg["content_blocks"]
                ]

    return messages


def _drop_thinking_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip stale reasoning from turns before the last user message.

    User/system/tool/reminder turns survive untouched, as does everything from
    the last user message onward. Earlier assistant turns keep their content but
    lose ``reasoning_content``; earlier developer turns are dropped entirely.
    """
    last_user_idx = find_last_user_index(messages)
    keep_roles = {
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    }
    result = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)

    return result


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Encode a conversation into a DeepSeek-V4 prompt string.

    Args:
        messages: Conversation in OpenAI format. A ``tool`` role is accepted and
            folded into the preceding user turn.
        thinking_mode: ``"thinking"`` (prompt ends with ``<think>``) or
            ``"chat"`` (ends with ``</think>``, suppressing reasoning).
        context: Already-established prefix turns. When given, BOS is not
            emitted again.
        drop_thinking: Drop reasoning from turns before the last user message.
            Forced off when any message defines tools, because tool-calling
            depends on the reasoning that produced earlier calls.
        add_default_bos_token: Emit BOS at the start of the conversation.
        reasoning_effort: ``"low"``/``None``, ``"high"`` or ``"max"``. Thinking
            mode only.

    Returns:
        The prompt string, ready to tokenize.
    """
    context = context or []

    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = BOS_TOKEN if add_default_bos_token and not context else ""

    # Tool-calling conversations need their full reasoning history: the model
    # has to see why it made the earlier calls.
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


# ---------------------------------------------------------------------------
# OpenAI API adaptation
# ---------------------------------------------------------------------------

# OpenAI exposes reasoning_effort as low/medium/high; DeepSeek-V4 defines
# low/high/max prompts plus "no thinking at all". "medium" has no distinct
# prompt of its own, so it maps onto "high" — as does any unrecognised value,
# which keeps a typo from silently disabling reasoning.
_EFFORT_ALIASES = {
    "low": "low",
    "minimal": "low",
    "medium": "high",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}


def resolve_thinking(
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    thinking_mode: str | None = None,
) -> tuple[str, str | None]:
    """Map OpenAI-style knobs onto ``(thinking_mode, reasoning_effort)``.

    ``reasoning_effort="none"`` and ``enable_thinking=False`` both select chat
    mode, in which the prompt is closed with ``</think>`` and the model skips
    reasoning entirely. In that mode the effort prefix is meaningless and is
    dropped.
    """
    if thinking_mode is not None:
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Invalid thinking_mode `{thinking_mode}`")
        mode = thinking_mode
    elif reasoning_effort == "none" or enable_thinking is False:
        mode = "chat"
    else:
        mode = "thinking"

    if mode == "chat" or reasoning_effort in (None, "none"):
        return mode, None

    effort = _EFFORT_ALIASES.get(str(reasoning_effort).lower())
    if effort is None:
        logger.warning(
            "Unknown reasoning_effort %r for deepseek_v4, treating as 'high'",
            reasoning_effort,
        )
        effort = "high"
    return mode, effort


def _attach_tools(
    conversation: list[dict[str, Any]], tools: list[dict] | None
) -> list[dict[str, Any]]:
    """Put the tool schemas where the encoder expects them: on a system turn.

    OpenAI passes tools as a top-level parameter, but DeepSeek-V4 renders them
    from a message field. A conversation that already declares tools on one of
    its messages is left alone.
    """
    if not tools or any(m.get("tools") for m in conversation):
        return conversation

    conversation = [dict(m) for m in conversation]
    for msg in conversation:
        if msg.get("role") == "system":
            msg["tools"] = tools
            return conversation

    return [{"role": "system", "content": "", "tools": tools}, *conversation]


def apply_chat_template(
    conversation: list[dict[str, Any]],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    thinking_mode: str | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    **_ignored: Any,
) -> str:
    """Build a DeepSeek-V4 prompt from an OpenAI-format conversation.

    Signature-compatible with ``tokenizer.apply_chat_template`` for the kwargs
    vllm-mlx actually passes. ``add_generation_prompt`` is accepted and ignored:
    the encoder always closes on the assistant prefix, which is the only mode
    the model was trained for.
    """
    mode, effort = resolve_thinking(enable_thinking, reasoning_effort, thinking_mode)
    conversation = _attach_tools(conversation, tools)
    return encode_messages(
        conversation,
        thinking_mode=mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=effort,
    )


def install(tokenizer: Any) -> Any:
    """Route ``tokenizer.apply_chat_template`` through the V4 encoder.

    DeepSeek-V4 carries no Jinja template, so the stock path either raises or
    falls back to naive ``"role: content"`` concatenation. Overriding the method
    on the tokenizer fixes every caller at once — the two engines and
    ``models/llm.py`` all reach the template through this one method.

    Idempotent; returns the same tokenizer for convenience.
    """
    if getattr(tokenizer, "_deepseek_v4_encoding_installed", False):
        return tokenizer

    def _apply(conversation, tools=None, tokenize=False, **kwargs):
        prompt = apply_chat_template(conversation, tools=tools, **kwargs)
        if tokenize:
            return tokenizer.encode(prompt)
        return prompt

    tokenizer.apply_chat_template = _apply
    tokenizer._deepseek_v4_encoding_installed = True
    logger.info("[deepseek_v4] installed programmatic chat template encoder")
    return tokenizer
