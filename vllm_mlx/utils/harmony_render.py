# SPDX-License-Identifier: Apache-2.0
"""Harmony-format prompt rendering for GPT-OSS via ``openai-harmony``.

GPT-OSS models are trained with OpenAI's harmony wire format (channeled
``<|start|>assistant<|channel|>commentary ...<|call|>`` tool calls,
``<|start|>functions.X to=assistant<|channel|>commentary<|message|>...``
tool results, etc.). Rendering harmony correctly from OpenAI-style chat
messages is delicate: prior assistant ``tool_calls`` must arrive at the
template as structural objects, not the bracket-text fallback that
``api.utils.extract_multimodal_content()`` produces for non-native parsers.

This module bypasses the Jinja chat template entirely for harmony-active
engines: it converts the OpenAI-format ``messages`` (plus ``tools``) to an
``openai_harmony.Conversation`` and asks the library — the canonical
renderer maintained by OpenAI — to serialize it. That sidesteps both the
text-flattening upstream and any template-vs-training-format drift.

The library is an optional dependency. ``HAS_HARMONY`` reflects import
success so the rest of the engine can fall back to ``apply_chat_template``
when the package is absent.

See https://github.com/waybarrios/vllm-mlx/issues/568 for the original
report and the patch shape Thump604 outlined.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import openai_harmony as _oh

    HAS_HARMONY = True
except ImportError:
    _oh = None
    HAS_HARMONY = False


def is_harmony_parser_name(parser_name: str | None) -> bool:
    """Return True when the active --tool-call-parser is a harmony alias.

    ``HarmonyToolParser`` registers under both ``"harmony"`` and ``"gpt-oss"``.
    """
    return parser_name in {"harmony", "gpt-oss"}


def _build_tools(tools: list[dict] | None) -> list[Any] | None:
    if not tools or _oh is None:
        return None
    tool_descs: list[Any] = []
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name")
        if not name:
            continue
        tool_descs.append(
            _oh.ToolDescription.new(
                name=name,
                description=fn.get("description") or "",
                parameters=fn.get("parameters") or {},
            )
        )
    return tool_descs or None


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI content (str | list[dict]) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _convert_message(msg: dict) -> list[Any]:
    """Convert one OpenAI-format message to one or more ``openai_harmony.Message``.

    A single assistant turn can carry multiple tool_calls; harmony represents
    each as its own commentary-channel message addressed to ``functions.X``.
    Prior reasoning lives in an analysis-channel message that precedes the
    tool calls.
    """
    if _oh is None:
        return []
    role = msg.get("role", "user")
    content_text = _content_to_text(msg.get("content"))
    out: list[Any] = []

    if role == "system":
        out.append(_oh.Message.from_role_and_content(_oh.Role.SYSTEM, content_text))
    elif role == "user":
        out.append(_oh.Message.from_role_and_content(_oh.Role.USER, content_text))
    elif role == "tool":
        # The tool name lives on a prior assistant tool_call; the caller is
        # expected to thread it through ``tool_call_id``-to-name mapping
        # before this conversion runs. The OpenAI schema doesn't carry the
        # function name on tool messages directly, so we leave the name slot
        # as the conventional ``functions.unknown`` if it isn't injected
        # under the ``name`` key.
        tool_name = msg.get("name") or "functions.unknown"
        if not tool_name.startswith("functions."):
            tool_name = f"functions.{tool_name}"
        out.append(
            _oh.Message(
                author=_oh.Author.new(_oh.Role.TOOL, name=tool_name),
                content=[_oh.TextContent(text=content_text)],
                channel="commentary",
                recipient="assistant",
            )
        )
    elif role == "assistant":
        thinking = msg.get("thinking") or msg.get("reasoning_content")
        tool_calls = msg.get("tool_calls") or []
        # Reasoning (analysis channel) — only renders when there are tool_calls
        # to follow; the harmony chat template otherwise drops it for prior
        # turns (matches gpt-oss training).
        if thinking and tool_calls:
            out.append(
                _oh.Message(
                    author=_oh.Author.new(_oh.Role.ASSISTANT, name=None),
                    content=[_oh.TextContent(text=str(thinking))],
                    channel="analysis",
                )
            )
        # User-visible final-channel content
        if content_text and not tool_calls:
            out.append(
                _oh.Message(
                    author=_oh.Author.new(_oh.Role.ASSISTANT, name=None),
                    content=[_oh.TextContent(text=content_text)],
                    channel="final",
                )
            )
        # Tool calls
        for tc in tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name", "unknown")
            args = fn.get("arguments")
            if isinstance(args, dict):
                args_text = json.dumps(args, ensure_ascii=False)
            elif args is None:
                args_text = "{}"
            else:
                args_text = str(args)
            out.append(
                _oh.Message(
                    author=_oh.Author.new(_oh.Role.ASSISTANT, name=None),
                    content=[_oh.TextContent(text=args_text)],
                    channel="commentary",
                    recipient=f"functions.{name}",
                    content_type="json",
                )
            )
    elif role == "developer":
        out.append(_oh.Message.from_role_and_content(_oh.Role.DEVELOPER, content_text))
    # Any other role is silently dropped (matches existing chat-template behavior)
    return out


def _resolve_tool_names(messages: list[dict]) -> list[dict]:
    """Stamp ``name=functions.X`` on each ``role=tool`` message by tracing back
    the most recent assistant ``tool_call_id`` -> function name."""
    by_call_id: dict[str, str] = {}
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                if tc_id and name:
                    by_call_id[tc_id] = name
            out.append(m)
            continue
        if m.get("role") == "tool":
            new_m = dict(m)
            if "name" not in new_m and (tc_id := new_m.get("tool_call_id")):
                name = by_call_id.get(tc_id)
                if name:
                    new_m["name"] = name
            out.append(new_m)
            continue
        out.append(m)
    return out


def render_messages(
    messages: list[dict],
    tools: list[dict] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Render OpenAI-format messages as a harmony-format prompt string.

    Raises ``RuntimeError`` if ``openai-harmony`` is not importable; callers
    should pre-check with :data:`HAS_HARMONY` and fall back to
    ``tokenizer.apply_chat_template`` when False.

    Args:
        messages: OpenAI chat-completions messages.
        tools: OpenAI-format tools list (each item ``{"type":"function","function":{...}}``).
        reasoning_effort: ``"low"``, ``"medium"``, or ``"high"``. Defaults to medium.

    Returns:
        Decoded harmony prompt with the trailing ``<|start|>assistant``
        marker ready for the model to begin generation.
    """
    if not HAS_HARMONY:
        raise RuntimeError(
            "openai-harmony is not installed. `pip install openai-harmony` or "
            "fall back to tokenizer.apply_chat_template."
        )

    resolved = _resolve_tool_names(messages)

    # Pull system + developer messages to the head; gpt-oss expects
    # ``<|start|>system|...|<|end|><|start|>developer|...|<|end|>`` before
    # any user/assistant turn.
    system_msgs: list[dict] = []
    developer_msgs: list[dict] = []
    other_msgs: list[dict] = []
    for m in resolved:
        if not isinstance(m, dict):
            other_msgs.append(m)
            continue
        role = m.get("role")
        if role == "system":
            system_msgs.append(m)
        elif role == "developer":
            developer_msgs.append(m)
        else:
            other_msgs.append(m)

    h_messages: list[Any] = []
    tool_descs = _build_tools(tools)

    # 1. System block (inject default if caller didn't provide one).
    if system_msgs:
        for m in system_msgs:
            h_messages.extend(_convert_message(m))
    else:
        sys_content = _oh.SystemContent.new()
        if reasoning_effort:
            try:
                level = getattr(_oh.ReasoningEffort, reasoning_effort.upper())
                sys_content = sys_content.with_reasoning_effort(level)
            except Exception:  # noqa: BLE001
                pass
        h_messages.append(
            _oh.Message.from_role_and_content(_oh.Role.SYSTEM, sys_content)
        )

    # 2. Developer block (tool schema). If caller passed an explicit
    #    developer message, render that; otherwise synthesize from tools.
    if developer_msgs:
        for m in developer_msgs:
            h_messages.extend(_convert_message(m))
    elif tool_descs:
        dev_content = _oh.DeveloperContent.new().with_function_tools(tool_descs)
        h_messages.append(
            _oh.Message.from_role_and_content(_oh.Role.DEVELOPER, dev_content)
        )

    # 3. Everything else, in original order.
    for m in other_msgs:
        if isinstance(m, dict):
            h_messages.extend(_convert_message(m))

    conv = _oh.Conversation.from_messages(h_messages)
    enc = _oh.load_harmony_encoding(_oh.HarmonyEncodingName.HARMONY_GPT_OSS)
    token_ids = enc.render_conversation_for_completion(
        conv, next_turn_role=_oh.Role.ASSISTANT
    )
    return enc.decode(token_ids)
