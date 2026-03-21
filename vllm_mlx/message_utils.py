# SPDX-License-Identifier: Apache-2.0
"""
Message normalization utilities.

ALL engine paths that call ``apply_chat_template`` MUST normalize messages
first.  Qwen 3.5 templates reject:

  - ``developer`` role (must be mapped to ``system``)
  - system messages anywhere other than position [0]
  - consecutive messages with the same role

Without normalization, multi-turn agent sessions crash on template
application.
"""

import logging

logger = logging.getLogger(__name__)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize chat messages for safe chat-template application.

    Performs three transformations in order:

    1. **Role mapping** -- ``developer`` role is mapped to ``system``.
    2. **System hoisting** -- all system messages are moved to position [0],
       concatenated with ``\\n\\n`` in their original relative order.
    3. **Consecutive merge** -- adjacent messages with the same role are
       merged (content joined with ``\\n\\n``).

    Args:
        messages: List of chat message dicts (each with at least ``role``
            and ``content`` keys).

    Returns:
        A new list of normalized message dicts.  The input list is not
        modified.
    """
    if not messages:
        return messages

    # --- 1. Map developer -> system ---
    mapped: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "developer":
            mapped.append({**msg, "role": "system"})
        else:
            mapped.append(msg)

    # --- 2. Hoist system messages to position [0] ---
    system_parts: list[str] = []
    non_system: list[dict] = []
    for msg in mapped:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
            elif isinstance(content, list):
                # Handle structured content (list of parts)
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                text = "\n\n".join(p for p in text_parts if p)
                if text:
                    system_parts.append(text)
        else:
            non_system.append(msg)

    hoisted: list[dict] = []
    if system_parts:
        hoisted.append({"role": "system", "content": "\n\n".join(system_parts)})
    hoisted.extend(non_system)

    # --- 3. Merge consecutive same-role messages ---
    merged: list[dict] = []
    for msg in hoisted:
        if merged and merged[-1].get("role") == msg.get("role"):
            prev_content = merged[-1].get("content", "")
            cur_content = msg.get("content", "")
            # Only merge when both are plain strings
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                merged[-1] = {
                    **merged[-1],
                    "content": prev_content + "\n\n" + cur_content,
                }
            else:
                # Cannot safely merge structured content; keep separate
                merged.append(msg)
        else:
            merged.append(msg)

    if len(merged) != len(messages):
        logger.debug(
            "Normalized messages: %d -> %d (hoisted system, merged consecutive)",
            len(messages),
            len(merged),
        )

    return merged
