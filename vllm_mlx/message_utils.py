# SPDX-License-Identifier: Apache-2.0
"""
Shared message normalization utilities.

Provides ``_normalize_messages()`` which maps non-standard roles, merges
consecutive same-role messages, and hoists system messages to position [0].
Used by both SimpleEngine and BatchedEngine before ``apply_chat_template``.
"""

import logging

logger = logging.getLogger(__name__)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message roles and merge consecutive same-role messages.

    1. Maps non-standard roles to standard ones (e.g. ``developer`` -> ``system``).
    2. Merges consecutive same-role messages to satisfy chat template constraints
       (Qwen 3.5, Llama, etc. require alternating roles).

    Only merges when both messages have string content. Messages with list
    content (multimodal) are left as-is to preserve image/video attachments.
    """
    _ROLE_MAP = {"developer": "system"}

    if not messages:
        return messages

    merged = [messages[0].copy()]
    if merged[0]["role"] in _ROLE_MAP:
        merged[0]["role"] = _ROLE_MAP[merged[0]["role"]]
    for msg in messages[1:]:
        prev = merged[-1]
        role = _ROLE_MAP.get(msg["role"], msg["role"])
        if (
            role == prev["role"]
            and isinstance(prev.get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            prev["content"] = prev["content"] + "\n\n" + msg["content"]
            logger.debug(
                f"Merged consecutive {role} messages "
                f"({len(prev['content'])} chars total)"
            )
        else:
            copy = msg.copy()
            copy["role"] = role
            merged.append(copy)

    # Hoist system messages to position [0] and merge them.
    # Many CLIs (OpenCode, Qwen Code, Kilo) send system messages mid-conversation;
    # the Qwen 3.5 chat template rejects any system message not at position [0].
    system_msgs = [m for m in merged if m["role"] == "system"]
    non_system = [m for m in merged if m["role"] != "system"]
    if system_msgs and (len(system_msgs) > 1 or merged[0]["role"] != "system"):
        # Combine all system message content (string only) into one
        parts = []
        for m in system_msgs:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                # Multimodal system message — extract text parts
                for part in c:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
        if parts:
            combined_system = {"role": "system", "content": "\n\n".join(parts)}
            merged = [combined_system] + non_system
            logger.info(
                f"Hoisted {len(system_msgs)} system message(s) to position [0] "
                f"({len(combined_system['content'])} chars)"
            )
        else:
            # No string content — just move the first system msg to front
            merged = system_msgs[:1] + non_system
            logger.info("Hoisted system message to position [0]")

    merged_count = len(messages) - len(merged)
    if merged_count:
        logger.info(f"Normalized messages: merged {len(messages)} -> {len(merged)}")

    return merged
