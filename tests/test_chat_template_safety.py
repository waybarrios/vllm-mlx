# SPDX-License-Identifier: Apache-2.0
"""Tests for chat-template normalization and cache-prefix probing."""

from vllm_mlx.engine.chat_template_safety import build_system_prompt_cache_prefix


class _TemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        rendered = []
        for message in messages:
            for tool_call in message.get("tool_calls", []):
                arguments = tool_call["function"]["arguments"]
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be a mapping")
            rendered.append(
                f"<{message['role']}>{message.get('content') or ''}"
                f"</{message['role']}>"
            )
        return "".join(rendered) + "<assistant>"


def test_system_cache_probe_uses_normalized_system_messages_after_tool_turn():
    messages = [
        {"role": "system", "content": "Cache stable policy."},
        {"role": "user", "content": "Check Bogota weather."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Bogota"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "Sunny",
        },
    ]

    prefix = build_system_prompt_cache_prefix(
        _TemplateTokenizer(),
        messages,
        template_kwargs={"tokenize": False, "add_generation_prompt": True},
    )

    assert prefix == "<system>Cache stable policy.</system><user>"
