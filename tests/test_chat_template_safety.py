# SPDX-License-Identifier: Apache-2.0
"""Tests for chat-template normalization and cache-prefix probing."""

import pytest

from vllm_mlx.engine.chat_template_safety import (
    build_system_prompt_cache_prefix,
    normalize_messages_for_chat_template,
)


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


def test_normalization_deep_copies_json_without_dropping_non_dict_items():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        None,
        7,
        "literal",
        [False, {"nested": [1]}],
    ]

    normalized = normalize_messages_for_chat_template(messages)

    assert normalized == messages
    assert normalized is not messages
    normalized[0]["content"][0]["text"] = "changed"
    normalized[4][1]["nested"].append(2)
    assert messages[0]["content"][0]["text"] == "hello"
    assert messages[4] == [False, {"nested": [1]}]


def test_normalization_converts_model_messages_and_stringifies_unknown_objects():
    class ModelMessage:
        def model_dump(self, *, exclude_none):
            assert exclude_none is True
            return {"role": "user", "content": "modern", "unused": None}

    class LegacyMessage:
        def dict(self):
            return {"role": "user", "content": "legacy", "unused": None}

    class UnknownMessage:
        def __str__(self):
            return "unknown message"

    assert normalize_messages_for_chat_template(
        [ModelMessage(), LegacyMessage(), UnknownMessage()]
    ) == [
        {"role": "user", "content": "modern"},
        {"role": "user", "content": "legacy"},
        "unknown message",
    ]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"city":"Bogota"}', {"city": "Bogota"}),
        ("[1,2]", {"value": [1, 2]}),
        ("null", {"value": None}),
        ("true", {"value": True}),
        ("7", {"value": 7}),
        ('"hello"', {"value": "hello"}),
        ("{broken", {"value": "{broken"}),
        ({"already": [1]}, {"already": [1]}),
        (None, None),
    ],
)
def test_normalization_preserves_tool_argument_mapping_policy(arguments, expected):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"arguments": arguments}}],
        }
    ]

    normalized = normalize_messages_for_chat_template(messages)

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == expected
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == arguments


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>reason<tool_call>x", "<think>reason</think><tool_call>x"),
        ("<think>reason</think><tool_call>x", "<think>reason</think><tool_call>x"),
        ("<tool_call>x<think>reason", "<tool_call>x<think>reason</think>"),
        ("<think>reason", "<think>reason"),
        (None, None),
    ],
)
def test_normalization_preserves_dangling_think_repair(content, expected):
    messages = [{"role": "assistant", "content": content}]

    assert normalize_messages_for_chat_template(messages)[0]["content"] == expected
    assert messages[0]["content"] == content
