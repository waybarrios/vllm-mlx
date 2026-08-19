# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 prompt encoder.

DeepSeek-V4 has no Jinja chat template — the prompt is built programmatically —
so these tests assert against prompts frozen from the reference implementation
shipped with the model weights (see ``deepseek_v4_golden_prompts.py``). A silent
divergence here would not raise anywhere; it would just degrade generation.
"""

import json

import pytest

from vllm_mlx.utils.deepseek_v4_encoding import (
    BOS_TOKEN,
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    apply_chat_template,
    encode_arguments_to_dsml,
    encode_messages,
    install,
    merge_tool_messages,
    resolve_thinking,
)

from .deepseek_v4_golden_prompts import GOLDEN_PROMPTS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.mark.parametrize("name", sorted(GOLDEN_PROMPTS))
def test_matches_reference_implementation(name):
    """Our encoder reproduces the reference byte for byte."""
    messages_json, kwargs, expected = GOLDEN_PROMPTS[name]
    assert encode_messages(json.loads(messages_json), **kwargs) == expected


class TestPromptShape:
    def test_thinking_mode_opens_reasoning(self):
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}], thinking_mode="thinking"
        )
        assert prompt.endswith(THINKING_START_TOKEN)
        assert prompt.startswith(BOS_TOKEN)

    def test_chat_mode_closes_reasoning(self):
        """Chat mode pre-closes </think> so the model skips reasoning."""
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}], thinking_mode="chat"
        )
        assert prompt.endswith(THINKING_END_TOKEN)

    def test_system_message_is_bare_text(self):
        """There is no system wrapper token — content follows BOS directly."""
        prompt = encode_messages(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Hi"}],
            thinking_mode="thinking",
        )
        assert prompt.startswith(BOS_TOKEN + "SYS<｜User｜>Hi")

    def test_bos_can_be_suppressed(self):
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}],
            thinking_mode="thinking",
            add_default_bos_token=False,
        )
        assert not prompt.startswith(BOS_TOKEN)

    def test_invalid_thinking_mode_rejected(self):
        with pytest.raises(ValueError, match="thinking_mode"):
            encode_messages([{"role": "user", "content": "Hi"}], thinking_mode="nope")


class TestReasoningEffort:
    @pytest.mark.parametrize(
        "effort,marker",
        [("high", "Absolute maximum"), ("max", "Beyond maximum")],
    )
    def test_prefix_is_prepended(self, effort, marker):
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )
        assert marker in prompt
        # The prefix sits after BOS but before the conversation.
        assert prompt.index(marker) < prompt.index("<｜User｜>")

    def test_low_adds_nothing(self):
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}],
            thinking_mode="thinking",
            reasoning_effort="low",
        )
        assert "Reasoning Effort" not in prompt

    def test_no_prefix_in_chat_mode(self):
        """Effort only shapes reasoning, and chat mode has none."""
        prompt = encode_messages(
            [{"role": "user", "content": "Hi"}],
            thinking_mode="chat",
            reasoning_effort="max",
        )
        assert "Reasoning Effort" not in prompt

    def test_unknown_effort_rejected_by_encoder(self):
        with pytest.raises(ValueError, match="reasoning effort"):
            encode_messages(
                [{"role": "user", "content": "Hi"}],
                thinking_mode="thinking",
                reasoning_effort="turbo",
            )


class TestResolveThinking:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, ("thinking", None)),
            ({"enable_thinking": False}, ("chat", None)),
            ({"enable_thinking": True}, ("thinking", None)),
            ({"reasoning_effort": "none"}, ("chat", None)),
            ({"reasoning_effort": "low"}, ("thinking", "low")),
            ({"reasoning_effort": "medium"}, ("thinking", "high")),
            ({"reasoning_effort": "high"}, ("thinking", "high")),
            ({"reasoning_effort": "max"}, ("thinking", "max")),
            ({"reasoning_effort": "xhigh"}, ("thinking", "max")),
            ({"thinking_mode": "chat"}, ("chat", None)),
            ({"enable_thinking": False, "reasoning_effort": "max"}, ("chat", None)),
        ],
    )
    def test_mapping(self, kwargs, expected):
        assert resolve_thinking(**kwargs) == expected

    def test_unknown_effort_falls_back_to_high(self):
        """A typo must not silently disable reasoning."""
        assert resolve_thinking(reasoning_effort="turbo") == ("thinking", "high")

    def test_invalid_thinking_mode_rejected(self):
        with pytest.raises(ValueError, match="thinking_mode"):
            resolve_thinking(thinking_mode="bogus")


class TestTools:
    def test_schema_lands_in_system_message(self):
        prompt = apply_chat_template(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Hi"}],
            tools=TOOLS,
        )
        assert "## Tools" in prompt
        assert "get_weather" in prompt
        assert prompt.index("## Tools") < prompt.index("<｜User｜>")

    def test_system_message_synthesised_when_absent(self):
        prompt = apply_chat_template([{"role": "user", "content": "Hi"}], tools=TOOLS)
        assert "## Tools" in prompt

    def test_existing_declaration_wins(self):
        """A conversation that already declares tools is left alone."""
        conversation = [
            {"role": "system", "content": "SYS", "tools": TOOLS},
            {"role": "user", "content": "Hi"},
        ]
        assert apply_chat_template(conversation) == apply_chat_template(
            conversation, tools=TOOLS
        )

    def test_tools_keep_reasoning_history(self):
        """drop_thinking is forced off when tools are in play.

        The model has to see why it made the earlier calls, so the reasoning
        that produced them must survive into the next turn.
        """
        messages = [
            {"role": "system", "content": "S", "tools": TOOLS},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "BECAUSE"},
            {"role": "user", "content": "Q2"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
        assert "BECAUSE" in prompt

    def test_reasoning_dropped_without_tools(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "STALE"},
            {"role": "user", "content": "Q2"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
        assert "STALE" not in prompt


class TestToolResults:
    def test_tool_role_becomes_user_block(self):
        """V4 has no tool role — results ride inside the user turn."""
        merged = merge_tool_messages(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "c1", "content": "RESULT"},
            ]
        )
        assert not any(m["role"] == "tool" for m in merged)
        assert merged[-1]["role"] == "user"
        assert merged[-1]["content_blocks"][0]["content"] == "RESULT"

    def test_result_rendered_as_tool_result_tag(self):
        prompt = encode_messages(
            [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "42"},
            ],
            thinking_mode="thinking",
        )
        assert "<tool_result>42</tool_result>" in prompt

    def test_arguments_accepted_as_decoded_mapping(self):
        """``api/utils.py`` decodes ``arguments`` in place for native format.

        Re-decoding an already-decoded mapping used to collapse every parameter
        into a single bogus ``arguments`` entry, so the model saw a malformed
        call in its own history.
        """
        as_string = encode_arguments_to_dsml(
            {"name": "f", "arguments": json.dumps({"city": "Prague", "days": 3})}
        )
        as_mapping = encode_arguments_to_dsml(
            {"name": "f", "arguments": {"city": "Prague", "days": 3}}
        )
        assert as_mapping == as_string
        assert 'name="city" string="true"' in as_mapping
        assert 'name="days" string="false"' in as_mapping

    def test_unparseable_arguments_are_not_dropped(self):
        rendered = encode_arguments_to_dsml({"name": "f", "arguments": "not json"})
        assert 'name="arguments"' in rendered
        assert "not json" in rendered

    def test_results_sorted_by_call_order(self):
        """Clients may answer out of order; the model expects call order."""
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "second",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "second", "content": "SECOND"},
            {"role": "tool", "tool_call_id": "first", "content": "FIRST"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking")
        assert prompt.index("FIRST") < prompt.index("SECOND")


class TestInstall:
    class _FakeTokenizer:
        def __init__(self):
            self.chat_template = None

        def encode(self, text):
            return [len(text)]

    def test_overrides_apply_chat_template(self):
        tokenizer = self._FakeTokenizer()
        install(tokenizer)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
        assert prompt.startswith(BOS_TOKEN)
        assert prompt.endswith(THINKING_START_TOKEN)

    def test_is_idempotent(self):
        tokenizer = self._FakeTokenizer()
        install(tokenizer)
        first = tokenizer.apply_chat_template
        install(tokenizer)
        assert tokenizer.apply_chat_template is first

    def test_tokenize_returns_ids(self):
        tokenizer = self._FakeTokenizer()
        install(tokenizer)
        assert tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}], tokenize=True
        ) == [len(tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}]))]

    def test_tolerates_unknown_kwargs(self):
        """Callers pass add_generation_prompt and friends; none may blow up."""
        tokenizer = self._FakeTokenizer()
        install(tokenizer)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
            some_future_kwarg=123,
        )
        assert isinstance(prompt, str)
