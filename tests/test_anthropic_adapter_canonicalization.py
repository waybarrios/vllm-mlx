# SPDX-License-Identifier: Apache-2.0
"""The Anthropic adapter must canonicalize system prompts the shared way.

`anthropic_to_openai()` used to carry its own unanchored `re.sub` for the
billing header. It runs *before* `canonicalize_system_messages()`, so anything
it removed was gone by the time the shared, line-anchored stripper saw the
messages — including the middle of a user's own sentence. These tests pin both
stages: what the adapter alone does, and what the prepared-message path
produces end to end.
"""

import pytest

from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.prompt_canonicalize import canonicalize_system_messages

HEADER = "x-anthropic-billing-header: cch=9f2a1b;"
BODY = "You are helpful.\nBe concise."


def _system_text(system) -> str:
    request = AnthropicRequest(
        model="test-model",
        max_tokens=16,
        system=system,
        messages=[{"role": "user", "content": "hi"}],
    )
    converted = anthropic_to_openai(request)
    systems = [m for m in converted.messages if m.role == "system"]
    assert len(systems) == 1, f"expected one system message, got {len(systems)}"
    return systems[0].content


def _two_stage(system) -> str:
    """Adapter output after the shared pass the server applies next."""
    text = _system_text(system)
    canonicalized = canonicalize_system_messages([{"role": "system", "content": text}])
    return canonicalized[0]["content"]


class TestStandaloneHeaderIsRemoved:
    @pytest.mark.parametrize(
        "header",
        [
            "x-anthropic-billing-header: cch=9f2a1b;",
            "X-Anthropic-Billing-Header: cch=9f2a1b;",
            "X-ANTHROPIC-BILLING-HEADER: cch=9f2a1b;",
        ],
        ids=["lower", "mixed", "upper"],
    )
    def test_case_insensitive_on_its_own_line(self, header):
        text = _system_text(f"You are helpful.\n{header}\nBe concise.")
        assert "cch=" not in text, f"volatile hash survived: {text!r}"
        assert "You are helpful." in text
        assert "Be concise." in text

    @pytest.mark.parametrize(
        "system",
        [HEADER, [{"type": "text", "text": HEADER}]],
        ids=["string", "blocks"],
    )
    def test_header_as_the_entire_prompt(self, system):
        assert _system_text(system) == ""
        assert _two_stage(system) == ""

    def test_trailing_header_without_newline(self):
        text = _system_text(f"{BODY}\n{HEADER}")
        assert "cch=" not in text
        assert "Be concise." in text


class TestUserTextIsPreserved:
    """The regression this replaces: an unanchored strip ate real prompt text."""

    def test_mid_line_mention_survives(self):
        prompt = "Explain what x-anthropic-billing-header: means in HTTP terms."
        assert _system_text(prompt) == prompt

    def test_mid_line_mention_survives_the_second_pass_too(self):
        prompt = "Explain what x-anthropic-billing-header: means in HTTP terms."
        assert _two_stage(prompt) == prompt

    @pytest.mark.parametrize(
        "prompt",
        [
            BODY,
            "Timestamps look like 2026-05-10T13:42:18.123Z in our logs.",
            "Use the header: Authorization: Bearer <token> when calling the API.",
            "Discuss anthropic billing headers in general.",
        ],
        ids=["plain", "timestamp", "other-header", "prose"],
    )
    def test_unrelated_content_is_untouched(self, prompt):
        assert _system_text(prompt) == prompt
        assert _two_stage(prompt) == prompt


class TestBothStagesAgree:
    """Adapter and shared pass must not disagree about what to remove."""

    @pytest.mark.parametrize(
        "prompt",
        [
            f"You are helpful.\n{HEADER}\nBe concise.",
            "You are helpful.\nX-Anthropic-Billing-Header: cch=abc;\nBe concise.",
            "Explain what x-anthropic-billing-header: means in HTTP terms.",
            BODY,
        ],
        ids=["lower", "mixed", "mid-line", "clean"],
    )
    def test_second_pass_is_a_no_op(self, prompt):
        """Whatever the adapter returns is already canonical."""
        once = _system_text(prompt)
        twice = canonicalize_system_messages([{"role": "system", "content": once}])[0][
            "content"
        ]
        assert twice == once, f"shared pass still changed {once!r} -> {twice!r}"
