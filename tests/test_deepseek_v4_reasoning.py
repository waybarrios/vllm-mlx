# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 reasoning parser.

V4 shares ``<think>``/``</think>`` with R1, so the interesting cases are the
ones R1 does not have: the prompt closes on ``<think>`` (leaving the opening tag
out of the output), and a tool call implicitly ends the reasoning block.
"""

import pytest

from vllm_mlx.reasoning import get_parser, list_parsers
from vllm_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser

D = "｜DSML｜"
TOOL_CALLS_START = f"<{D}tool_calls>"

TOOL_BLOCK = (
    f"{TOOL_CALLS_START}\n"
    f'<{D}invoke name="f">\n'
    f'<{D}parameter name="a" string="true">1</{D}parameter>\n'
    f"</{D}invoke>\n"
    f"</{D}tool_calls>"
)


@pytest.fixture
def parser():
    p = DeepSeekV4ReasoningParser()
    p.reset_state()
    return p


class TestRegistration:
    def test_registered(self):
        assert "deepseek_v4" in list_parsers()

    def test_resolves_to_v4_parser(self):
        assert get_parser("deepseek_v4") is DeepSeekV4ReasoningParser


class TestExtraction:
    def test_implicit_opening_tag(self, parser):
        """The encoder ends the prompt with <think>, so output starts inside it."""
        reasoning, content = parser.extract_reasoning("weighing it</think>Answer.")
        assert reasoning == "weighing it"
        assert content == "Answer."

    def test_explicit_tags(self, parser):
        reasoning, content = parser.extract_reasoning(
            "<think>weighing it</think>Answer."
        )
        assert reasoning == "weighing it"
        assert content == "Answer."

    def test_no_tags_is_pure_content(self, parser):
        reasoning, content = parser.extract_reasoning("Just an answer.")
        assert reasoning is None
        assert content == "Just an answer."


class TestToolCallInteraction:
    def test_tool_call_closes_unterminated_reasoning(self, parser):
        """Without this the whole DSML payload would be swallowed as reasoning."""
        reasoning, content = parser.extract_reasoning(f"I should call it{TOOL_BLOCK}")
        assert reasoning == "I should call it"
        assert content.startswith(TOOL_CALLS_START)

    def test_markup_is_left_for_the_tool_parser(self, parser):
        _, content = parser.extract_reasoning(f"thinking{TOOL_BLOCK}")
        assert TOOL_BLOCK in content

    def test_explicit_open_tag_with_tool_call(self, parser):
        reasoning, content = parser.extract_reasoning(f"<think>thinking{TOOL_BLOCK}")
        assert reasoning == "thinking"
        assert content.startswith(TOOL_CALLS_START)

    def test_closed_reasoning_before_tool_call(self, parser):
        """When </think> came first it wins — the split is already unambiguous."""
        reasoning, content = parser.extract_reasoning(
            f"thinking</think>Calling.\n\n{TOOL_BLOCK}"
        )
        assert reasoning == "thinking"
        assert content.startswith("Calling.")
        assert TOOL_CALLS_START in content


class TestStreaming:
    @staticmethod
    def _stream(text, chunk):
        parser = DeepSeekV4ReasoningParser()
        parser.reset_state()
        previous, reasoning, content = "", [], []
        for i in range(0, len(text), chunk):
            delta = text[i : i + chunk]
            current = previous + delta
            message = parser.extract_reasoning_streaming(previous, current, delta)
            if message is not None:
                if message.reasoning:
                    reasoning.append(message.reasoning)
                if message.content:
                    content.append(message.content)
            previous = current
        return "".join(reasoning), "".join(content)

    @pytest.mark.parametrize("chunk", [1, 3, 7, 16, 64])
    def test_split_matches_non_streaming(self, chunk):
        text = "weighing it</think>Answer."
        reasoning, content = self._stream(text, chunk)
        expected_reasoning, expected_content = (
            DeepSeekV4ReasoningParser().extract_reasoning(text)
        )
        assert reasoning.strip() == (expected_reasoning or "")
        assert content.strip() == (expected_content or "")

    @pytest.mark.parametrize("chunk", [1, 3, 7, 16, 64])
    def test_tool_call_ends_reasoning_mid_stream(self, chunk):
        reasoning, content = self._stream(f"deciding{TOOL_BLOCK}", chunk)
        assert reasoning.strip() == "deciding"
        assert TOOL_CALLS_START in content
        assert D not in reasoning

    @pytest.mark.parametrize("chunk", [1, 3, 7, 16, 64])
    def test_nothing_is_lost(self, chunk):
        """Every character must land in exactly one of the two channels."""
        text = f"deciding{TOOL_BLOCK}"
        reasoning, content = self._stream(text, chunk)
        assert len(reasoning) + len(content) == len(text)

    @pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 6, 7])
    def test_split_think_tag_does_not_leak(self, chunk):
        """A split ``</think>`` must not leak fragments into reasoning.

        V4 has its own ids for the think tags, so they normally arrive whole;
        this pins the behaviour for any detokenizer that splits them, which is
        the same failure mode the multi-token DSML marker hits for real.
        """
        reasoning, content = self._stream("weighing it</think>Answer.", chunk)
        assert reasoning.strip() == "weighing it"
        assert content.strip() == "Answer."
        for fragment in ("</", "<", "think", ">"):
            assert fragment not in reasoning

    def test_finalize_releases_ordinary_partial_marker(self, parser):
        """A final ordinary ``<`` is text, not a marker to drop at EOS."""
        message = parser.extract_reasoning_streaming("", "2 <", "2 <")
        final = parser.finalize_stream()

        assert message is not None
        assert message.reasoning == "2 "
        assert final is not None
        assert final.reasoning == "<"
