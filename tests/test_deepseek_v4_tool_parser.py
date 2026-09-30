# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4 DSML tool call parser."""

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.deepseek_v4_tool_parser import (
    TOOL_CALLS_END,
    TOOL_CALLS_START,
    DeepSeekV4ToolParser,
)

D = "｜DSML｜"


def invoke(name: str, *params: str) -> str:
    body = "\n".join(params)
    return f'<{D}invoke name="{name}">\n{body}\n</{D}invoke>'


def param(name: str, value: str, *, is_str: bool) -> str:
    flag = "true" if is_str else "false"
    return f'<{D}parameter name="{name}" string="{flag}">{value}</{D}parameter>'


def block(*invokes: str) -> str:
    return f"{TOOL_CALLS_START}\n" + "\n".join(invokes) + f"\n{TOOL_CALLS_END}"


@pytest.fixture
def parser():
    p = DeepSeekV4ToolParser()
    p.reset()
    return p


def args_of(result, index=0):
    return json.loads(result.tool_calls[index]["arguments"])


class TestRegistration:
    @pytest.mark.parametrize("name", ["deepseek_v4", "dsml"])
    def test_registered_under(self, name):
        assert ToolParserManager.get_tool_parser(name) is DeepSeekV4ToolParser

    def test_distinct_from_v3_parser(self):
        """V3/R1 use <｜tool▁calls▁begin｜> + fenced JSON; V4 shares nothing."""
        v3 = ToolParserManager.get_tool_parser("deepseek")
        assert v3 is not DeepSeekV4ToolParser

    def test_declares_native_tool_format(self):
        """The encoder consumes role="tool" and tool_calls directly.

        Declaring otherwise makes the server flatten them into
        "[Tool Result (id)]: ..." and "[Calling tool: name(...)]" before the
        encoder ever runs, so the model would see a shape it was never trained
        on instead of <tool_result> blocks and DSML.
        """
        assert DeepSeekV4ToolParser.supports_native_format() is True


class TestExtraction:
    def test_single_call(self, parser):
        text = "Let me check.\n\n" + block(
            invoke("get_weather", param("city", "Prague", is_str=True))
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Let me check."
        assert result.tool_calls[0]["name"] == "get_weather"
        assert args_of(result) == {"city": "Prague"}

    def test_parallel_calls(self, parser):
        text = block(
            invoke("search", param("q", "mlx", is_str=True)),
            invoke("get_weather", param("city", "Brno", is_str=True)),
        )
        result = parser.extract_tool_calls(text)
        assert [tc["name"] for tc in result.tool_calls] == ["search", "get_weather"]
        assert args_of(result, 0) == {"q": "mlx"}
        assert args_of(result, 1) == {"city": "Brno"}

    def test_call_ids_are_unique(self, parser):
        text = block(
            invoke("a", param("x", "1", is_str=True)),
            invoke("b", param("x", "2", is_str=True)),
        )
        result = parser.extract_tool_calls(text)
        assert result.tool_calls[0]["id"] != result.tool_calls[1]["id"]

    def test_no_markup_is_plain_content(self, parser):
        result = parser.extract_tool_calls("Just an answer.")
        assert not result.tools_called
        assert result.content == "Just an answer."

    def test_content_after_reasoning_only(self, parser):
        """Reasoning belongs to the reasoning parser, not to content."""
        text = "deliberating</think>Checking.\n\n" + block(
            invoke("f", param("x", "1", is_str=True))
        )
        result = parser.extract_tool_calls(text)
        assert result.content == "Checking."


class TestParameterTyping:
    """``string="true"`` is verbatim, ``string="false"`` is JSON."""

    def test_string_parameter_kept_raw(self, parser):
        result = parser.extract_tool_calls(
            block(invoke("f", param("s", "42", is_str=True)))
        )
        assert args_of(result) == {"s": "42"}

    def test_non_string_parameter_decoded(self, parser):
        result = parser.extract_tool_calls(
            block(invoke("f", param("n", "42", is_str=False)))
        )
        assert args_of(result) == {"n": 42}

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("3", 3),
            ("3.5", 3.5),
            ("true", True),
            ("false", False),
            ("null", None),
            ("[1, 2]", [1, 2]),
            ('{"a": {"b": [1]}}', {"a": {"b": [1]}}),
        ],
    )
    def test_json_value_kinds(self, parser, raw, expected):
        result = parser.extract_tool_calls(
            block(invoke("f", param("v", raw, is_str=False)))
        )
        assert args_of(result) == {"v": expected}

    def test_string_value_may_contain_markup_like_text(self, parser):
        """A string parameter is why this cannot be a regex over name=value."""
        tricky = 'has "quotes" & <tags> and {"json": 1}'
        result = parser.extract_tool_calls(
            block(invoke("f", param("s", tricky, is_str=True)))
        )
        assert args_of(result) == {"s": tricky}

    def test_unparseable_json_falls_back_to_raw(self, parser):
        result = parser.extract_tool_calls(
            block(invoke("f", param("v", "{not json", is_str=False)))
        )
        assert args_of(result) == {"v": "{not json"}

    def test_multiple_parameters(self, parser):
        result = parser.extract_tool_calls(
            block(
                invoke(
                    "f",
                    param("a", "x", is_str=True),
                    param("b", "2", is_str=False),
                    param("c", "[true]", is_str=False),
                )
            )
        )
        assert args_of(result) == {"a": "x", "b": 2, "c": [True]}

    def test_call_without_parameters(self, parser):
        result = parser.extract_tool_calls(
            f"{TOOL_CALLS_START}\n"
            f'<{D}invoke name="ping">\n'
            f"</{D}invoke>\n{TOOL_CALLS_END}"
        )
        assert result.tools_called
        assert args_of(result) == {}


class TestMalformedInput:
    """Half-generated markup must degrade, never raise."""

    def test_truncated_after_start_marker(self, parser):
        result = parser.extract_tool_calls(f"text{TOOL_CALLS_START}\n")
        assert not result.tools_called
        assert result.content

    def test_truncated_mid_invoke_header(self, parser):
        result = parser.extract_tool_calls(f'{TOOL_CALLS_START}\n<{D}invoke name="f')
        assert not result.tools_called

    def test_truncated_parameter_is_dropped(self, parser):
        text = (
            f"{TOOL_CALLS_START}\n"
            f'<{D}invoke name="f">\n'
            f'<{D}parameter name="a" string="true">x</{D}parameter>\n'
            f'<{D}parameter name="b" string="true">unterminated'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert args_of(result) == {"a": "x"}

    def test_missing_end_marker_still_parses(self, parser):
        text = f"{TOOL_CALLS_START}\n" + invoke("f", param("a", "1", is_str=True))
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert args_of(result) == {"a": "1"}


class TestStreaming:
    """Deltas arrive token-sized, so every marker can straddle a boundary."""

    FULL = "Checking.\n\n" + block(
        invoke(
            "get_weather",
            param("city", "Prague", is_str=True),
            param("days", "3", is_str=False),
        ),
        invoke("search", param("q", "mlx", is_str=True)),
    )

    @staticmethod
    def _stream(text, chunk):
        parser = DeepSeekV4ToolParser()
        parser.reset()
        previous, content, tool_deltas = "", [], []
        for i in range(0, len(text), chunk):
            delta = text[i : i + chunk]
            current = previous + delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            if result:
                if "content" in result:
                    content.append(result["content"])
                if "tool_calls" in result:
                    tool_deltas.append(result["tool_calls"])
            previous = current
        return "".join(content), tool_deltas

    @pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 13, 31, 64, 128])
    def test_matches_non_streaming(self, chunk):
        content, tool_deltas = self._stream(self.FULL, chunk)
        expected = DeepSeekV4ToolParser().extract_tool_calls(self.FULL)

        assert len(tool_deltas) == 1, "tool calls must be emitted exactly once"
        streamed = [
            (tc["function"]["name"], tc["function"]["arguments"])
            for tc in tool_deltas[0]
        ]
        assert streamed == [(tc["name"], tc["arguments"]) for tc in expected.tool_calls]
        assert content.strip() == "Checking."

    @pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 13, 31, 64, 128])
    def test_markup_never_leaks_into_content(self, chunk):
        content, _ = self._stream(self.FULL, chunk)
        assert D not in content

    def test_indices_are_sequential(self):
        _, tool_deltas = self._stream(self.FULL, 5)
        assert [tc["index"] for tc in tool_deltas[0]] == [0, 1]

    def test_partial_marker_is_not_swallowed(self):
        """Text that merely looks like the marker must still be delivered."""
        parser = DeepSeekV4ToolParser()
        parser.reset()
        previous, out = "", []
        for delta in ["hello <", "｜not-dsml", " after"]:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            if result and "content" in result:
                out.append(result["content"])
            previous = current
        assert "".join(out) == "hello <｜not-dsml after"

    def test_plain_text_streams_through(self):
        content, tool_deltas = self._stream("no tools here at all", 4)
        assert content == "no tools here at all"
        assert tool_deltas == []

    def test_reset_clears_state_between_requests(self):
        parser = DeepSeekV4ToolParser()
        parser.reset()
        parser.extract_tool_calls_streaming("", "hello <", "hello <")
        parser.reset()
        result = parser.extract_tool_calls_streaming("", "plain", "plain")
        assert result == {"content": "plain"}

    @pytest.mark.parametrize("text", ["2 <", f'{TOOL_CALLS_START}\n<{D}invoke name="f'])
    def test_finalize_releases_unparsed_suffix(self, text):
        """EOS must not discard a partial marker or malformed DSML block."""
        parser = DeepSeekV4ToolParser()
        previous, content = "", []
        for delta in text:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            if result and result.get("content"):
                content.append(result["content"])
            previous = current

        result = parser.finalize_streaming(previous)
        if result and result.get("content"):
            content.append(result["content"])

        assert "".join(content) == text


class TestAutoDetection:
    def test_auto_parser_routes_dsml(self):
        from vllm_mlx.tool_parsers.auto_tool_parser import AutoToolParser

        text = block(invoke("f", param("a", "1", is_str=True)))
        result = AutoToolParser().extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "f"

    def test_auto_parser_routes_character_streaming_dsml(self):
        from vllm_mlx.tool_parsers.auto_tool_parser import AutoToolParser

        text = "Checking.\n\n" + block(invoke("f", param("a", "1", is_str=True)))
        parser = AutoToolParser()
        parser.reset()
        previous, content, calls = "", [], []
        for delta in text:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            if result:
                if result.get("content"):
                    content.append(result["content"])
                if result.get("tool_calls"):
                    calls.append(result["tool_calls"])
            previous = current

        assert "".join(content).strip() == "Checking."
        assert D not in "".join(content)
        assert len(calls) == 1
        assert calls[0][0]["function"]["name"] == "f"


class TestChainedWithReasoningParser:
    """The server feeds the reasoning parser's content to the tool parser.

    Testing the two in isolation misses everything that can go wrong between
    them, which is where the markup actually crosses over.
    """

    FULL = "deciding what to call</think>Checking.\n\n" + block(
        invoke(
            "get_weather",
            param("city", "Prague", is_str=True),
            param("days", "3", is_str=False),
        ),
        invoke("search", param("q", "mlx", is_str=True)),
    )

    @staticmethod
    def _pipeline(text, chunk):
        from vllm_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser

        reasoner = DeepSeekV4ReasoningParser()
        reasoner.reset_state()
        tools = DeepSeekV4ToolParser()
        tools.reset()

        accumulated, tool_acc = "", ""
        reasoning, content, calls = [], [], []
        for i in range(0, len(text), chunk):
            delta = text[i : i + chunk]
            previous, accumulated = accumulated, accumulated + delta
            msg = reasoner.extract_reasoning_streaming(previous, accumulated, delta)
            if msg is None:
                continue
            if msg.reasoning:
                reasoning.append(msg.reasoning)
            if not msg.content:
                continue
            prev_tool, tool_acc = tool_acc, tool_acc + msg.content
            result = tools.extract_tool_calls_streaming(
                prev_tool, tool_acc, msg.content
            )
            if result is None:
                continue
            if "tool_calls" in result:
                calls.append(result["tool_calls"])
            elif result.get("content"):
                content.append(result["content"])
        return "".join(reasoning), "".join(content), calls

    @pytest.mark.parametrize("chunk", [1, 3, 8, 25, 64])
    def test_calls_survive_the_chain(self, chunk):
        _, _, calls = self._pipeline(self.FULL, chunk)
        expected = DeepSeekV4ToolParser().extract_tool_calls(self.FULL)
        assert len(calls) == 1
        assert [
            (tc["function"]["name"], tc["function"]["arguments"]) for tc in calls[0]
        ] == [(tc["name"], tc["arguments"]) for tc in expected.tool_calls]

    @pytest.mark.parametrize("chunk", [1, 3, 8, 25, 64])
    def test_no_markup_reaches_the_client(self, chunk):
        reasoning, content, _ = self._pipeline(self.FULL, chunk)
        for channel in (reasoning, content):
            assert D not in channel
            assert "<think>" not in channel
            assert "</think>" not in channel

    @pytest.mark.parametrize("chunk", [1, 3, 8, 25, 64])
    def test_reasoning_and_content_land_in_the_right_channel(self, chunk):
        reasoning, content, _ = self._pipeline(self.FULL, chunk)
        assert reasoning.strip() == "deciding what to call"
        assert content.strip() == "Checking."


class TestTextAndCallsInOneDelta:
    """A block that opens and closes in one delta must not eat the text before it.

    Chunked arrival always worked, which is exactly why this hid: the streaming
    tests drive the parser chunk by chunk, and only a whole response arriving as
    a single delta reaches the branch that used to skip the flush. The result
    was an assistant message whose text disappeared based on nothing but how the
    model's output happened to be chunked.
    """

    SAMPLE = "Checking.\n\n" + block(
        invoke("get_weather", param("city", "Prague", is_str=True))
    )

    @staticmethod
    def _drive(deltas):
        parser = DeepSeekV4ToolParser()
        previous, content, calls = "", [], 0
        for delta in deltas:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            if result:
                if result.get("content"):
                    content.append(result["content"])
                if result.get("tool_calls"):
                    calls += len(result["tool_calls"])
            previous = current
        return "".join(content), calls

    def test_single_delta_keeps_both(self):
        content, calls = self._drive([self.SAMPLE])

        assert calls == 1
        assert "Checking." in content, (
            "text preceding the tool call was dropped because the block opened "
            "and closed in the same delta"
        )

    @pytest.mark.parametrize("chunk", [1, 3, 4, 17, 64, 4096])
    def test_result_does_not_depend_on_chunk_size(self, chunk):
        whole = self._drive([self.SAMPLE])
        chunked = self._drive(
            [self.SAMPLE[i : i + chunk] for i in range(0, len(self.SAMPLE), chunk)]
        )

        assert (
            whole == chunked
        ), f"chunk={chunk} changed the response: {whole!r} vs {chunked!r}"

    def test_text_is_not_emitted_twice(self):
        """The head is flushed once, whether it rides along or goes ahead."""
        content, _ = self._drive([self.SAMPLE])
        assert content.count("Checking.") == 1
