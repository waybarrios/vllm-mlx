# SPDX-License-Identifier: Apache-2.0
"""
Streaming chunk-boundary regression tests for Qwen3XMLToolParser.

These tests target the streaming state machine's behavior when model output
is split at arbitrary byte offsets, which is what actually happens when the
server streams tokens one at a time from MLX.

The existing test_qwen3_xml_parser.py covers complete-text extraction and
loosely checks that streaming produces "some" deltas. This module pins down
the contract: regardless of how the same model output is sliced into chunks,
the aggregated streamed tool calls must match the non-streaming reference,
and plain text that merely looks like XML must not be misparsed as tool
calls.
"""

import json
from typing import Iterable

import pytest

pytest.importorskip("transformers")

from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import Qwen3XMLToolParser


def _slice(text: str, sizes: Iterable[int]) -> list[str]:
    """Slice text into chunks of the given sizes (last chunk may be shorter)."""
    out = []
    pos = 0
    for n in sizes:
        if pos >= len(text):
            break
        out.append(text[pos : pos + n])
        pos += n
    if pos < len(text):
        out.append(text[pos:])
    return out


def _fixed_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _stream(parser: Qwen3XMLToolParser, chunks: list[str]) -> dict:
    """
    Drive extract_tool_calls_streaming for the given chunk sequence and
    aggregate the per-chunk deltas back into a single result with the same
    shape as extract_tool_calls.

    Returns:
        {
            "tool_calls": list of {"index", "name", "arguments"},
            "content": concatenated plain-text content (str, possibly empty),
        }
    """
    accumulated = ""
    aggregated: dict[int, dict] = {}
    content_parts: list[str] = []

    for chunk in chunks:
        prev = accumulated
        accumulated = prev + chunk
        delta = parser.extract_tool_calls_streaming(prev, accumulated, chunk)
        if delta is None:
            continue
        if "content" in delta and delta["content"]:
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls", []) or []:
            idx = tc.get("index", 0)
            slot = aggregated.setdefault(
                idx, {"index": idx, "name": None, "arguments": ""}
            )
            func = tc.get("function") or {}
            if func.get("name"):
                slot["name"] = func["name"]
            args_piece = func.get("arguments")
            if args_piece:
                slot["arguments"] += args_piece

    return {
        "tool_calls": [aggregated[i] for i in sorted(aggregated)],
        "content": "".join(content_parts),
    }


def _expected_calls(text: str, parser: Qwen3XMLToolParser, request=None):
    """Use non-streaming extract_tool_calls as the reference oracle."""
    parser._xml_parser.reset_streaming_state()
    res = parser.extract_tool_calls(text, request=request)
    if not res.tools_called:
        return None
    return [
        {"name": tc["name"], "arguments": json.loads(tc["arguments"])}
        for tc in res.tool_calls
    ]


def _normalize_streamed(streamed: dict) -> list[dict]:
    return [
        {"name": tc["name"], "arguments": json.loads(tc["arguments"])}
        for tc in streamed["tool_calls"]
        if tc["name"]
    ]


# ---------------------------------------------------------------------------
# Streaming / non-streaming parity at multiple chunk sizes
# ---------------------------------------------------------------------------


SINGLE_CALL = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>Tokyo</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


TWO_CALLS = (
    "<tool_call>\n"
    "<function=read_file>\n"
    "<parameter=path>/a.py</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=path>/b.py</parameter>\n"
    "<parameter=content>hello</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 17, 64, 4096])
def test_single_tool_call_parity_across_chunk_sizes(chunk_size):
    parser = Qwen3XMLToolParser(None)
    expected = _expected_calls(SINGLE_CALL, Qwen3XMLToolParser(None))
    streamed = _stream(parser, _fixed_chunks(SINGLE_CALL, chunk_size))
    assert _normalize_streamed(streamed) == expected


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 17, 64])
def test_two_adjacent_tool_calls_parity_across_chunk_sizes(chunk_size):
    parser = Qwen3XMLToolParser(None)
    expected = _expected_calls(TWO_CALLS, Qwen3XMLToolParser(None))
    streamed = _stream(parser, _fixed_chunks(TWO_CALLS, chunk_size))
    assert _normalize_streamed(streamed) == expected


def test_two_tool_calls_with_no_separator():
    """Adjacent tool calls without any whitespace between them."""
    parser = Qwen3XMLToolParser(None)
    text = (
        "<tool_call>"
        "<function=a><parameter=x>1</parameter></function>"
        "</tool_call>"
        "<tool_call>"
        "<function=b><parameter=y>2</parameter></function>"
        "</tool_call>"
    )
    expected = _expected_calls(text, Qwen3XMLToolParser(None))
    assert expected is not None
    assert [c["name"] for c in expected] == ["a", "b"]
    streamed = _stream(parser, _fixed_chunks(text, 7))
    assert _normalize_streamed(streamed) == expected


# ---------------------------------------------------------------------------
# Tag split at every byte boundary
# ---------------------------------------------------------------------------


def test_tag_split_at_every_byte_position():
    """
    Slice the text at every possible position (chunk_size=1 covers all single
    boundaries; this loop additionally tests a single 2-chunk split at each
    offset, which forces large arbitrary boundaries inside tags). Every
    slicing must produce the same tool call as the non-streaming oracle.
    """
    expected = _expected_calls(SINGLE_CALL, Qwen3XMLToolParser(None))
    assert expected is not None
    for split in range(1, len(SINGLE_CALL)):
        parser = Qwen3XMLToolParser(None)
        streamed = _stream(parser, [SINGLE_CALL[:split], SINGLE_CALL[split:]])
        assert _normalize_streamed(streamed) == expected, (
            f"mismatch when splitting at offset {split} "
            f"({SINGLE_CALL[:split]!r} | {SINGLE_CALL[split:]!r})"
        )


# ---------------------------------------------------------------------------
# Parameter value split inside the body
# ---------------------------------------------------------------------------


def test_long_string_parameter_split_mid_value():
    parser = Qwen3XMLToolParser(None)
    long_value = "abcdefghij" * 25  # 250 chars
    text = (
        "<tool_call>\n"
        "<function=write_file>\n"
        f"<parameter=content>{long_value}</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    expected = _expected_calls(text, Qwen3XMLToolParser(None))
    streamed = _stream(parser, _fixed_chunks(text, 13))
    assert _normalize_streamed(streamed) == expected
    assert _normalize_streamed(streamed)[0]["arguments"]["content"] == long_value


def test_multiline_parameter_split_at_newlines():
    parser = Qwen3XMLToolParser(None)
    body = "def hello():\n    print('hi')\n    return 0\n"
    text = (
        "<tool_call>\n"
        "<function=write_file>\n"
        f"<parameter=content>{body}</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    expected = _expected_calls(text, Qwen3XMLToolParser(None))
    chunks = _slice(
        text,
        [len("<tool_call>\n<function=write_file>\n<parameter=content>")]
        + [len("def hello():\n"), len("    print('hi')\n"), 999],
    )
    streamed = _stream(parser, chunks)
    assert _normalize_streamed(streamed) == expected


# ---------------------------------------------------------------------------
# Plain text that resembles XML must not be parsed as tool calls
# ---------------------------------------------------------------------------


def test_plain_text_with_xml_like_fragments_is_not_a_tool_call():
    parser = Qwen3XMLToolParser(None)
    text = (
        "Sure, here is the XML I would write: "
        "<configuration>\n  <param=key>value</param>\n</configuration>"
    )
    res = parser.extract_tool_calls(text)
    assert not res.tools_called
    # extract_tool_calls returns the original text in `content` when no tool
    # calls are detected, so the user's prose must be preserved verbatim.
    assert res.content == text


def test_streaming_plain_text_is_not_misclassified_as_tool_call():
    """
    A response that is pure prose (no <tool_call>) must stream through
    without producing any tool_calls in any chunk. This pins down the
    invariant that the state machine's pre-tool_call buffer never leaks
    into a synthesized tool call.
    """
    parser = Qwen3XMLToolParser(None)
    text = (
        "I considered using <function=foo> but decided against it. " "The answer is 42."
    )
    aggregated_calls = []
    accumulated = ""
    for piece in _fixed_chunks(text, 4):
        prev = accumulated
        accumulated += piece
        delta = parser.extract_tool_calls_streaming(prev, accumulated, piece)
        if delta and delta.get("tool_calls"):
            aggregated_calls.extend(delta["tool_calls"])
    assert aggregated_calls == []


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8, 256])
def test_streaming_plain_text_preserves_content_across_chunk_sizes(chunk_size):
    """
    Regression for the bare-<function=> rollback path swallowing user-
    visible prose. When the model emits something like
        "I considered using <function=foo> but decided against it."
    the parser must (a) not emit any tool_call AND (b) stream back the
    full original text as content — no characters dropped or rewritten.
    Pre-fix, the abandonment path in _char_data dropped both the raw
    `<function=foo>` fragment and the char-data event that triggered the
    rollback, leaving the response truncated.
    """
    parser = Qwen3XMLToolParser(None)
    text = (
        "I considered using <function=foo> but decided against it. " "The answer is 42."
    )
    streamed = _stream(parser, _fixed_chunks(text, chunk_size))
    assert streamed["tool_calls"] == []
    assert streamed["content"] == text


# ---------------------------------------------------------------------------
# Malformed / truncated input
# ---------------------------------------------------------------------------


def test_streaming_truncated_at_end_of_parameter_recovers_tool_call():
    """Output ends right after </parameter> with no </function></tool_call>."""
    parser = Qwen3XMLToolParser(None)
    text = "<tool_call>\n" "<function=f>\n" "<parameter=x>1</parameter>\n"
    streamed = _stream(parser, _fixed_chunks(text, 3))
    # The non-streaming path already auto-closes; streaming must do the
    # same once a downstream "finalize" path is implemented. Today the
    # streaming wrapper does not synthesize a final flush, so we only
    # assert that the partial result is consistent with what the parser
    # has emitted so far. If the name was emitted, that's enough to
    # signal the tool call has begun, and no spurious extra calls are
    # created.
    names = [tc["name"] for tc in streamed["tool_calls"] if tc["name"]]
    assert names == [] or names == ["f"]


def test_streaming_extra_text_after_tool_call_is_preserved():
    parser = Qwen3XMLToolParser(None)
    pre = "Looking up the weather. "
    post = " Done."
    text = pre + SINGLE_CALL + post

    expected = _expected_calls(text, Qwen3XMLToolParser(None))
    streamed = _stream(parser, _fixed_chunks(text, 11))
    assert _normalize_streamed(streamed) == expected


# ---------------------------------------------------------------------------
# Object / array parameter type coercion across chunk boundaries
# ---------------------------------------------------------------------------


def test_object_parameter_with_single_quotes_split_mid_literal():
    """
    Objects containing single quotes use the deferred-literal-eval path in
    _preprocess_xml_chunk. Chunk-splitting inside such a literal must not
    corrupt the buffered raw value.
    """
    parser = Qwen3XMLToolParser(None)
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "set_config",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "object"}},
                    },
                },
            }
        ]
    }
    text = (
        "<tool_call>\n"
        "<function=set_config>\n"
        "<parameter=value>{'name': 'alice', 'age': 30}</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    expected = _expected_calls(text, Qwen3XMLToolParser(None), request=request)
    assert expected is not None
    assert expected[0]["arguments"]["value"] == {"name": "alice", "age": 30}

    p2 = Qwen3XMLToolParser(None)
    # Streaming with a tool list still has to drive the tools through the
    # request kwarg so the type hint reaches the parser.
    accumulated = ""
    aggregated: dict[int, dict] = {}
    for piece in _fixed_chunks(text, 6):
        prev = accumulated
        accumulated += piece
        delta = p2.extract_tool_calls_streaming(
            prev, accumulated, piece, request=request
        )
        if delta is None:
            continue
        for tc in delta.get("tool_calls", []) or []:
            idx = tc.get("index", 0)
            slot = aggregated.setdefault(
                idx, {"index": idx, "name": None, "arguments": ""}
            )
            func = tc.get("function") or {}
            if func.get("name"):
                slot["name"] = func["name"]
            args_piece = func.get("arguments")
            if args_piece:
                slot["arguments"] += args_piece
    streamed_calls = [
        {"name": v["name"], "arguments": json.loads(v["arguments"])}
        for v in (aggregated[i] for i in sorted(aggregated))
        if v["name"]
    ]
    assert streamed_calls == expected


# ---------------------------------------------------------------------------
# Bare <function=...> (no <tool_call> wrapper)
#
# Qwen3-Coder under heavy load (large system prompt + many tools) sometimes
# drops the outer `<tool_call></tool_call>` wrapper and emits only the inner
# `<function=NAME><parameter=...>...</function>` block. The streaming parser
# must still recognise these as tool calls without misclassifying prose that
# merely contains `<function=foo>` as text (covered by
# `test_streaming_plain_text_is_not_misclassified_as_tool_call` above).
# ---------------------------------------------------------------------------


_BARE_SINGLE = (
    "Thinking.\n<function=Agent>\n"
    "<parameter=description>\nfind bug\n</parameter>\n"
    "<parameter=prompt>\nlook at code\n</parameter>\n"
    "<parameter=subagent_type>\nExplore\n</parameter>\n"
    "</function>"
)

_BARE_TWO = (
    "Step 1.\n<function=Read>\n<parameter=file_path>\n/a.py\n</parameter>\n</function>\n"
    "Step 2.\n<function=Agent>\n<parameter=description>\nfind bug\n</parameter>\n"
    "<parameter=prompt>\nlook\n</parameter>\n<parameter=subagent_type>\nExplore\n</parameter>\n"
    "</function>"
)


@pytest.mark.parametrize("chunk_size", [1, 5, 7, 24, 256])
def test_streaming_bare_function_without_tool_call_wrapper(chunk_size):
    """A bare <function=...> block must stream into a structured tool call.

    Pins down the contract that the model can drop the outer <tool_call>
    wrapper (a documented Qwen3-Coder failure mode under heavy CC load)
    and the streaming parser still emits a correct tool call with
    name + arguments, regardless of how the input is chunked.
    """
    parser = Qwen3XMLToolParser(None)
    expected = _expected_calls(_BARE_SINGLE, parser)
    assert expected == [
        {
            "name": "Agent",
            "arguments": {
                "description": "find bug",
                "prompt": "look at code",
                "subagent_type": "Explore",
            },
        }
    ]
    parser = Qwen3XMLToolParser(None)
    streamed = _stream(parser, _fixed_chunks(_BARE_SINGLE, chunk_size))
    streamed_calls = [
        {"name": tc["name"], "arguments": json.loads(tc["arguments"])}
        for tc in streamed["tool_calls"]
        if tc["name"]
    ]
    assert streamed_calls == expected
    # No leaked tag fragments in user-visible content.
    for fragment in ("<function=", "</function>", "<parameter=", "</parameter>"):
        assert (
            fragment not in streamed["content"]
        ), f"leaked tag fragment {fragment!r} in streamed content"


@pytest.mark.parametrize("chunk_size", [1, 5, 7, 24, 256])
def test_streaming_two_sequential_bare_functions(chunk_size):
    """Two bare <function=...> blocks back-to-back must become two distinct
    tool calls with independent indexes and ids.

    Without the implicit-wrapper close, the second function would reuse
    the first call's id/index and the merged stream would emit a single
    malformed call with doubled brace closers.
    """
    parser = Qwen3XMLToolParser(None)
    expected = _expected_calls(_BARE_TWO, parser)
    assert expected == [
        {"name": "Read", "arguments": {"file_path": "/a.py"}},
        {
            "name": "Agent",
            "arguments": {
                "description": "find bug",
                "prompt": "look",
                "subagent_type": "Explore",
            },
        },
    ]
    parser = Qwen3XMLToolParser(None)
    streamed = _stream(parser, _fixed_chunks(_BARE_TWO, chunk_size))
    streamed_calls = [
        {"name": tc["name"], "arguments": json.loads(tc["arguments"])}
        for tc in streamed["tool_calls"]
        if tc["name"]
    ]
    assert streamed_calls == expected


class TestQwenXmlDeltaStateContract:
    """Characterize the existing delta storage, aliasing and request boundaries."""

    @pytest.mark.parametrize("initial_count", [0, 2])
    def test_empty_merge_does_not_emit_or_replace_history(self, initial_count):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        history = parser.deltas
        result = parser._merge_new_deltas_to_single_response(initial_count)
        assert result.content is None and result.tool_calls is None
        assert parser.deltas is history and history == []

    @pytest.mark.parametrize("offset", [0, 1])
    def test_singleton_merge_returns_original_delta(self, offset):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaMessage,
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        if offset:
            parser._emit_delta(DeltaMessage(content="earlier"))
        delta = DeltaMessage(content="now", role="assistant", reasoning_content="why")
        parser._emit_delta(delta)
        assert parser._merge_new_deltas_to_single_response(offset) is delta

    def test_writer_preserves_order_and_content_merge_skips_old_entries(self):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaMessage,
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        history = parser.deltas
        deltas = [DeltaMessage(content=value) for value in ("old", "A", "B")]
        for delta in deltas:
            parser._emit_delta(delta)
        assert parser.deltas is history
        assert all(actual is delta for actual, delta in zip(history, deltas))
        merged = parser._merge_new_deltas_to_single_response(1)
        assert merged.content == "AB" and merged.tool_calls == []
        assert all(merged is not delta for delta in deltas)
        assert [delta.content for delta in history] == ["old", "A", "B"]

    def test_multi_merge_keeps_nested_identity_order_and_existing_mutation(self):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        function = DeltaFunctionCall(name="initial", arguments="{")
        first = DeltaToolCall(index=0, id="a", function=function)
        other = DeltaToolCall(
            index=1, id="b", function=DeltaFunctionCall(name="other", arguments="{}")
        )
        continuation = DeltaToolCall(
            id="a",
            type="",
            function=DeltaFunctionCall(name="updated", arguments='"x":1}'),
        )
        for call in (first, other, continuation):
            parser._emit_delta(DeltaMessage(tool_calls=[call]))
        merged = parser._merge_new_deltas_to_single_response(0)
        assert merged.tool_calls[0] is first and merged.tool_calls[1] is other
        assert len(merged.tool_calls) == 2
        assert first.function is function
        assert function.name == "updated" and function.arguments == '{"x":1}'
        assert first.type == "function"
        assert parser.deltas[0].tool_calls[0] is first
        assert continuation.function.arguments == '"x":1}'

    def test_full_reset_replaces_list_without_clearing_old_alias(self):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaMessage,
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        delta = DeltaMessage(content="previous request")
        parser._emit_delta(delta)
        history, expat = parser.deltas, parser.parser
        parser.reset_streaming_state()
        assert parser.deltas == [] and parser.deltas is not history
        assert len(history) == 1 and history[0] is delta
        assert parser.parser is not expat
        assert parser.tool_call_index == 0 and parser.current_call_id is None

    def test_parser_instances_do_not_share_delta_state(self):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaMessage,
            StreamingXMLToolCallParser,
        )

        left, right = StreamingXMLToolCallParser(), StreamingXMLToolCallParser()
        delta = DeltaMessage(content="left only")
        left._emit_delta(delta)
        assert left.deltas is not right.deltas and right.deltas == []
        right.reset_streaming_state()
        assert len(left.deltas) == 1 and left.deltas[0] is delta

    def test_per_tool_reset_retains_history_across_actual_xml_calls(self):
        from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (
            DeltaMessage,
            StreamingXMLToolCallParser,
        )

        parser = StreamingXMLToolCallParser()
        history = parser.deltas
        first = parser.parse_single_streaming_chunks(SINGLE_CALL)
        prior_entries = tuple(history)
        old_expat = parser.parser
        parser._reset_xml_parser_after_tool_call()
        assert parser.deltas is history and parser.parser is not old_expat
        second = parser.parse_single_streaming_chunks(SINGLE_CALL)
        assert parser.deltas is history and len(history) > len(prior_entries)
        assert all(history[i] is delta for i, delta in enumerate(prior_entries))
        assert all(isinstance(delta, DeltaMessage) for delta in history)
        assert first.tool_calls[0].index == 0 and second.tool_calls[0].index == 1
        assert first.tool_calls[0].id != second.tool_calls[0].id
        assert first.tool_calls[0].function.arguments == '{"city": "Tokyo"}'
        assert second.tool_calls[0].function.arguments == '{"city": "Tokyo"}'

    def test_complete_request_reuse_replaces_history_and_retains_old_entries(self):
        parser = Qwen3XMLToolParser(None)
        first = parser.extract_tool_calls(SINGLE_CALL)
        history = parser._xml_parser.deltas
        prior_entries = tuple(history)
        second = parser.extract_tool_calls(SINGLE_CALL.replace("Tokyo", "Paris"))
        assert parser._xml_parser.deltas is not history
        assert tuple(history) == prior_entries and history
        assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]
        assert json.loads(first.tool_calls[0]["arguments"]) == {"city": "Tokyo"}
        assert json.loads(second.tool_calls[0]["arguments"]) == {"city": "Paris"}

    @pytest.mark.parametrize("chunk_size", [1, 7, 4096])
    def test_streaming_request_reuse_preserves_fragments_ids_and_content(
        self, chunk_size
    ):
        parser = Qwen3XMLToolParser(None)
        old_history = parser._xml_parser.deltas
        request_ids = []
        for text, city in (
            ("Hello.", None),
            (SINGLE_CALL, "Tokyo"),
            (SINGLE_CALL.replace("Tokyo", "Paris"), "Paris"),
        ):
            prior_entries = tuple(old_history)
            previous, content, arguments = "", "", ""
            names, ids = [], []
            for chunk in _fixed_chunks(text, chunk_size):
                delta = parser.extract_tool_calls_streaming(
                    previous, previous + chunk, chunk
                )
                previous += chunk
                if delta is None:
                    continue
                content += delta.get("content", "")
                for call in delta.get("tool_calls", []):
                    assert call["index"] == 0 and call["type"] == "function"
                    ids.append(call["id"])
                    function = call.get("function", {})
                    if function.get("name"):
                        names.append(function["name"])
                    arguments += function.get("arguments", "")
            assert parser._xml_parser.deltas is not old_history
            assert len(old_history) == len(prior_entries)
            assert all(old_history[i] is delta for i, delta in enumerate(prior_entries))
            old_history = parser._xml_parser.deltas
            if city is None:
                assert content == "Hello." and names == ids == [] and arguments == ""
            else:
                assert content == "" and names == ["get_weather"]
                assert json.loads(arguments) == {"city": city}
                assert arguments == '{"city": "' + city + '"}'
                assert ids and len(set(ids)) == 1 and ids[0].startswith("call_")
                request_ids.append(ids[0])
        assert len(set(request_ids)) == 2
