# SPDX-License-Identifier: Apache-2.0
"""Tests for ``StreamingJsonFenceStripper`` — strips markdown code fences
from streamed ``response_format`` content.
"""

from vllm_mlx.api.tool_calling import StreamingJsonFenceStripper


def _stream(chunks):
    """Feed ``chunks`` through a fresh stripper and collect emitted text."""
    s = StreamingJsonFenceStripper()
    out = []
    for c in chunks:
        out.append(s.feed(c))
    out.append(s.finalize())
    return "".join(out)


class TestStreamingFenceStripperNoFence:
    def test_single_delta_no_fence(self):
        assert _stream(['{"answer": true}']) == '{"answer": true}'

    def test_multiple_deltas_no_fence(self):
        assert _stream(["{", '"a"', ":", "1", "}"]) == '{"a":1}'

    def test_empty_input(self):
        assert _stream([]) == ""

    def test_only_empty_deltas(self):
        assert _stream(["", "", ""]) == ""

    def test_content_with_inner_backticks_preserved(self):
        # Backticks inside a string value must not be interpreted as a fence.
        text = '{"code": "x `y` z"}'
        assert _stream([text]) == text


class TestStreamingFenceStripperLeadingFence:
    def test_leading_backticks_json_newline(self):
        assert _stream(['```json\n{"a": 1}']) == '{"a": 1}'

    def test_leading_backticks_json(self):
        assert _stream(['```json{"a": 1}']) == '{"a": 1}'

    def test_leading_backticks_newline(self):
        assert _stream(['```\n{"a": 1}']) == '{"a": 1}'

    def test_leading_bare_backticks(self):
        assert _stream(['```{"a": 1}']) == '{"a": 1}'

    def test_leading_whitespace_then_fence(self):
        assert _stream(['  \n ```json\n{"a": 1}']) == '{"a": 1}'

    def test_fence_split_across_deltas_at_backticks(self):
        # Classic case: tokenizer splits ``` from ``json``.
        assert _stream(["```", 'json\n{"a":1}']) == '{"a":1}'

    def test_fence_split_inside_json_word(self):
        assert _stream(["```jso", 'n\n{"a":1}']) == '{"a":1}'

    def test_fence_split_after_each_char(self):
        assert (
            _stream(["`", "`", "`", "j", "s", "o", "n", "\n", '{"a":1}']) == '{"a":1}'
        )

    def test_no_fence_but_starts_with_backtick_char(self):
        # A single backtick should NOT be treated as a fence prefix forever —
        # as soon as a non-fence-prefix char arrives, we emit.
        assert _stream(["`hello`"]) == "`hello`"


class TestStreamingFenceStripperTrailingFence:
    def test_trailing_backticks(self):
        assert _stream(['{"a": 1}```']) == '{"a": 1}'

    def test_trailing_newline_backticks(self):
        assert _stream(['{"a": 1}\n```']) == '{"a": 1}'

    def test_trailing_newline_backticks_newline(self):
        assert _stream(['{"a": 1}\n```\n']) == '{"a": 1}'

    def test_trailing_fence_split_across_deltas(self):
        assert _stream(['{"a": 1}', "\n`", "``\n"]) == '{"a": 1}'

    def test_full_wrap_single_delta(self):
        assert _stream(['```json\n{"a": 1}\n```']) == '{"a": 1}'

    def test_full_wrap_split_deltas(self):
        assert _stream(["```json\n", '{"a": ', "1}", "\n", "```"]) == '{"a": 1}'


class TestStreamingFenceStripperOrdering:
    def test_no_fence_emits_incrementally(self):
        s = StreamingJsonFenceStripper()
        # Long-enough chunk: should emit everything except last 5 chars now.
        first = s.feed('{"result": "ok", "count": 42}')
        # 29 chars in; held back = 5, so 24 emitted.
        assert first == '{"result": "ok", "count"'
        last = s.finalize()
        assert last == ": 42}"
        assert first + last == '{"result": "ok", "count": 42}'

    def test_leading_fence_emits_past_holdback(self):
        s = StreamingJsonFenceStripper()
        out1 = s.feed('```json\n{"a": 1, "b": 2, "c": 3}')
        # After stripping fence: '{"a": 1, "b": 2, "c": 3}' (24 chars).
        # Emit 24 - 5 = 19 chars now.
        assert out1 == '{"a": 1, "b": 2, "c'
        out2 = s.finalize()
        assert out2 == '": 3}'
        assert out1 + out2 == '{"a": 1, "b": 2, "c": 3}'

    def test_leading_fence_only_no_content(self):
        # Stream ends with just the fence and no JSON — should emit nothing.
        assert _stream(["```json\n"]) == ""

    def test_leading_partial_fence_only(self):
        # Stream ends with just a partial fence — finalize should strip it.
        assert _stream(["```js"]) == ""


class TestStreamingFenceStripperEdgeCases:
    def test_content_ending_with_backtick_in_string(self):
        # Single trailing backtick at end of content — held back then emitted.
        text = '{"note": "end `"}'
        assert _stream([text]) == text

    def test_fence_with_extra_whitespace_around_closing(self):
        # Extra whitespace after closing fence still strips.
        assert _stream(['{"a": 1}\n```   ']) == '{"a": 1}'

    def test_array_output(self):
        assert _stream(["```json\n[1, 2, 3]\n```"]) == "[1, 2, 3]"

    def test_empty_object(self):
        assert _stream(["```json\n{}\n```"]) == "{}"
