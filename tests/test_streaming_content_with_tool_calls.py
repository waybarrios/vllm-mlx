# SPDX-License-Identifier: Apache-2.0
"""A streaming delta may carry assistant text *and* tool calls.

The streaming paths treated ``tool_calls in result`` as "suppress everything",
so a parser that had buffered prose and then saw the whole tool-call block
arrive in one delta lost that prose. It has nowhere else to put it: the block
is complete, so there is no later delta to flush into, and the non-streaming
path returns the same text happily. The result is an assistant message whose
text silently disappears depending only on how the model's output happened to
be chunked.
"""

from vllm_mlx.server import _parse_streaming_tool_content


class _StubParser:
    """Returns one canned streaming result, like a parser mid-block."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def extract_tool_calls_streaming(self, *args, **kwargs):
        self.calls += 1
        return self._result


def _suppressed(result) -> bool:
    _, _, suppress = _parse_streaming_tool_content(
        _StubParser(result), "accumulated", "delta", {}
    )
    return suppress


class TestSuppressionKeepsText:
    def test_text_alongside_tool_calls_is_not_suppressed(self):
        """The regression: both present, and the text used to be dropped."""
        assert (
            _suppressed({"content": "Checking.", "tool_calls": [{"id": "1"}]}) is False
        )

    def test_tool_calls_alone_are_still_suppressed(self):
        """Nothing to show — the calls go out through their own path."""
        assert _suppressed({"tool_calls": [{"id": "1"}]}) is True

    def test_empty_text_alongside_tool_calls_is_suppressed(self):
        """An empty string must not open a text block."""
        assert _suppressed({"content": "", "tool_calls": [{"id": "1"}]}) is True

    def test_plain_text_is_unaffected(self):
        assert _suppressed({"content": "hello"}) is False

    def test_none_is_still_suppressed(self):
        """None means "inside markup, hold everything"."""
        assert _suppressed(None) is True

    def test_accumulated_text_still_advances(self):
        accumulated, result, _ = _parse_streaming_tool_content(
            _StubParser({"content": "x", "tool_calls": [{"id": "1"}]}),
            "abc",
            "def",
            {},
        )
        assert accumulated == "abcdef"
        assert result["content"] == "x"
