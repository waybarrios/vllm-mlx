# SPDX-License-Identifier: Apache-2.0
"""Tests for per-token logprob extraction and OpenAI formatting."""

import math
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.logprobs import (  # noqa: E402
    LOGPROB_FLOOR,
    MAX_TOP_LOGPROBS,
    TokenLogprob,
    extract_token_logprob,
    logprobs_for_step,
    to_chat_logprobs,
    to_completion_logprobs,
)


class _CharTokenizer:
    """Tokenizer stand-in that decodes token id ``i`` to ``chr(ord('a') + i)``."""

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


def _log_softmax(values):
    logits = mx.array(values, dtype=mx.float32)
    return logits - mx.logsumexp(logits)


def _expected(values, index):
    top = max(values)
    total = sum(math.exp(v - top) for v in values)
    return values[index] - top - math.log(total)


class TestExtractTokenLogprob:
    def test_chosen_token_and_sorted_top(self):
        values = [0.0, 3.0, 1.0, 2.0]
        entry = extract_token_logprob(_log_softmax(values), 2, 2, _CharTokenizer())

        assert entry.token_id == 2
        assert entry.token == "c"
        assert entry.logprob == pytest.approx(_expected(values, 2), abs=1e-5)
        assert [(tid, tok) for tid, tok, _ in entry.top_logprobs] == [
            (1, "b"),
            (3, "d"),
        ]
        assert entry.top_logprobs[0][2] == pytest.approx(_expected(values, 1), abs=1e-5)

    def test_zero_top_reports_only_the_sampled_token(self):
        entry = extract_token_logprob(_log_softmax([0.0, 1.0]), 1, 0, _CharTokenizer())
        assert entry.top_logprobs == []

    def test_masked_logits_are_clamped_to_floor(self):
        row = _log_softmax([0.0, -math.inf, 1.0])
        entry = extract_token_logprob(row, 1, 3, _CharTokenizer())

        assert entry.logprob == LOGPROB_FLOOR
        assert all(math.isfinite(lp) for _, _, lp in entry.top_logprobs)
        assert entry.top_logprobs[-1][2] == LOGPROB_FLOOR

    def test_placeholder_row_returns_none(self):
        # Error paths emit ``mx.zeros(1)`` instead of a vocabulary row.
        assert extract_token_logprob(mx.zeros(1), 5, 1, _CharTokenizer()) is None

    def test_top_is_capped(self):
        row = _log_softmax([float(i) for i in range(MAX_TOP_LOGPROBS + 10)])
        entry = extract_token_logprob(row, 0, 50, _CharTokenizer())
        assert len(entry.top_logprobs) == MAX_TOP_LOGPROBS


class TestLogprobsForStep:
    def _request(self):
        return SimpleNamespace(output_logprobs=None)

    def _response(self, token, finish_reason=None):
        return SimpleNamespace(
            token=token,
            logprobs=_log_softmax([0.0, 1.0, 2.0]),
            finish_reason=finish_reason,
        )

    def test_disabled_when_not_requested(self):
        request = self._request()
        assert (
            logprobs_for_step(request, self._response(1), None, _CharTokenizer())
            is None
        )
        assert request.output_logprobs is None

    def test_appends_content_tokens(self):
        request = self._request()
        step = logprobs_for_step(request, self._response(2), 1, _CharTokenizer())

        assert [entry.token for entry in step] == ["c"]
        assert request.output_logprobs == step

    def test_stop_token_is_excluded(self):
        request = self._request()
        step = logprobs_for_step(
            request, self._response(0, finish_reason="stop"), 1, _CharTokenizer()
        )

        assert step == []
        assert request.output_logprobs == []


class TestFormatting:
    def _entries(self):
        return [
            TokenLogprob(
                token_id=0,
                token="Hé",
                logprob=-0.1,
                top_logprobs=[(0, "Hé", -0.1), (1, "Hi", -2.5)],
            ),
            TokenLogprob(token_id=2, token="!", logprob=-0.3),
        ]

    def test_chat_format(self):
        logprobs = to_chat_logprobs(self._entries())

        first = logprobs.content[0]
        assert first.token == "Hé"
        assert first.bytes == list("Hé".encode("utf-8"))
        assert [top.token for top in first.top_logprobs] == ["Hé", "Hi"]
        assert logprobs.content[1].top_logprobs == []
        assert logprobs.refusal is None

    def test_completion_format(self):
        logprobs = to_completion_logprobs(self._entries(), text_offset=5)

        assert logprobs.tokens == ["Hé", "!"]
        assert logprobs.token_logprobs == [-0.1, -0.3]
        assert logprobs.text_offset == [5, 7]
        # The legacy format always includes the sampled token among the top ones.
        assert logprobs.top_logprobs[1] == {"!": -0.3}
        assert logprobs.top_logprobs[0] == {"Hé": -0.1, "Hi": -2.5}
