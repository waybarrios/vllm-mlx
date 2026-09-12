# SPDX-License-Identifier: Apache-2.0
"""Tests for per-token logprob extraction, recording and OpenAI formatting."""

import math
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.logprobs import (  # noqa: E402
    LOGPROB_FLOOR,
    MAX_TOP_LOGPROBS,
    TokenLogprob,
    extract_token_logprob,
    record_step_logprobs,
    to_chat_logprobs,
    to_completion_logprobs,
)
from vllm_mlx.request import RequestOutput, SamplingParams  # noqa: E402


class _CharTokenizer:
    """Tokenizer stand-in that decodes token id ``i`` to ``chr(ord('a') + i)``."""

    clean_up_tokenization_spaces = False

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


def _log_softmax(values):
    logits = mx.array(values, dtype=mx.float32)
    return logits - mx.logsumexp(logits)


def _reference_logprob(values, index):
    top = max(values)
    total = sum(math.exp(v - top) for v in values)
    return values[index] - top - math.log(total)


def _entry(token_id, token, logprob=-0.5):
    return TokenLogprob(token_id=token_id, token=token, logprob=logprob)


class TestExtractTokenLogprob:
    def test_chosen_token_and_sorted_top(self):
        values = [0.0, 3.0, 1.0, 2.0]
        entry = extract_token_logprob(_log_softmax(values), 2, 2, _CharTokenizer())

        assert entry.token_id == 2
        assert entry.token == "c"
        assert entry.logprob == pytest.approx(_reference_logprob(values, 2), abs=1e-5)
        (best_id, best_token, best_logprob), (next_id, next_token, _) = (
            entry.top_logprobs
        )
        assert (best_id, best_token) == (1, "b")
        assert (next_id, next_token) == (3, "d")
        assert best_logprob == pytest.approx(_reference_logprob(values, 1), abs=1e-5)

    def test_zero_top_reports_only_the_sampled_token(self):
        entry = extract_token_logprob(_log_softmax([0.0, 1.0]), 1, 0, _CharTokenizer())
        assert entry.top_logprobs == []

    def test_masked_logits_are_clamped_to_floor(self):
        row = _log_softmax([0.0, -math.inf, 1.0])
        entry = extract_token_logprob(row, 1, 3, _CharTokenizer())

        assert entry.logprob == LOGPROB_FLOOR
        top_values = [logprob for _, _, logprob in entry.top_logprobs]
        assert all(math.isfinite(value) for value in top_values)
        assert top_values[-1] == LOGPROB_FLOOR

    def test_placeholder_row_returns_none(self):
        # Error paths emit ``mx.zeros(1)`` instead of a vocabulary row.
        assert extract_token_logprob(mx.zeros(1), 5, 1, _CharTokenizer()) is None

    def test_top_is_capped(self):
        vocab_size = MAX_TOP_LOGPROBS + 10
        row = _log_softmax([float(i) for i in range(vocab_size)])
        entry = extract_token_logprob(row, 0, MAX_TOP_LOGPROBS + 30, _CharTokenizer())
        assert len(entry.top_logprobs) == MAX_TOP_LOGPROBS


class TestRecordStepLogprobs:
    def _request(self, num_top):
        return SimpleNamespace(
            output_logprobs=None, sampling_params=SimpleNamespace(logprobs=num_top)
        )

    def _response(self, token, finish_reason=None):
        return SimpleNamespace(
            token=token,
            logprobs=_log_softmax([0.0, 1.0, 2.0]),
            finish_reason=finish_reason,
        )

    def test_disabled_when_not_requested(self):
        request = self._request(None)
        step = record_step_logprobs(request, self._response(1), _CharTokenizer())

        assert step is None
        assert request.output_logprobs is None

    def test_appends_content_tokens(self):
        request = self._request(1)
        step = record_step_logprobs(request, self._response(2), _CharTokenizer())

        assert [entry.token for entry in step] == ["c"]
        assert request.output_logprobs == step

    def test_stop_token_is_excluded(self):
        request = self._request(1)
        step = record_step_logprobs(
            request, self._response(0, finish_reason="stop"), _CharTokenizer()
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


class TestOutputCollectorMerge:
    """The collector merges outputs when the producer runs ahead of the consumer."""

    def _merge(self, existing, new):
        from vllm_mlx.output_collector import RequestOutputCollector

        return RequestOutputCollector()._merge_outputs(existing, new)

    def test_not_requested_stays_none(self):
        merged = self._merge(RequestOutput("r"), RequestOutput("r"))
        assert merged.new_logprobs is None
        assert merged.output_logprobs is None

    def test_step_entries_concatenate_and_latest_cumulative_wins(self):
        first, second = _entry(0, "a"), _entry(1, "b")
        existing = RequestOutput("r", new_logprobs=[first], output_logprobs=[first])
        new = RequestOutput("r", new_logprobs=[second], output_logprobs=[first, second])

        merged = self._merge(existing, new)

        assert merged.new_logprobs == [first, second]
        assert merged.output_logprobs == [first, second]

    def test_requested_but_empty_is_preserved(self):
        merged = self._merge(RequestOutput("r"), RequestOutput("r", new_logprobs=[]))
        assert merged.new_logprobs == []


class TestSchedulersRecordLogprobs:
    """Both schedulers attach this step's entries and the cumulative list."""

    def _mllm_scheduler(self, request):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._detokenizer_pool = {}
        scheduler.uid_to_request_id = {0: request.request_id}
        scheduler.running = {request.request_id: request}
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0
        scheduler.processor = SimpleNamespace(tokenizer=_CharTokenizer())
        return scheduler

    def _mllm_response(self, request, token, finish_reason=None):
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        return MLLMBatchResponse(
            uid=0,
            request_id=request.request_id,
            token=token,
            logprobs=_log_softmax([0.0, 1.0, 2.0]),
            finish_reason=finish_reason,
        )

    def _mllm_request(self, num_top):
        from vllm_mlx.mllm_scheduler import MLLMRequest

        return MLLMRequest(
            request_id="req-1",
            prompt="p",
            sampling_params=SamplingParams(logprobs=num_top),
        )

    def test_mllm_scheduler_records_content_and_skips_stop(self):
        request = self._mllm_request(1)
        scheduler = self._mllm_scheduler(request)

        (step,), _ = scheduler._process_batch_responses(
            [self._mllm_response(request, 2)]
        )
        (last,), finished = scheduler._process_batch_responses(
            [self._mllm_response(request, 0, finish_reason="stop")]
        )

        assert [entry.token for entry in step.new_logprobs] == ["c"]
        assert last.new_logprobs == []
        assert [entry.token for entry in last.output_logprobs] == ["c"]
        assert finished == {"req-1"}

    def test_mllm_scheduler_leaves_fields_unset_when_not_requested(self):
        request = self._mllm_request(None)
        scheduler = self._mllm_scheduler(request)

        (step,), _ = scheduler._process_batch_responses(
            [self._mllm_response(request, 2)]
        )

        assert step.new_logprobs is None
        assert step.output_logprobs is None

    def test_llm_scheduler_records_content_and_skips_stop(self):
        from vllm_mlx.request import Request
        from vllm_mlx.scheduler import Scheduler

        request = Request(
            request_id="req-1",
            prompt="p",
            sampling_params=SamplingParams(logprobs=1),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._actual_tokenizer = _CharTokenizer()
        scheduler._detokenizer_pool = {}
        scheduler.uid_to_request_id = {0: "req-1"}
        scheduler.running = {"req-1": request}
        scheduler.block_aware_cache = None
        scheduler.total_completion_tokens = 0
        scheduler.num_requests_processed = 0
        scheduler._store_prompt_only_cache = lambda request, response: None

        def response(token, finish_reason=None):
            return SimpleNamespace(
                uid=0,
                token=token,
                logprobs=_log_softmax([0.0, 1.0, 2.0]),
                finish_reason=finish_reason,
            )

        (step,), _ = scheduler._process_batch_responses([response(2)])
        (last,), finished = scheduler._process_batch_responses(
            [response(0, finish_reason="stop")]
        )

        assert [entry.token for entry in step.new_logprobs] == ["c"]
        assert last.new_logprobs == []
        assert [entry.token for entry in last.output_logprobs] == ["c"]
        assert finished == {"req-1"}
