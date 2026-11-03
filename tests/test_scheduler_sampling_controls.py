# SPDX-License-Identifier: Apache-2.0
"""Request sampling settings must survive text continuous batching."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.models.llm import MLXLanguageModel
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class _RecordingBatch:
    def __init__(self):
        self.inserts = []

    def insert(self, prompts, **kwargs):
        self.inserts.append((prompts, kwargs))
        return [len(self.inserts)]


def _scheduler(monkeypatch, recording=True):
    scheduler = Scheduler(
        SimpleNamespace(layers=[object()]),
        SimpleNamespace(eos_token_id=-1, encode=lambda text: list(text)),
        SchedulerConfig(enable_prefix_cache=False),
    )
    if recording:
        scheduler.batch_generator = _RecordingBatch()
        monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda _params: None)
    return scheduler


def _request(request_id, max_tokens=2, **kwargs):
    return Request(
        request_id=request_id,
        prompt=[0, 1, 2],
        sampling_params=SamplingParams(max_tokens=max_tokens, **kwargs),
    )


def _sample(sampler):
    mx.random.seed(123)
    logits = mx.broadcast_to(mx.array([[1.0, 1.01, 1.02, 1.03]]), (64, 4))
    return set(sampler(logits).tolist())


@pytest.mark.parametrize("top_k", [0, 1, 2])
def test_generator_default_applies_top_k(monkeypatch, top_k):
    scheduler = _scheduler(monkeypatch, recording=False)
    monkeypatch.setattr(
        "vllm_mlx.scheduler.BatchGenerator", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    batch = scheduler._create_batch_generator(
        SamplingParams(temperature=2, top_k=top_k)
    )

    assert _sample(batch.sampler) == (
        set(range(4)) if top_k == 0 else set(range(4 - top_k, 4))
    )


def test_idle_generator_rebuilds_when_only_top_k_changes(monkeypatch):
    scheduler = _scheduler(monkeypatch, recording=False)
    created = []

    def make_batch(**kwargs):
        batch = SimpleNamespace(**kwargs, close=lambda: None)
        created.append(batch)
        return batch

    monkeypatch.setattr("vllm_mlx.scheduler.BatchGenerator", make_batch)
    scheduler._ensure_batch_generator(SamplingParams(temperature=2, top_k=0))
    scheduler._ensure_batch_generator(SamplingParams(temperature=2, top_k=1))

    assert len(created) == 2
    assert _sample(scheduler.batch_generator.sampler) == {3}


def test_concurrent_requests_keep_independent_top_k_samplers(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    for index, top_k in enumerate((1, 2, 0)):
        scheduler.add_request(_request(str(index), temperature=2, top_k=top_k))
        scheduler._schedule_waiting()

    samplers = [
        kwargs["samplers"][0] for _, kwargs in scheduler.batch_generator.inserts
    ]
    assert len({id(sampler) for sampler in samplers}) == 3
    assert [_sample(sampler) for sampler in samplers] == [{3}, {2, 3}, {0, 1, 2, 3}]


@pytest.mark.parametrize("penalty", [-2.0, 0.0, 2.0])
def test_presence_penalty_changes_only_seen_token_logits(monkeypatch, penalty):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_request("presence", presence_penalty=penalty))

    scheduler._schedule_waiting()

    processors = scheduler.batch_generator.inserts[-1][1]["logits_processors"][0]
    logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
    for processor in processors:
        logits = processor(mx.array([0, 0, 2]), logits)
    assert logits.tolist() == [[1.0 - penalty, 2.0, 3.0 - penalty, 4.0]]
    if penalty == 0:
        assert processors == []


def test_presence_composes_with_repetition_before_caller_processors(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    seen = []

    def observe(_tokens, logits):
        seen.append(logits.tolist())
        return logits

    scheduler.add_request(
        _request(
            "combined",
            repetition_penalty=2.0,
            presence_penalty=1.0,
            logits_processors=[observe],
        )
    )
    scheduler._schedule_waiting()
    processors = scheduler.batch_generator.inserts[-1][1]["logits_processors"][0]
    logits = mx.array([[4.0, 3.0, -2.0, 1.0]])
    for processor in processors:
        logits = processor(mx.array([0, 2]), logits)

    assert seen == [[[1.0, 3.0, -5.0, 1.0]]]
    assert processors[-1] is observe


@pytest.mark.parametrize(
    "penalty, expected",
    [(2, [[0.0, -1.5, 1.0, -20.0]]), (-2, [[0.0, 2.5, 1.0, -20.0]])],
)
def test_presence_history_window_matches_simple_engine(monkeypatch, penalty, expected):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_request("history-window", presence_penalty=penalty))
    scheduler._schedule_waiting()
    batched_processors = scheduler.batch_generator.inserts[-1][1]["logits_processors"][
        0
    ]
    simple_processors = MLXLanguageModel("test-model")._create_logits_processors(
        presence_penalty=penalty
    )

    for processors in (simple_processors, batched_processors):
        logits = mx.array([[0.0, 0.5, 1.0, -20.0]])
        for processor in processors:
            logits = processor(mx.array([2] + [1] * 20), logits)
        assert logits.tolist() == expected


class _ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [object()]

    def __call__(self, inputs, cache=None):
        if cache:
            data = mx.zeros((inputs.shape[0], 1, inputs.shape[1], 4))
            cache[0].update_and_fetch(data, data)
        return mx.broadcast_to(mx.array([0.0, 0.5, 1.0, -20.0]), (*inputs.shape, 4))


def _native_scheduler():
    return Scheduler(
        _ConstantModel(),
        SimpleNamespace(
            eos_token_id=3,
            clean_up_tokenization_spaces=False,
            encode=lambda text: list(text),
            decode=lambda tokens, **_kwargs: "".join(map(str, tokens)),
        ),
        SchedulerConfig(enable_prefix_cache=False, prefill_step_size=1),
    )


def _drain(scheduler, wanted, token_count=2):
    outputs = {request_id: [] for request_id in wanted}
    for _ in range(token_count + 40):
        result = scheduler.step(max_retries=0)
        for response in result.outputs:
            assert response.finish_reason != "error"
            outputs[response.request_id].extend(response.new_token_ids)
        if all(len(tokens) >= token_count for tokens in outputs.values()):
            return outputs
    pytest.fail("native batch did not finish within the step budget")


def test_real_batch_top_k_one_matches_greedy_at_high_temperature():
    scheduler = _native_scheduler()
    request = _request("top-one", temperature=2, top_k=1)
    scheduler.add_request(request)
    scheduler._schedule_waiting()
    try:
        outputs = _drain(scheduler, [request.request_id])
        assert outputs[request.request_id] == [2, 2]
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


@pytest.mark.parametrize("first_top_k", [0, 1])
def test_real_active_batch_keeps_independent_top_k_samplers(first_top_k):
    mx.random.seed(123)
    scheduler = _native_scheduler()
    first = _request("first", max_tokens=64, temperature=2, top_k=first_top_k)
    first.prompt = [2] * 5
    scheduler.add_request(first)
    scheduler._schedule_waiting()
    scheduler.step(max_retries=0)
    batch = scheduler.batch_generator

    second = _request("second", max_tokens=64, temperature=2, top_k=1 - first_top_k)
    second.prompt = [2] * 5
    scheduler.add_request(second)
    scheduler._schedule_waiting()
    try:
        assert scheduler.batch_generator is batch
        outputs = _drain(
            scheduler, [first.request_id, second.request_id], token_count=64
        )
        restricted = first if first_top_k == 1 else second
        unrestricted = second if first_top_k == 1 else first
        assert outputs[restricted.request_id] == [2] * 64
        assert set(outputs[unrestricted.request_id]) == {0, 1, 2}
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


def test_real_batch_adds_presence_request_while_plain_prompt_is_processing():
    scheduler = _native_scheduler()
    plain = _request("plain", temperature=0)
    plain.prompt = [2] * 5
    scheduler.add_request(plain)
    scheduler._schedule_waiting()
    scheduler.step(max_retries=0)
    penalized = _request("penalized", temperature=0, presence_penalty=2)
    penalized.prompt = [2] * 5
    scheduler.add_request(penalized)
    scheduler._schedule_waiting()
    try:
        outputs = _drain(scheduler, [plain.request_id, penalized.request_id])
        assert outputs[plain.request_id] == [2, 2]
        assert outputs[penalized.request_id] == [1, 0]
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


@pytest.mark.parametrize("cached", [False, True], ids=["cold", "cached-prefix"])
def test_real_presence_penalty_includes_cached_prompt_history(cached):
    scheduler = _native_scheduler()
    prompt = [2, 1, 1, 1]
    if cached:
        scheduler.memory_aware_cache = MemoryAwarePrefixCache(
            scheduler.model, MemoryCacheConfig(max_memory_mb=8, min_prefix_tokens=1)
        )
        cache = make_prompt_cache(scheduler.model)
        mx.eval(scheduler.model(mx.array([prompt[:-1]]), cache=cache))
        assert scheduler.memory_aware_cache.store(prompt[:-1], cache)
    request = _request("history", temperature=0, presence_penalty=2)
    request.prompt = prompt
    scheduler.add_request(request)
    scheduler._schedule_waiting()
    if cached:
        assert request.cached_tokens > 0
    try:
        outputs = _drain(scheduler, [request.request_id])
        assert outputs[request.request_id] == [0, 2]
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


@pytest.mark.parametrize("cached", [False, True], ids=["cold", "cached-prefix"])
@pytest.mark.parametrize("penalty, expected", [(2, [2, 0]), (-2, [1, 1])])
def test_real_presence_respects_simple_engine_history_window(cached, penalty, expected):
    scheduler = _native_scheduler()
    prompt = [2] + [1] * 20
    if cached:
        scheduler.memory_aware_cache = MemoryAwarePrefixCache(
            scheduler.model, MemoryCacheConfig(max_memory_mb=8, min_prefix_tokens=1)
        )
        cache = make_prompt_cache(scheduler.model)
        mx.eval(scheduler.model(mx.array([prompt[:-1]]), cache=cache))
        assert scheduler.memory_aware_cache.store(prompt[:-1], cache)
    request = _request("long-history", temperature=0, presence_penalty=penalty)
    request.prompt = prompt
    scheduler.add_request(request)
    scheduler._schedule_waiting()
    if cached:
        assert request.cached_tokens > 0
    try:
        outputs = _drain(scheduler, [request.request_id])
        assert outputs[request.request_id] == expected
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


def test_cached_insert_retry_resets_history_without_changing_sampler(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    request = _request("retry", temperature=2, top_k=1, presence_penalty=2)
    scheduler.add_request(request)
    request.prompt_cache = [object()]
    request.remaining_tokens = request.prompt_token_ids[-1:]
    request.cached_tokens = 2
    monkeypatch.setattr(
        scheduler, "_restored_cache_matches_kv_bound", lambda _cache: True
    )
    monkeypatch.setattr(scheduler, "_validate_cache", lambda _cache: True)
    batch = scheduler.batch_generator
    original_insert = batch.insert

    def reject_cache(prompts, **kwargs):
        uids = original_insert(prompts, **kwargs)
        if kwargs["caches"] is not None:
            raise ValueError("rejected restored cache")
        return uids

    monkeypatch.setattr(batch, "insert", reject_cache)

    scheduler._schedule_waiting()

    first, second = [kwargs for _, kwargs in batch.inserts]
    assert first["all_tokens"] == [[0, 1]]
    assert second["all_tokens"] == [[]]
    assert batch.inserts[1][0] == [[0, 1, 2]]
    assert second["samplers"][0] is first["samplers"][0]
    assert second["logits_processors"][0] is first["logits_processors"][0]
    assert _sample(second["samplers"][0]) == {3}
