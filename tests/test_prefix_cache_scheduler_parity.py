# SPDX-License-Identifier: Apache-2.0
"""Scheduler-level parity for prefix-cache reuse safety.

These tests use tiny deterministic MLX modules that write real ``ArraysCache``
and ``KVCache`` state through mlx-lm's ``BatchGenerator``.  They intentionally
avoid model downloads while exercising the production scheduler/cache boundary.
"""

from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402


class _Tokenizer:
    eos_token_id = 99
    clean_up_tokenization_spaces = False

    @staticmethod
    def encode(text, **_kwargs):
        return [int(token) for token in text.split()]

    @staticmethod
    def decode(tokens, **_kwargs):
        return "".join(chr(65 + int(token) % 26) for token in tokens)


class _CacheWritingModel(nn.Module):
    """A deterministic model whose logits depend on cumulative cache state."""

    def __init__(self, cache_kind):
        super().__init__()
        self.cache_kind = cache_kind
        # ``make_prompt_cache`` only needs a layer count when no model-owned
        # factory exists.  Keep this populated to match a normal language model.
        self.layers = [object()]

    def make_cache(self):
        if self.cache_kind == "arrays":
            return [cache_mod.ArraysCache(1)]
        return [cache_mod.KVCache()]

    def __call__(self, tokens, cache=None):
        values = tokens.astype(mx.float32)
        layer = cache[0]

        if self.cache_kind == "arrays":
            prior = mx.zeros((values.shape[0], 1)) if layer[0] is None else layer[0]
            cumulative = prior + mx.cumsum(values, axis=1)
            layer[0] = cumulative[:, -1:]
        else:
            prior = (
                mx.zeros((values.shape[0], 1))
                if layer.keys is None
                else mx.sum(layer.keys, axis=(1, 2, 3))[:, None]
            )
            cumulative = prior + mx.cumsum(values, axis=1)
            kv = values[:, None, :, None]
            layer.update_and_fetch(kv, kv)

        selected = cumulative.astype(mx.int32) % 8
        vocabulary = mx.arange(8)[None, None, :]
        return mx.where(vocabulary == selected[:, :, None], 10.0, -10.0)


def _scheduler(cache_kind):
    scheduler = Scheduler(
        _CacheWritingModel(cache_kind),
        _Tokenizer(),
        SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            cache_memory_mb=32,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
        ),
    )
    # Keep the production minimum unchanged; these tiny prompts need a lower
    # threshold solely so the completion-store/reuse path is exercised.
    scheduler.memory_aware_cache._config = replace(
        scheduler.memory_aware_cache._config, min_prefix_tokens=1
    )
    return scheduler


def _generate(scheduler, request_id, prompt, max_tokens=2):
    request = Request(
        request_id=request_id,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    scheduler.add_request(request)
    emitted = []
    for _ in range(16):
        result = scheduler.step()
        for output in result.outputs:
            emitted.extend(output.new_token_ids)
        if request_id in result.finished_request_ids:
            return emitted, request
    raise AssertionError(f"request {request_id} did not finish")


def test_arrays_cache_next_turn_matches_cold_when_coverage_is_unknown():
    warm = _scheduler("arrays")
    prompt = [1, 2, 3]
    first_tokens, _ = _generate(warm, "arrays-first", prompt)

    # ArraysCache has cumulative recurrent state but no authoritative offset.
    # The first response must not create a guessed prompt-keyed entry.
    assert list(warm.memory_aware_cache._entries) == []

    next_prompt = prompt + first_tokens + [7]
    warm_tokens, warm_request = _generate(warm, "arrays-next", next_prompt)
    cold_tokens, cold_request = _generate(
        _scheduler("arrays"), "arrays-cold", next_prompt
    )

    assert warm_request.cached_tokens == cold_request.cached_tokens == 0
    assert warm_request.remaining_tokens == cold_request.remaining_tokens == next_prompt
    assert warm_tokens == cold_tokens


def test_identical_plain_kv_prompt_never_replays_a_supersequence():
    warm = _scheduler("kv")
    prompt = [1, 2, 3]
    cold_tokens, _ = _generate(warm, "kv-first", prompt)

    entries = list(warm.memory_aware_cache._entries)
    assert len(entries) == 1
    assert list(entries[0]) == prompt + cold_tokens

    replay_tokens, replay_request = _generate(warm, "kv-identical", prompt)
    oracle_tokens, _ = _generate(_scheduler("kv"), "kv-cold", prompt)

    # The only stored entry is N+completion tokens and must not be trimmed into
    # an apparent N-token hit.  Cache coverage plus replay is therefore exactly N.
    assert replay_request.cached_tokens == 0
    assert replay_request.remaining_tokens == prompt
    assert replay_request.cached_tokens + len(replay_request.remaining_tokens) == len(
        prompt
    )
    assert replay_tokens == oracle_tokens == cold_tokens
    assert len(warm.memory_aware_cache._entries) == 1
