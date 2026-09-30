# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint scheduler parity for prefix-cache safety."""

import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache  # noqa: E402

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

pytestmark = pytest.mark.slow


def _required_model_path(variable):
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get(variable)
    if not value or not Path(value).joinpath("config.json").is_file():
        pytest.fail(f"{variable} must name a complete local model artifact")
    return value


def _scheduler(model, tokenizer):
    return Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            cache_memory_mb=256,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
        ),
    )


def _load_vlm_text_model(path):
    from mlx_vlm import load as load_vlm

    from vllm_mlx.text_model_from_vlm import build_text_model

    vlm_model, processor = load_vlm(path, strict=False)
    text_model = build_text_model(vlm_model, path)
    assert text_model is not None
    tokenizer = getattr(processor, "tokenizer", processor)
    assert callable(getattr(tokenizer, "encode", None))
    return text_model, tokenizer


def _long_prompt(tokenizer, minimum=144):
    seed = tokenizer.encode(
        "The cache must preserve every represented token across scheduler reuse."
    )
    assert seed
    prompt = (seed * ((minimum + len(seed) - 1) // len(seed)))[:minimum]
    return [int(token) for token in prompt]


def _generate(scheduler, request_id, prompt, max_tokens=4):
    request = Request(
        request_id=request_id,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    scheduler.add_request(request)
    emitted = []
    for _ in range(32):
        result = scheduler.step()
        for output in result.outputs:
            emitted.extend(output.new_token_ids)
        if request_id in result.finished_request_ids:
            return emitted, request
    raise AssertionError(f"request {request_id} did not finish")


def test_real_arrays_cache_next_turn_matches_cold():
    path = _required_model_path("VLLM_MLX_ARRAYS_MODEL")
    model, tokenizer = load(path)
    topology = make_prompt_cache(model)
    assert topology and all(isinstance(layer, ArraysCache) for layer in topology)

    warm = _scheduler(model, tokenizer)
    prompt = _long_prompt(tokenizer)
    first_tokens, _ = _generate(warm, "arrays-first", prompt)
    assert first_tokens
    assert list(warm.memory_aware_cache._entries) == []

    suffix = tokenizer.encode(" Next turn.")
    assert suffix
    next_prompt = prompt + first_tokens + [int(suffix[0])]
    warm_tokens, warm_request = _generate(warm, "arrays-next", next_prompt)
    cold_tokens, cold_request = _generate(
        _scheduler(model, tokenizer), "arrays-cold", next_prompt
    )

    assert warm_request.cached_tokens == cold_request.cached_tokens == 0
    assert warm_request.remaining_tokens == cold_request.remaining_tokens == next_prompt
    assert warm_tokens == cold_tokens


def test_real_kv_cache_identical_prompt_matches_cold_without_replay():
    path = _required_model_path("VLLM_MLX_KV_MODEL")
    model, tokenizer = load(path)
    topology = make_prompt_cache(model)
    assert topology and all(isinstance(layer, KVCache) for layer in topology)

    warm = _scheduler(model, tokenizer)
    prompt = _long_prompt(tokenizer)
    first_tokens, _ = _generate(warm, "kv-first", prompt)
    assert first_tokens

    entries = list(warm.memory_aware_cache._entries)
    assert len(entries) == 1
    assert list(entries[0]) == prompt + first_tokens

    replay_tokens, replay_request = _generate(warm, "kv-identical", prompt)
    cold_tokens, _ = _generate(_scheduler(model, tokenizer), "kv-cold", prompt)

    assert replay_tokens == cold_tokens == first_tokens, (
        f"warm={replay_tokens} cold={cold_tokens} first={first_tokens} "
        f"hit={replay_request.cache_hit_type} cached={replay_request.cached_tokens}"
    )
    assert replay_request.cached_tokens == 0
    assert replay_request.remaining_tokens == prompt
    assert replay_request.cached_tokens + len(replay_request.remaining_tokens) == len(
        prompt
    )
    assert len(warm.memory_aware_cache._entries) == 1


def test_real_hybrid_qwen_next_turn_matches_cold():
    path = _required_model_path("VLLM_MLX_HYBRID_MODEL")
    model, tokenizer = _load_vlm_text_model(path)
    topology = make_prompt_cache(model)
    assert any(isinstance(layer, ArraysCache) for layer in topology)
    assert any(isinstance(layer, KVCache) for layer in topology)

    warm = _scheduler(model, tokenizer)
    prompt = _long_prompt(tokenizer)
    first_tokens, _ = _generate(warm, "hybrid-first", prompt)
    assert first_tokens

    suffix = tokenizer.encode(" Next turn.")
    next_prompt = prompt + first_tokens + [int(suffix[0])]
    warm_tokens, warm_request = _generate(warm, "hybrid-next", next_prompt)
    cold_tokens, cold_request = _generate(
        _scheduler(model, tokenizer), "hybrid-cold", next_prompt
    )

    assert warm_tokens == cold_tokens
    assert warm_request.cache_hit_type == "prefix"
    assert warm_request.cached_tokens > 0
    assert warm_request.cached_tokens + len(warm_request.remaining_tokens) == len(
        next_prompt
    )
    assert cold_request.cached_tokens == 0


def test_real_gemma_identical_prompt_matches_cold_without_replay():
    path = _required_model_path("VLLM_MLX_GEMMA_MODEL")
    model, tokenizer = _load_vlm_text_model(path)
    topology = make_prompt_cache(model)
    assert topology
    assert not any(isinstance(layer, ArraysCache) for layer in topology)

    warm = _scheduler(model, tokenizer)
    prompt = _long_prompt(tokenizer)
    first_tokens, _ = _generate(warm, "gemma-first", prompt)
    assert first_tokens
    entries = list(warm.memory_aware_cache._entries)
    assert len(entries) == 1
    assert list(entries[0]) == prompt + first_tokens

    replay_tokens, replay_request = _generate(warm, "gemma-identical", prompt)
    cold_tokens, _ = _generate(_scheduler(model, tokenizer), "gemma-cold", prompt)

    assert replay_tokens == cold_tokens == first_tokens
    assert replay_request.cached_tokens == 0
    assert replay_request.remaining_tokens == prompt
    assert replay_request.cached_tokens + len(replay_request.remaining_tokens) == len(
        prompt
    )
