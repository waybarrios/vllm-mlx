# SPDX-License-Identifier: Apache-2.0
"""Regression guard for issue #407.

EngineCore previously ran scheduler.step on a separate worker thread, which
caused mx.eval over KV cache state to raise
``RuntimeError: There is no Stream(gpu, N) in current thread`` for Llama 3.x
because mlx-lm's ``generation_stream`` is thread-local. The fix keeps step on
the event-loop thread. This test catches any regression that moves step back
onto a non-owning thread.
"""

import asyncio
import logging

import pytest

# Llama 3.x reliably surfaces the cross-thread stream mismatch. Qwen3 does not.
TEST_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    try:
        from mlx_lm import load

        return load(TEST_MODEL)
    except Exception as e:
        pytest.skip(f"Could not load model {TEST_MODEL}: {e}")


@pytest.mark.anyio
async def test_engine_core_no_cross_thread_stream_error(model_and_tokenizer, caplog):
    """EngineCore must run prefill + decode without raising
    ``There is no Stream(gpu, N) in current thread``.

    A regression that moves ``scheduler.step`` off the event-loop thread
    (e.g. re-introducing a ThreadPoolExecutor) reintroduces issue #407.
    """
    from vllm_mlx import AsyncEngineCore, SamplingParams

    model, tokenizer = model_and_tokenizer
    params = SamplingParams(max_tokens=5, temperature=0.0)

    caplog.set_level(logging.ERROR, logger="vllm_mlx.scheduler")

    engine = AsyncEngineCore(model, tokenizer)
    await engine.__aenter__()
    await asyncio.sleep(0.05)

    rid = await engine.add_request("Hello", params)
    tokens = 0
    async for out in engine.stream_outputs(rid, timeout=30):
        tokens += 1
        if out.finished:
            break

    bg = engine.engine.scheduler.batch_generator
    await engine.__aexit__(None, None, None)

    stream_errors = [
        r.message
        for r in caplog.records
        if "Stream(gpu" in r.message or "no Stream" in r.message
    ]
    assert (
        not stream_errors
    ), f"scheduler logged cross-thread stream errors: {stream_errors}"
    assert tokens > 0, "no tokens streamed"
    assert bg is not None, (
        "batch generator was None after generation, meaning the scheduler's "
        "error-recovery path fired. See issue #407."
    )
