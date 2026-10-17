# SPDX-License-Identifier: Apache-2.0
"""MLLMScheduler.stop() must not leave the model reachable after it returns.

Registry idle unload calls BatchedEngine.stop(), which drops its references
and then collects and clears MLX's cache in the same event-loop step. If
stop() re-raises the processing loop's CancelledError, asyncio keeps that
exception alive past the step, and its traceback pins the scheduler (and
``_process_loop``'s ``bg`` local, the batch generator). Chunked prefill makes
the batch generator self-referential, so once that pin finally drops nothing
collects it again and the weights stay resident.

``vllm_mlx.mllm_scheduler`` hard-imports ``mlx.core``; where it isn't
installed ``tests._mlx_stub`` stubs it. The test only exercises asyncio and
Python reference bookkeeping, so it is meaningful either way. Kept in its own
CI step for the reason given in _mlx_stub.py's docstring.
"""

from tests import _mlx_stub

_mlx_stub.install_if_unavailable()

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import asyncio
import gc
import weakref

import pytest

pytestmark = pytest.mark.skipif(
    not HAS_MLX, reason="mlx.core not importable (even as a stub)"
)


class _FakeTokenizer:
    eos_token_id = None

    def encode(self, text):
        return [1, 2, 3]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeModel:
    config = None


class _SelfReferencingBatchGenerator:
    """Stands in for an MLLMBatchGenerator patched by chunked prefill/MTP,
    which install closures over the generator onto the generator itself."""

    def __init__(self):
        self.unprocessed_requests = []
        self._next = lambda: self

    def close(self):
        pass


@pytest.mark.anyio
async def test_stop_does_not_pin_scheduler_or_batch_generator():
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    scheduler = MLLMScheduler(model=_FakeModel(), processor=_FakeProcessor())
    scheduler.batch_generator = _SelfReferencingBatchGenerator()
    scheduler._running = True
    scheduler._processing_task = asyncio.create_task(scheduler._process_loop())
    # Let the loop take its `bg` local and park on its idle sleep.
    await asyncio.sleep(0.05)

    scheduler_ref = weakref.ref(scheduler)
    model_ref = weakref.ref(scheduler.model)
    batch_generator_ref = weakref.ref(scheduler.batch_generator)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        await scheduler.stop()
        del scheduler
        # Same event-loop step as stop()'s return, like BatchedEngine.stop().
        gc.collect()

        assert batch_generator_ref() is None
        assert scheduler_ref() is None
        assert model_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
