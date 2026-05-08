# SPDX-License-Identifier: Apache-2.0
"""Tests for MLXWorkerThread persistent worker thread.

Verifies that model loading and inference on the same persistent thread avoids
the ``RuntimeError: There is no Stream(gpu, N) in current thread`` crash that
occurs when MLX >= 0.31.2 thread-local CommandEncoders are accessed from
transient worker threads (e.g. asyncio.to_thread).
"""

import asyncio
import threading

import pytest


def _mlx_available() -> bool:
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except (ImportError, AttributeError):
        return False


@pytest.mark.anyio
async def test_mlx_worker_thread_runs_on_persistent_thread():
    """All submissions execute on the same OS thread."""
    from vllm_mlx.mlx_streams import MLXWorkerThread

    worker = MLXWorkerThread(name="test-worker")
    loop = asyncio.get_event_loop()

    thread_ids = []
    for _ in range(5):
        tid = await worker.submit(loop, threading.get_ident)
        thread_ids.append(tid)

    assert len(set(thread_ids)) == 1, "Worker must use a single persistent thread"
    assert thread_ids[0] != threading.get_ident(), (
        "Worker thread must differ from event loop thread"
    )
    worker.shutdown()


@pytest.mark.anyio
async def test_mlx_worker_thread_preserves_exception():
    """Exceptions from submitted callables propagate correctly."""
    from vllm_mlx.mlx_streams import MLXWorkerThread

    worker = MLXWorkerThread(name="test-exc")
    loop = asyncio.get_event_loop()

    def raise_value_error():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        await worker.submit(loop, raise_value_error)

    worker.shutdown()


@pytest.mark.anyio
async def test_mlx_worker_thread_sequential_execution():
    """Tasks execute sequentially (FIFO) on the single worker thread."""
    from vllm_mlx.mlx_streams import MLXWorkerThread

    worker = MLXWorkerThread(name="test-seq")
    loop = asyncio.get_event_loop()

    results = []

    def append_value(val):
        results.append(val)
        return val

    futs = [worker.submit(loop, append_value, i) for i in range(10)]
    await asyncio.gather(*futs)

    assert results == list(range(10)), "Tasks must execute in submission order"
    worker.shutdown()


@pytest.mark.anyio
@pytest.mark.skipif(
    not _mlx_available(),
    reason="MLX not available",
)
async def test_mlx_ops_on_worker_thread_no_stream_error():
    """MLX array operations on worker thread do not raise stream errors."""
    from vllm_mlx.mlx_streams import MLXWorkerThread

    worker = MLXWorkerThread(name="test-mlx")
    loop = asyncio.get_event_loop()

    def mlx_matmul():
        import mlx.core as mx

        a = mx.ones((32, 32))
        b = mx.ones((32, 32))
        c = a @ b
        mx.eval(c)
        return c[0, 0].item()

    result = await worker.submit(loop, mlx_matmul)
    assert result == 32.0
    worker.shutdown()
