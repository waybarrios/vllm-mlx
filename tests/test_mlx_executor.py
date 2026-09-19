# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MLXExecutor."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_mlx.mlx_executor import MLXExecutor


def test_executor_runs_on_dedicated_thread():
    executor = MLXExecutor(thread_name_prefix="test-mlx")
    try:
        main_tid = threading.get_ident()
        worker_tid = executor.run(threading.get_ident)

        assert worker_tid != main_tid
        assert executor.owner_thread_id == worker_tid

        # Multiple calls stay on the exact same thread
        for _ in range(5):
            assert executor.run(threading.get_ident) == worker_tid
    finally:
        executor.shutdown(wait=True)


def test_executor_inline_detection_on_owner_thread():
    executor = MLXExecutor()
    try:
        def check_inside_worker():
            assert executor.is_on_owner_thread()
            # Nested run() should execute inline
            nested_tid = executor.run(threading.get_ident)
            assert nested_tid == threading.get_ident()
            return True

        assert not executor.is_on_owner_thread()
        assert executor.run(check_inside_worker) is True
    finally:
        executor.shutdown(wait=True)


@pytest.mark.anyio
async def test_executor_arun_from_event_loop():
    executor = MLXExecutor()
    try:
        main_tid = threading.get_ident()

        def compute(x, y):
            return x + y, threading.get_ident()

        res, task_tid = await executor.arun(compute, 10, 20)
        assert res == 30
        assert task_tid != main_tid
        assert task_tid == executor.owner_thread_id
    finally:
        executor.shutdown(wait=True)


@pytest.mark.anyio
async def test_executor_arun_inline_async():
    executor = MLXExecutor(inline=True)
    assert executor.is_inline

    async def async_fn(x):
        await asyncio.sleep(0.01)
        return x * 2

    res = await executor.arun(async_fn, 21)
    assert res == 42


def test_executor_bind_streams_runs_on_worker():
    executor = MLXExecutor()
    try:
        bind_tids = []

        def fake_bind():
            bind_tids.append(threading.get_ident())

        executor.bind_streams(bind_fn=fake_bind)
        assert len(bind_tids) == 1
        assert bind_tids[0] == executor.owner_thread_id
    finally:
        executor.shutdown(wait=True)


def test_executor_unowned_worker_not_shut_down():
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        executor = MLXExecutor(worker=worker, owns_worker=False)
        assert executor.worker is worker

        executor.shutdown(wait=True)
        # Worker should still be alive because executor did not own it
        assert worker.submit(lambda: "alive").result() == "alive"
    finally:
        worker.shutdown(wait=True)
