# SPDX-License-Identifier: Apache-2.0
"""Tests for shared engine helpers in vllm_mlx/engine/base.py."""

import asyncio
import gc
import time

import pytest

from vllm_mlx.engine.base import shield_task


class TestShieldTask:
    """shield_task() is a drop-in replacement for asyncio.shield() used
    wherever a cancelled awaiter must still let a background task run to
    completion and retrieve its result/exception deterministically (see
    SimpleEngine._run_blocking_serialized, ResidencyManager._prepare_engine_start).
    """

    @pytest.mark.anyio
    async def test_returns_result_when_not_cancelled(self):
        task = asyncio.create_task(asyncio.sleep(0, result="done"))
        assert await shield_task(task) == "done"

    @pytest.mark.anyio
    async def test_propagates_task_exception_when_not_cancelled(self):
        async def boom():
            raise ValueError("kaboom")

        task = asyncio.create_task(boom())
        with pytest.raises(ValueError, match="kaboom"):
            await shield_task(task)

    @pytest.mark.anyio
    async def test_cancelling_awaiter_raises_cancelled_without_cancelling_task(self):
        """Cancelling the coroutine awaiting shield_task() must not cancel
        the shielded task itself -- it should keep running in the background."""
        started = asyncio.Event()
        allow_finish = asyncio.Event()

        async def work():
            started.set()
            await allow_finish.wait()
            return "finished"

        task = asyncio.create_task(work())

        async def consume():
            return await shield_task(task)

        consumer = asyncio.create_task(consume())
        await started.wait()
        consumer.cancel()

        with pytest.raises(asyncio.CancelledError):
            await consumer

        assert not task.cancelled()
        assert not task.done()

        allow_finish.set()
        assert await task == "finished"

    @pytest.mark.anyio
    async def test_cancellation_does_not_trigger_loop_exception_handler(self):
        """Regression test: asyncio.shield() unconditionally reassigns the
        shielded task's done-callback to an internal handler that logs any
        later exception via loop.call_exception_handler() once the shield's
        own wrapper future is cancelled -- even if the caller retrieves that
        exception itself afterward. Test runners that fail tests on any
        logged loop exception (e.g. pytest-anyio) then report a spurious
        failure despite the exception being fully handled. shield_task()
        must never trigger the loop's exception handler in this scenario.
        """
        loop = asyncio.get_running_loop()
        logged_contexts = []
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda _loop, context: logged_contexts.append(context)
        )
        try:
            cancel_requested = asyncio.Event()

            def blocking_work():
                while not cancel_requested.is_set():
                    time.sleep(0.01)
                raise RuntimeError("expected background failure")

            task = asyncio.create_task(asyncio.to_thread(blocking_work))

            async def consume():
                try:
                    return await shield_task(task)
                except asyncio.CancelledError:
                    cancel_requested.set()
                    try:
                        await task
                    except BaseException:
                        pass
                    raise

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            consumer.cancel()

            with pytest.raises(asyncio.CancelledError):
                await consumer

            # Let the event loop process any pending callbacks (including a
            # would-be call_exception_handler invocation) before asserting.
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(original_handler)

        assert logged_contexts == []

    @pytest.mark.anyio
    async def test_cancel_before_task_exception_does_not_leak_unretrieved_exception(
        self,
    ):
        """Deterministic regression test for the race inside `_propagate`
        itself (not asyncio.shield()): if the awaiter is cancelled *before*
        `task` finishes, `waiter` is already done by the time `_propagate`
        runs. Unlike test_cancellation_does_not_trigger_loop_exception_handler
        above, this test never separately does `await task` afterward --
        emulating a retry loop that gives up rather than re-awaiting. If
        `_propagate` doesn't retrieve `task`'s exception on that
        already-done-`waiter` path, Python logs "Task exception was never
        retrieved" once `task` is garbage collected, reproducing the exact
        symptom this PR exists to eliminate via a different path.
        """
        loop = asyncio.get_running_loop()
        logged_contexts = []
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda _loop, context: logged_contexts.append(context)
        )
        try:
            release_task = asyncio.Event()

            async def work():
                await release_task.wait()
                raise RuntimeError("expected background failure")

            task = asyncio.create_task(work())

            async def consume():
                return await shield_task(task)

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0)  # let consumer start awaiting shield_task
            consumer.cancel()

            with pytest.raises(asyncio.CancelledError):
                await consumer

            # The race window: `waiter` inside shield_task is already
            # cancelled, but `task` is still running.
            assert not task.done()

            release_task.set()
            # Let `task` raise WITHOUT the caller ever awaiting it or
            # reading its exception itself.
            for _ in range(10):
                await asyncio.sleep(0)
            assert task.done()

            task = None
            consumer = None
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(original_handler)

        assert logged_contexts == []
