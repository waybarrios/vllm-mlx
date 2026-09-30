# SPDX-License-Identifier: Apache-2.0
"""Tests for shared engine helpers in vllm_mlx/engine/base.py."""

import asyncio
import gc
import time

import anyio
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
        """Regression test for CPython 3.14+ specifically: on 3.14+,
        asyncio.shield() unconditionally reassigns the shielded task's
        done-callback to an internal handler that logs any later exception
        via loop.call_exception_handler() once the shield's own wrapper
        future is cancelled -- even if the caller retrieves that exception
        itself afterward. Test runners that fail tests on any logged loop
        exception (e.g. pytest-anyio) then report a spurious failure despite
        the exception being fully handled. On 3.10-3.13 this does not
        reproduce (see shield_task()'s docstring), so this test only
        exercises the 3.14+ code path meaningfully -- it still runs on
        3.10-3.13 (asserting the always-true absence of the bug there), but
        the ``test-python-3-14`` CI job is what actually covers the
        behavior it targets. shield_task() must never trigger the loop's
        exception handler in this scenario, on any supported version.
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

    @pytest.mark.anyio
    async def test_already_done_task_returns_result_via_fast_path(self):
        """asyncio.shield() returns an already-done task immediately instead
        of adding a done-callback and waiting an extra event loop turn.
        shield_task() must match that (PR #643 review, blocker 2) -- a
        scheduled cancellation racing the extra turn could otherwise replace
        an already-available result with CancelledError."""
        task = asyncio.create_task(asyncio.sleep(0, result="done"))
        while not task.done():
            await asyncio.sleep(0)
        assert await shield_task(task) == "done"

    @pytest.mark.anyio
    async def test_already_done_task_reraises_exception_via_fast_path(self):
        async def boom():
            raise ValueError("kaboom")

        task = asyncio.create_task(boom())
        while not task.done():
            await asyncio.sleep(0)
        with pytest.raises(ValueError, match="kaboom"):
            await shield_task(task)

    @pytest.mark.anyio
    async def test_already_cancelled_task_reraises_cancelled_via_fast_path(self):
        task = asyncio.create_task(asyncio.sleep(10))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await shield_task(task)

    @pytest.mark.anyio
    async def test_repeated_cancellation_does_not_leak_propagate_callbacks(self):
        """Regression for PR #643 review blocker 1: every shield_task() call
        left its `_propagate` done-callback attached to `task` even after the
        awaiter moved on, so a retry loop that re-shields the same still-
        running task after each cancellation (see run_blocking_startup_work)
        accumulated one abandoned callback per cancellation -- reported by
        review as 8,516 callbacks and ~2x cancellation latency under
        repeated AnyIO cancellation. Growth must be bounded: at most one
        _propagate callback attached to `task` at a time, however many
        times it gets re-shielded and re-cancelled.
        """
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(work())
        await started.wait()

        for _ in range(50):
            scope = anyio.CancelScope()
            with scope:
                scope.cancel()
                await shield_task(task)
            assert scope.cancelled_caught
            assert not task.done()
            # `_callbacks` is None (not an empty list) once every callback
            # has been removed -- CPython lazily allocates it.
            assert len(task._callbacks or ()) <= 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.anyio
    async def test_concurrent_shielders_of_same_task_all_receive_result(self):
        """shield_task() must support multiple concurrent callers shielding
        the same task -- real asyncio.shield() does this (each call gets its
        own outer future), and shield_task()'s docstring claims equivalence.
        The single per-task done-callback fans a result out to every
        currently-registered waiter, not just the first one."""
        task = asyncio.create_task(asyncio.sleep(0, result="done"))

        results = await asyncio.gather(
            shield_task(task), shield_task(task), shield_task(task)
        )
        assert results == ["done", "done", "done"]

    @pytest.mark.anyio
    async def test_concurrent_shielders_of_same_task_all_receive_exception(self):
        async def boom():
            await asyncio.sleep(0)
            raise ValueError("kaboom")

        task = asyncio.create_task(boom())

        async def shield_and_capture():
            try:
                await shield_task(task)
                return None
            except ValueError as exc:
                return exc

        results = await asyncio.gather(shield_and_capture(), shield_and_capture())
        assert all(isinstance(r, ValueError) for r in results)

    @pytest.mark.anyio
    async def test_cancelling_one_shielder_does_not_affect_concurrent_shielders(self):
        """One caller cancelling its own shield_task() await must not
        disturb a second, concurrent caller shielding the exact same task --
        proving the shared per-task waiter list removes only the cancelled
        caller's own entry, leaving the other waiter registered."""
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(0.05)
            return "finished"

        task = asyncio.create_task(work())
        await started.wait()

        async def consume():
            return await shield_task(task)

        consumer_a = asyncio.create_task(consume())
        consumer_b = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let both register as waiters

        consumer_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer_a

        assert await consumer_b == "finished"
