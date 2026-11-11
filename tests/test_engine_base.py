# SPDX-License-Identifier: Apache-2.0
"""Tests for shared engine helpers in vllm_mlx/engine/base.py."""

import ast
import asyncio
import gc
import inspect
import json
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import MISSING, asdict, fields
from typing import get_type_hints

import anyio
import pytest

from vllm_mlx.engine.base import BaseEngine, GenerationOutput, shield_task


class TestGenerationOutput:
    @pytest.mark.parametrize("name", ["mtp_drafts", "mtp_accepted"])
    def test_mtp_counter_has_one_source_declaration(self, name):
        # Runtime annotations deduplicate names, hiding the static contract error.
        declaration = ast.parse(inspect.getsource(GenerationOutput)).body[0]
        counters = [
            node.target.id
            for node in declaration.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        ]
        assert counters.count(name) == 1

    def test_field_and_positional_constructor_contract(self):
        expected_names = (
            "text",
            "tokens",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
            "mtp_drafts",
            "mtp_accepted",
            "new_text",
            "finished",
            "specprefill_outcome",
        )
        declared = fields(GenerationOutput)
        assert tuple(item.name for item in declared) == expected_names
        assert tuple(item.type for item in declared) == (
            str,
            list[int],
            int,
            int,
            str | None,
            int,
            int,
            str,
            bool,
            "SpecPrefillOutcome | None",
        )
        assert declared[0].default is MISSING
        assert declared[1].default is MISSING
        assert declared[1].default_factory is list
        assert tuple(item.default for item in declared[2:]) == (
            0,
            0,
            "stop",
            0,
            0,
            "",
            True,
            None,
        )
        assert all(item.default_factory is MISSING for item in declared[2:])
        assert declared[0].default_factory is MISSING
        assert all(item.init and item.repr and item.compare for item in declared)
        assert all(not item.metadata and not item.kw_only for item in declared)
        signature = inspect.signature(GenerationOutput)
        assert tuple(signature.parameters) == expected_names
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert GenerationOutput.__match_args__ == expected_names
        with pytest.raises(TypeError, match="text"):
            GenerationOutput()

    def test_defaults_serialization_and_independent_token_lists(self):
        first = GenerationOutput("hello")
        second = GenerationOutput("hello")
        expected = {
            "text": "hello",
            "tokens": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "stop",
            "mtp_drafts": 0,
            "mtp_accepted": 0,
            "new_text": "",
            "finished": True,
            "specprefill_outcome": None,
        }
        assert asdict(first) == expected
        assert json.dumps(asdict(first)) == (
            '{"text": "hello", "tokens": [], "prompt_tokens": 0, '
            '"completion_tokens": 0, "finish_reason": "stop", "mtp_drafts": 0, '
            '"mtp_accepted": 0, "new_text": "", "finished": true, '
            '"specprefill_outcome": null}'
        )
        assert repr(first) == (
            "GenerationOutput(text='hello', tokens=[], prompt_tokens=0, "
            "completion_tokens=0, finish_reason='stop', mtp_drafts=0, "
            "mtp_accepted=0, new_text='', finished=True, specprefill_outcome=None)"
        )
        assert first == second
        assert first.tokens is not second.tokens
        first.tokens.append(7)
        assert second.tokens == []
        assert first != second

    def test_populated_positional_and_keyword_values_serialize_unchanged(self):
        positional = GenerationOutput(
            "hello", [7, 8], 3, 2, None, 9, 4, "lo", False, None
        )
        keyword = GenerationOutput(
            text="hello",
            tokens=[7, 8],
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason=None,
            mtp_drafts=9,
            mtp_accepted=4,
            new_text="lo",
            finished=False,
            specprefill_outcome=None,
        )
        assert positional == keyword
        assert asdict(positional) == {
            "text": "hello",
            "tokens": [7, 8],
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": None,
            "mtp_drafts": 9,
            "mtp_accepted": 4,
            "new_text": "lo",
            "finished": False,
            "specprefill_outcome": None,
        }
        assert json.dumps(asdict(positional)) == (
            '{"text": "hello", "tokens": [7, 8], "prompt_tokens": 3, '
            '"completion_tokens": 2, "finish_reason": null, "mtp_drafts": 9, '
            '"mtp_accepted": 4, "new_text": "lo", "finished": false, '
            '"specprefill_outcome": null}'
        )
        keyword.mtp_accepted = 3
        assert positional != keyword


class _EngineWithoutStreams(BaseEngine):
    @property
    def model_name(self):
        return "test-engine"

    @property
    def is_mllm(self):
        return False

    @property
    def tokenizer(self):
        return None

    async def start(self):
        pass

    async def stop(self):
        pass

    async def generate(self, *args, **kwargs):
        return GenerationOutput(text="complete")

    async def chat(self, *args, **kwargs):
        return GenerationOutput(text="complete")


class _StreamingEngine(_EngineWithoutStreams):
    def __init__(self, *, error=None, block=False):
        self.chunks = [
            GenerationOutput(text="first", new_text="first", finished=False),
            GenerationOutput(text="first second", new_text=" second"),
        ]
        self.error = error
        self.block = block
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.cleanups = 0

    async def _outputs(self) -> AsyncIterator[GenerationOutput]:
        try:
            yield self.chunks[0]
            if self.error is not None:
                raise self.error
            if self.block:
                self.waiting.set()
                await self.release.wait()
            yield self.chunks[1]
        finally:
            self.cleanups += 1

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        async with aclosing(self._outputs()) as outputs:
            async for output in outputs:
                yield output

    async def stream_chat(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        async with aclosing(self._outputs()) as outputs:
            async for output in outputs:
                yield output


def _engine_stream(engine: BaseEngine, method: str) -> AsyncIterator[GenerationOutput]:
    if method == "stream_generate":
        return engine.stream_generate("prompt")
    return engine.stream_chat([{"role": "user", "content": "prompt"}])


class TestBaseEngineStreaming:
    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    def test_abstract_stream_is_an_iterator_factory_not_a_coroutine(self, method):
        declaration = getattr(BaseEngine, method)
        assert not inspect.iscoroutinefunction(declaration)
        assert not inspect.isasyncgenfunction(declaration)
        assert declaration.__isabstractmethod__
        assert get_type_hints(declaration)["return"] == AsyncIterator[GenerationOutput]

    @pytest.mark.parametrize("method", ["start", "stop", "generate", "chat"])
    def test_nonstream_methods_remain_abstract_coroutines(self, method):
        declaration = getattr(BaseEngine, method)
        assert inspect.iscoroutinefunction(declaration)
        assert declaration.__isabstractmethod__

    def test_both_stream_implementations_are_required(self):
        assert _EngineWithoutStreams.__abstractmethods__ == {
            "stream_generate",
            "stream_chat",
        }
        with pytest.raises(TypeError, match="abstract"):
            _EngineWithoutStreams()

    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    def test_one_stream_implementation_does_not_satisfy_the_other(self, method):
        partial_engine = type(
            "PartialEngine",
            (_EngineWithoutStreams,),
            {method: getattr(_StreamingEngine, method)},
        )
        other = "stream_chat" if method == "stream_generate" else "stream_generate"
        assert partial_engine.__abstractmethods__ == {other}
        with pytest.raises(TypeError, match=other):
            partial_engine()

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    async def test_direct_iteration_preserves_chunk_identity_and_order(self, method):
        engine = _StreamingEngine()
        stream = _engine_stream(engine, method)
        assert inspect.isasyncgen(stream)
        assert not inspect.isawaitable(stream)
        received = [output async for output in stream]
        assert len(received) == 2
        assert received[0] is engine.chunks[0]
        assert received[1] is engine.chunks[1]
        assert engine.cleanups == 1

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    async def test_error_after_first_chunk_propagates_and_cleans_up(self, method):
        error = ValueError("stream failed")
        engine = _StreamingEngine(error=error)
        stream = _engine_stream(engine, method)
        assert await anext(stream) is engine.chunks[0]
        with pytest.raises(ValueError, match="stream failed") as raised:
            await anext(stream)
        assert raised.value is error
        assert engine.cleanups == 1

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    async def test_closing_partially_consumed_generator_cleans_up(self, method):
        engine = _StreamingEngine()
        async with aclosing(_engine_stream(engine, method)) as stream:
            assert await anext(stream) is engine.chunks[0]
            assert engine.cleanups == 0
        assert engine.cleanups == 1
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["stream_generate", "stream_chat"])
    async def test_cancelling_partially_consumed_generator_cleans_up(self, method):
        engine = _StreamingEngine(block=True)
        received = []

        async def consume():
            async for output in _engine_stream(engine, method):
                received.append(output)

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(engine.waiting.wait(), timeout=1)
            assert len(received) == 1
            assert received[0] is engine.chunks[0]
            assert engine.cleanups == 0
        finally:
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
        assert engine.cleanups == 1


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
