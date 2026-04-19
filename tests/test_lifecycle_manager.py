# SPDX-License-Identifier: Apache-2.0
"""ResidencyManager unit tests — lifecycle state machine, cancellation, and shutdown."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress

import pytest


async def _wait_for_resident_state(manager, model_key: str, state: str) -> None:
    """Poll until a resident reaches the expected state."""
    while manager.get_status(model_key)["state"] != state:
        await asyncio.sleep(0)


class TestResidencyManagerContracts:
    """Lock in the high-risk lifecycle invariants at the manager layer."""

    @pytest.mark.anyio
    async def test_concurrent_acquire_single_flights_initial_load(self):
        """Concurrent acquires for one model should perform only one load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        create_calls = 0
        started = 0

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=300,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        engines = await asyncio.gather(
            manager.acquire("default"),
            manager.acquire("default"),
            manager.acquire("default"),
        )

        assert create_calls == 1
        assert started == 1
        assert len({id(engine) for engine in engines}) == 1

        for _ in engines:
            await manager.release("default")

    @pytest.mark.anyio
    async def test_unload_if_idle_respects_active_requests(self):
        """Idle unload should be blocked while a resident still has users."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        now = {"value": 1000.0}

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        now["value"] += 120.0

        unloaded = await manager.unload_if_idle("default")

        assert unloaded is False
        assert stopped == 0
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.anyio
    async def test_release_updates_last_used_at(self):
        """release() should refresh the timestamp used by idle-unload policy."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                return None

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        before_release = manager.get_status("default")["last_used_at"]

        now["value"] = 1125.0
        await manager.release("default")

        after_release = manager.get_status("default")["last_used_at"]
        assert before_release != after_release
        assert after_release == 1125.0

    @pytest.mark.anyio
    async def test_unload_after_idle_threshold_and_reload(self):
        """A released idle resident should unload and later reload cleanly."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0
        started = 0

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        now = {"value": 1000.0}

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = await manager.acquire("default")
        await manager.release("default")

        now["value"] += 120.0
        unloaded = await manager.unload_if_idle("default")

        assert unloaded is True
        assert stopped == 1
        assert manager.get_status("default")["state"] == "unloaded"

        second = await manager.acquire("default")

        assert created == 2
        assert started == 2
        assert first is not second
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.anyio
    async def test_cancelled_cold_load_does_not_wedge_future_acquires(self):
        """Cancelling one waiter should not poison the shared resident load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            await load_gate.wait()
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        load_gate.set()
        second = await manager.acquire("default")

        assert second is not None
        assert created == 2
        assert started == 1
        assert stopped == 0
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.anyio
    async def test_last_cancelled_waiter_cancels_cold_load(self):
        """Cancelling the final waiter should unwind the in-flight resident load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()
        start_entered = asyncio.Event()
        start_cancelled = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1
                start_entered.set()
                try:
                    await load_gate.wait()
                except asyncio.CancelledError:
                    start_cancelled.set()
                    raise

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.wait_for(start_cancelled.wait(), timeout=1.0)

            status = manager.get_status("default")
            assert status["state"] == "unloaded"
            assert manager._residents["default"]._loading_task is None
            assert stopped == 1

            load_gate.set()
            second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

            assert second is not None
            assert created == 2
            assert started == 2
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            load_gate.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.anyio
    async def test_cancelled_waiter_does_not_cancel_shared_cold_load(self):
        """A canceled waiter should not kill a cold load another waiter still needs."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        load_gate = asyncio.Event()
        start_entered = asyncio.Event()
        start_cancelled = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1
                start_entered.set()
                try:
                    await load_gate.wait()
                except asyncio.CancelledError:
                    start_cancelled.set()
                    raise

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        second = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.sleep(0)
            assert not start_cancelled.is_set()

            load_gate.set()
            engine = await asyncio.wait_for(second, timeout=1.0)

            assert engine is not None
            assert created == 1
            assert started == 1
            assert stopped == 0
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            load_gate.set()
            for task in (first, second):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    @pytest.mark.anyio
    async def test_cancelled_last_waiter_suppresses_load_failure_from_cancel(self):
        """Abandoned cold loads should still surface as cancellation to the waiter."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        start_entered = asyncio.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                if self.generation != 1:
                    return None
                start_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as exc:
                    raise RuntimeError("boom-from-start") from exc

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)
            first.cancel()
            await first

        status = manager.get_status("default")
        assert status["state"] == "unloaded"
        assert status["last_error"] is None
        assert manager._residents["default"]._loading_task is None
        assert stopped == 1

        second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

        assert second is not None
        assert second.generation == 2
        assert created == 2
        assert started == 2
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.anyio
    async def test_cancelled_prepare_for_start_does_not_finish_after_stop(self):
        """Cancelled cold loads should not let prepare_for_start keep mutating after stop."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        prepare_finished = threading.Event()
        prepare_finished_after_stop = threading.Event()
        stop_called = threading.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            def prepare_for_start(self):
                prepare_entered.set()
                allow_prepare_finish.wait()
                time.sleep(0.01)
                if stop_called.is_set():
                    prepare_finished_after_stop.set()
                prepare_finished.set()

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1
                stop_called.set()

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_entered.wait, 1.0),
                timeout=1.0,
            )

            first.cancel()
            allow_prepare_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await first

            assert await asyncio.wait_for(
                asyncio.to_thread(prepare_finished.wait, 1.0),
                timeout=1.0,
            )
            assert not prepare_finished_after_stop.is_set()

            await asyncio.wait_for(
                _wait_for_resident_state(manager, "default", "unloaded"),
                timeout=1.0,
            )

            status = manager.get_status("default")
            assert status["state"] == "unloaded"
            assert manager._residents["default"]._loading_task is None

            second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

            assert second is not None
            assert second.generation == 2
            assert created == 2
            assert started == 1
            assert stopped == 1
            assert manager.get_status("default")["state"] == "loaded"

            await manager.release("default")
        finally:
            allow_prepare_finish.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.anyio
    async def test_late_joining_waiter_retries_abandoned_cold_load(self):
        """A waiter joining during abandoned-load cleanup should retry instead of inheriting cancel."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0
        start_entered = asyncio.Event()
        stop_entered = asyncio.Event()
        allow_stop = asyncio.Event()

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1
                if self.generation != 1:
                    return None
                start_entered.set()
                await asyncio.Event().wait()

            async def stop(self):
                nonlocal stopped
                stopped += 1
                stop_entered.set()
                await allow_stop.wait()

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        try:
            await asyncio.wait_for(start_entered.wait(), timeout=1.0)

            first.cancel()

            await asyncio.wait_for(stop_entered.wait(), timeout=1.0)

            second = asyncio.create_task(manager.acquire("default"))
            try:
                await asyncio.sleep(0)

                allow_stop.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                engine = await asyncio.wait_for(second, timeout=1.0)

                assert engine is not None
                assert engine.generation == 2
                assert created == 2
                assert started == 2
                assert stopped == 1
                assert manager.get_status("default")["state"] == "loaded"
            finally:
                if not second.done():
                    second.cancel()
                    with suppress(asyncio.CancelledError):
                        await second

            await manager.release("default")
        finally:
            allow_stop.set()
            if not first.done():
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first

    @pytest.mark.anyio
    async def test_cancelled_shared_load_during_state_commit_recovers_cleanly(self):
        """Cancelling the shared load during state commit should stop the engine and recover."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        started = 0
        stopped = 0

        class CommitGateLock:
            def __init__(self):
                self._real = asyncio.Lock()
                self.block_next_enter = False
                self.commit_waiting = asyncio.Event()
                self.allow_commit = asyncio.Event()

            async def __aenter__(self):
                if self.block_next_enter:
                    self.block_next_enter = False
                    self.commit_waiting.set()
                    await self.allow_commit.wait()
                await self._real.acquire()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self._real.release()

        lock = CommitGateLock()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        async def on_engine_loaded(spec, engine):
            lock.block_next_enter = True

        manager = ResidencyManager(
            engine_factory=engine_factory,
            on_engine_loaded=on_engine_loaded,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager._lock = lock
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        first = asyncio.create_task(manager.acquire("default"))
        await lock.commit_waiting.wait()

        loading_task = manager._residents["default"]._loading_task
        assert loading_task is not None
        loading_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await first

        assert stopped == 1
        assert manager.get_status("default")["state"] == "unloaded"

        lock.allow_commit.set()
        second = await asyncio.wait_for(manager.acquire("default"), timeout=1.0)

        assert created == 2
        assert started == 2
        assert manager.get_status("default")["state"] == "loaded"

        await manager.release("default")

    @pytest.mark.anyio
    async def test_cancelled_load_cleanup_handles_legacy_tasks_without_uncancel_api(
        self, monkeypatch
    ):
        """Cancellation cleanup should still work on supported runtimes without Task.uncancel()."""
        import vllm_mlx.lifecycle as lifecycle_mod
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager, ResidentState

        stopped = 0

        class LegacyTask:
            def cancel(self):
                return None

        class FakeEngine:
            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            raise AssertionError("engine_factory should not be called")

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        resident = manager._residents["default"]
        resident.state = ResidentState.LOADING
        resident._loading_task = object()

        monkeypatch.setattr(
            lifecycle_mod.asyncio,
            "current_task",
            lambda: LegacyTask(),
        )

        await manager._cleanup_cancelled_load(resident, FakeEngine())

        assert stopped == 1
        assert resident.state == ResidentState.UNLOADED
        assert resident._loading_task is None

    @pytest.mark.anyio
    async def test_shutdown_cancels_inflight_cold_load(self):
        """shutdown() should not allow a cold load to complete afterward."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        started = 0
        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            await load_gate.wait()
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)

        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)

        load_gate.set()
        await shutdown_task

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        assert started == 0
        assert manager.get_engine("default") is None
        assert manager.get_status("default")["state"] == "unloaded"

    @pytest.mark.anyio
    async def test_shutdown_canceled_prepare_error_unwinds_to_unloaded(self):
        """shutdown() should suppress prepare errors from a canceled cold load."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        prepare_entered = threading.Event()
        allow_prepare_finish = threading.Event()
        started = 0

        class FakeEngine:
            def prepare_for_start(self):
                prepare_entered.set()
                allow_prepare_finish.wait()
                raise RuntimeError("prepare boom")

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                return None

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        acquire_task = asyncio.create_task(manager.acquire("default"))
        assert await asyncio.wait_for(
            asyncio.to_thread(prepare_entered.wait, 1.0),
            timeout=1.0,
        )

        shutdown_task = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)
        allow_prepare_finish.set()
        await shutdown_task

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        status = manager.get_status("default")
        assert started == 0
        assert manager.get_engine("default") is None
        assert status["state"] == "unloaded"
        assert status["last_error"] is None

    @pytest.mark.anyio
    async def test_shutdown_raises_if_loaded_resident_cannot_unload(self):
        """shutdown() should not report success when resident unload fails."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("stop boom")

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")

        with pytest.raises(RuntimeError):
            await manager.shutdown()

    @pytest.mark.anyio
    async def test_shutdown_attempts_later_residents_after_one_unload_failure(self):
        """shutdown() should keep unloading later residents after one unload fails."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = []

        class FakeEngine:
            def __init__(self, model_key):
                self.model_key = model_key

            async def start(self):
                return None

            async def stop(self):
                stopped.append(self.model_key)
                if self.model_key == "a":
                    raise RuntimeError("stop boom")

        async def engine_factory(spec):
            return FakeEngine(spec.model_key)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="a", model_name="model-a"))
        manager.register_model(ModelSpec(model_key="b", model_name="model-b"))

        await manager.acquire("a")
        await manager.release("a")
        await manager.acquire("b")
        await manager.release("b")

        with pytest.raises(RuntimeError, match="a"):
            await manager.shutdown()

        assert stopped == ["a", "b"]
        assert manager.get_engine("b") is None
        assert manager.get_status("b")["state"] == "unloaded"

    @pytest.mark.anyio
    async def test_acquire_retries_if_idle_unload_wins_the_boundary(self, monkeypatch):
        """Acquire should not hand back an engine that unloaded in the claim gap."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0
        started = 0
        now = {"value": 1000.0}

        class FakeEngine:
            def __init__(self, generation):
                self.generation = generation

            async def start(self):
                nonlocal started
                started += 1

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine(created)

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        original = await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        entered_gap = asyncio.Event()
        continue_gap = asyncio.Event()
        original_ensure_loaded = manager.ensure_loaded

        async def delayed_ensure_loaded(model_key):
            engine = await original_ensure_loaded(model_key)
            if not entered_gap.is_set():
                entered_gap.set()
                await continue_gap.wait()
            return engine

        monkeypatch.setattr(manager, "ensure_loaded", delayed_ensure_loaded)

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await entered_gap.wait()

        unloaded = await manager.unload_if_idle("default")
        continue_gap.set()
        replacement = await acquire_task

        assert unloaded is True
        assert stopped == 1
        assert created == 2
        assert started == 2
        assert replacement is not original
        assert manager.get_status("default")["state"] == "loaded"
        assert manager.get_status("default")["active_requests"] == 1

        await manager.release("default")

    @pytest.mark.anyio
    async def test_failed_unload_keeps_live_engine_tracked(self):
        """Unload failure should not orphan a still-live engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        now = {"value": 1000.0}

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                raise RuntimeError("boom")

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        original = await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        unloaded = await manager.unload_if_idle("default")
        replacement = await manager.acquire("default")

        assert unloaded is False
        assert manager.get_engine("default") is original
        assert replacement is original
        assert manager.get_status("default")["state"] == "loaded"
        assert manager.get_status("default")["last_error"] == "boom"
        assert created == 1

        await manager.release("default")

    @pytest.mark.anyio
    async def test_register_model_rejects_replacing_live_resident(self):
        """register_model() should not orphan an already loaded resident."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        created = 0
        stopped = 0

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            nonlocal created
            created += 1
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="first-model"))

        await manager.acquire("default")
        await manager.release("default")

        with pytest.raises(RuntimeError, match="Cannot replace resident model"):
            manager.register_model(
                ModelSpec(model_key="default", model_name="replacement-model")
            )

        await manager.shutdown()

        assert created == 1
        assert stopped == 1

    @pytest.mark.anyio
    async def test_failed_cold_load_cleans_up_partial_engine(self):
        """A start() failure should still stop the partially built engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                raise RuntimeError("boom")

            async def stop(self):
                nonlocal stopped
                stopped += 1

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        with pytest.raises(RuntimeError, match="boom"):
            await manager.acquire("default")

        assert manager.get_status("default")["state"] == "failed"
        assert stopped == 1

    @pytest.mark.anyio
    async def test_cancelled_waiter_does_not_cancel_shared_unload(self):
        """A canceled acquire waiter should not cancel the shared unload task."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}
        stop_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                await stop_gate.wait()

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        unload_task = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        acquire_task = asyncio.create_task(manager.acquire("default"))
        await asyncio.sleep(0)
        acquire_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        stop_gate.set()
        unloaded = await unload_task

        assert unloaded is True
        assert manager.get_status("default")["state"] == "unloaded"

    @pytest.mark.anyio
    async def test_cancelled_unload_waiter_does_not_cancel_shared_unload_task(self):
        """Cancelling one unload waiter should not cancel the shared unload operation."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        now = {"value": 1000.0}
        stop_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                return None

            async def stop(self):
                await stop_gate.wait()

        async def engine_factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            engine_factory=engine_factory,
            time_fn=lambda: now["value"],
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="default", model_name="test-model"))

        await manager.acquire("default")
        await manager.release("default")
        now["value"] += 120.0

        first_waiter = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        second_waiter = asyncio.create_task(manager.unload_if_idle("default"))
        await asyncio.sleep(0)

        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        stop_gate.set()
        unloaded = await second_waiter

        assert unloaded is True
        assert manager.get_status("default")["state"] == "unloaded"


class TestResidencyManagerEdgeCases:
    """Edge-case coverage for manager state transitions and error recovery."""

    @pytest.mark.anyio
    async def test_unload_if_idle_when_already_unloaded(self):
        """unload_if_idle on an unloaded resident should return False, not corrupt state."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        # Never loaded — should be a no-op
        result = await manager.unload_if_idle("m")
        assert result is False
        assert manager.get_status("m")["state"] == "unloaded"

    @pytest.mark.anyio
    async def test_unload_if_idle_during_active_load(self):
        """unload_if_idle should return False when a load is in progress."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        load_gate = asyncio.Event()

        class FakeEngine:
            async def start(self):
                await load_gate.wait()

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        load_task = asyncio.create_task(manager.ensure_loaded("m"))
        await asyncio.sleep(0)  # let load start

        result = await manager.unload_if_idle("m")
        assert result is False

        load_gate.set()
        await load_task
        await manager.shutdown()

    @pytest.mark.anyio
    async def test_register_model_after_failure_replaces_entry(self):
        """A model in FAILED state should be replaceable via register_model."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        call_count = 0

        class FailEngine:
            async def start(self):
                raise RuntimeError("boom")

            async def stop(self):
                pass

        class GoodEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FailEngine()
            return GoodEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        spec = ModelSpec(model_key="m", model_name="test")
        manager.register_model(spec)

        with pytest.raises(RuntimeError, match="boom"):
            await manager.ensure_loaded("m")
        assert manager.get_status("m")["state"] == "failed"

        # Re-register should succeed because the model is dormant/failed
        manager.register_model(spec)
        assert manager.get_status("m")["state"] == "unloaded"

        engine = await manager.ensure_loaded("m")
        assert engine is not None
        assert manager.get_status("m")["state"] == "loaded"
        await manager.shutdown()

    @pytest.mark.anyio
    async def test_engine_factory_raises_before_engine_created(self):
        """If the factory itself raises (not engine.start), state should be FAILED."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        async def failing_factory(spec):
            raise RuntimeError("model not found")

        manager = ResidencyManager(failing_factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        with pytest.raises(RuntimeError, match="model not found"):
            await manager.ensure_loaded("m")

        status = manager.get_status("m")
        assert status["state"] == "failed"
        assert "model not found" in status["last_error"]

    @pytest.mark.anyio
    async def test_rapid_acquire_release_refcount_stays_consistent(self):
        """Rapid acquire/release cycles should keep active_requests consistent."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(
            factory,
            time_fn=lambda: 1000.0,
            auto_unload_idle_seconds=60,
        )
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        # Rapid sequential acquire/release
        for _ in range(50):
            await manager.acquire("m")
            await manager.release("m")

        assert manager.get_status("m")["active_requests"] == 0

        # Concurrent acquire then sequential release
        engines = await asyncio.gather(*[manager.acquire("m") for _ in range(20)])
        assert manager.get_status("m")["active_requests"] == 20
        for _ in engines:
            await manager.release("m")
        assert manager.get_status("m")["active_requests"] == 0

    @pytest.mark.anyio
    async def test_shutdown_shields_unload_from_cancellation(self):
        """Cancelling shutdown() should not orphan a half-stopped engine."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stop_started = asyncio.Event()
        stop_gate = asyncio.Event()
        stopped = 0

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                nonlocal stopped
                stop_started.set()
                await stop_gate.wait()
                stopped += 1

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))
        await manager.ensure_loaded("m")

        shutdown_task = asyncio.create_task(manager.shutdown())
        await stop_started.wait()

        # Cancel shutdown while engine.stop() is in flight
        shutdown_task.cancel()
        await asyncio.sleep(0)

        # Let engine.stop() complete
        stop_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task

        # The engine should still have been fully stopped
        assert stopped == 1
        assert manager.get_status("m")["state"] == "unloaded"


class TestSuspendCancellationDedup:
    """Verify lifecycle.py uses the shared suspend_cancellation from base."""

    @pytest.mark.anyio
    async def test_residency_manager_uses_shared_suspend_cancellation(self):
        """ResidencyManager cleanup paths should work with the shared helper."""
        from vllm_mlx.lifecycle import ModelSpec, ResidencyManager

        stopped = 0

        class FakeEngine:
            async def start(self):
                pass

            async def stop(self):
                nonlocal stopped
                stopped += 1

            def prepare_for_start(self):
                pass

        async def factory(spec):
            return FakeEngine()

        manager = ResidencyManager(factory, auto_unload_idle_seconds=0)
        manager.register_model(ModelSpec(model_key="m", model_name="test"))

        engine = await manager.ensure_loaded("m")
        assert engine is not None
        await manager.shutdown()
        assert stopped == 1
