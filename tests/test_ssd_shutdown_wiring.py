# SPDX-License-Identifier: Apache-2.0
"""Regression guard: SSD cache writer threads must be stopped on shutdown.

`close_ssd_tier()` / `SSDCacheTier.close()` used to be reachable only from
`Scheduler.reset()`. Neither `EngineCore.stop()` (text engine) nor
`MLLMScheduler.stop()` (MLLM engine) ever closed the SSD tier, so every
normal shutdown leaked the writer thread and silently dropped any spills
still sitting in the queue. These tests prove that `stop()` on both engines
now drains and joins the writer thread. Model-free: no real MLX model is
loaded, only a real `SSDCacheTier` (fast, disk-only).
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheTier


def _mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.decode = lambda x: " ".join(str(t) for t in x)
    tokenizer.eos_token_id = 0
    tokenizer.eos_token_ids = {0}
    return tokenizer


class _MockKVCacheLayer:
    """Minimal KVCache-shaped layer for enqueue_spill (numpy/MLX only)."""

    def __init__(self):
        self.keys = mx.random.normal((1, 2, 4, 8)).astype(mx.float16)
        self.values = mx.random.normal((1, 2, 4, 8)).astype(mx.float16)
        self.offset = 4


def _tier_with_queued_spill(tmp_path):
    """A started SSDCacheTier with one spill already enqueued."""
    tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path)))
    tier.start_writer()
    enqueued = tier.enqueue_spill(
        (1, 2, 3, 4), [_MockKVCacheLayer()], memory_bytes=1024
    )
    assert enqueued
    return tier


class TestEngineCoreStopClosesSSDTier:
    """Text engine: EngineCore.stop() must close the scheduler's SSD tier."""

    @pytest.mark.anyio
    async def test_start_recreates_configured_ssd_tier(self):
        from vllm_mlx.engine_core import EngineCore

        engine = object.__new__(EngineCore)
        engine._running = False
        engine._task = None

        class RestartScheduler:
            def __init__(self):
                self.ensure_calls = 0

            def ensure_ssd_tier(self):
                self.ensure_calls += 1

        async def empty_loop():
            engine._running = False

        engine.scheduler = RestartScheduler()
        engine._engine_loop = empty_loop

        await engine.start()
        await engine._task

        assert engine.scheduler.ensure_calls == 1

    @pytest.mark.anyio
    async def test_stop_joins_writer_thread_and_clears_tier(self, tmp_path):
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore(MagicMock(), _mock_tokenizer())

        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path)))
        tier.start_writer()
        writer_thread = tier._writer_thread
        engine.scheduler._ssd_tier = tier

        assert writer_thread.is_alive()

        await engine.stop()

        assert not writer_thread.is_alive()
        assert engine.scheduler._ssd_tier is None

    @pytest.mark.anyio
    async def test_stop_is_a_noop_when_no_ssd_tier(self):
        """No SSD tier configured -> stop() must not raise."""
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore(MagicMock(), _mock_tokenizer())
        assert engine.scheduler._ssd_tier is None

        await engine.stop()  # should not raise

        assert engine.scheduler._ssd_tier is None

    @pytest.mark.anyio
    async def test_stop_flushes_queued_spill(self, tmp_path):
        """A spill still sitting in the queue at shutdown must be written,
        not silently dropped."""
        from vllm_mlx.engine_core import EngineCore

        engine = EngineCore(MagicMock(), _mock_tokenizer())
        tier = _tier_with_queued_spill(tmp_path)
        engine.scheduler._ssd_tier = tier

        await engine.stop()

        assert tier._stats.spill_count == 1

    @pytest.mark.anyio
    async def test_stop_keeps_event_loop_responsive_while_ssd_closes(self):
        from vllm_mlx.engine_core import EngineCore

        engine = object.__new__(EngineCore)
        engine._running = False
        engine._task = None
        release_close = threading.Event()
        close_finished = threading.Event()
        heartbeat_ran_during_close = False

        class BlockingScheduler:
            def _close_batch_generator(self):
                pass

            def close_ssd_tier(self):
                assert release_close.wait(timeout=2.0)
                close_finished.set()

        async def heartbeat():
            nonlocal heartbeat_ran_during_close
            await asyncio.sleep(0)
            heartbeat_ran_during_close = not close_finished.is_set()

        engine.scheduler = BlockingScheduler()
        heartbeat_task = asyncio.create_task(heartbeat())
        release_timer = threading.Timer(0.1, release_close.set)
        release_timer.start()

        try:
            await engine.stop()
            await heartbeat_task
        finally:
            release_close.set()
            release_timer.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()

        assert heartbeat_ran_during_close


class TestMLLMSchedulerStopClosesSSDTier:
    """MLLM engine: MLLMScheduler.stop() must close its own SSD tier."""

    @staticmethod
    def _bare_scheduler():
        """Construct an MLLMScheduler without running its heavy __init__
        (mirrors tests/test_mllm_ssd_spill.py's helper of the same name)."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched._running = False
        sched._processing_task = None
        sched.batch_generator = None
        return sched

    @pytest.mark.anyio
    async def test_stop_calls_close_on_ssd_tier(self):
        sched = self._bare_scheduler()

        class FakeTier:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        tier = FakeTier()
        sched._ssd_tier = tier

        await sched.stop()

        assert tier.closed is True
        assert sched._ssd_tier is None

    @pytest.mark.anyio
    async def test_stop_keeps_event_loop_responsive_while_ssd_closes(self):
        sched = self._bare_scheduler()
        release_close = threading.Event()
        close_finished = threading.Event()
        heartbeat_ran_during_close = False

        class BlockingTier:
            def close(self):
                assert release_close.wait(timeout=2.0)
                close_finished.set()

        async def heartbeat():
            nonlocal heartbeat_ran_during_close
            await asyncio.sleep(0)
            heartbeat_ran_during_close = not close_finished.is_set()

        sched._ssd_tier = BlockingTier()
        heartbeat_task = asyncio.create_task(heartbeat())
        release_timer = threading.Timer(0.1, release_close.set)
        release_timer.start()

        try:
            await sched.stop()
            await heartbeat_task
        finally:
            release_close.set()
            release_timer.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()

        assert heartbeat_ran_during_close

    @pytest.mark.anyio
    async def test_stop_joins_real_writer_thread(self, tmp_path):
        sched = self._bare_scheduler()

        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path)))
        tier.start_writer()
        writer_thread = tier._writer_thread
        sched._ssd_tier = tier

        assert writer_thread.is_alive()

        await sched.stop()

        assert not writer_thread.is_alive()
        assert sched._ssd_tier is None

    @pytest.mark.anyio
    async def test_stop_is_a_noop_without_ssd_tier(self):
        sched = self._bare_scheduler()
        sched._ssd_tier = None

        await sched.stop()  # should not raise

        assert sched._ssd_tier is None

    @pytest.mark.anyio
    async def test_stop_flushes_queued_spill(self, tmp_path):
        """A spill still sitting in the queue at shutdown must be written,
        not silently dropped."""
        sched = self._bare_scheduler()
        tier = _tier_with_queued_spill(tmp_path)
        sched._ssd_tier = tier

        await sched.stop()

        assert tier._stats.spill_count == 1

    def test_init_always_sets_ssd_tier_attribute(self):
        """_ssd_tier must exist right after __init__ (before the lazy
        _ensure_batch_generator() ever runs) so stop() can safely check it
        even when no request was ever processed."""
        from types import SimpleNamespace

        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                eos_token_id=0, eos_token_ids=None, name_or_path=None
            )
        )

        sched = MLLMScheduler(SimpleNamespace(), processor, MLLMSchedulerConfig())

        assert sched._ssd_tier is None


class TestSchedulerSSDTierLifecycle:
    def test_closed_tier_is_detached_and_recreated_for_restart(self, monkeypatch):
        import vllm_mlx.scheduler as scheduler_module
        from vllm_mlx.scheduler import Scheduler

        created_tiers = []

        class FakeTier:
            def __init__(self, config):
                self.config = config
                self.started = False
                self.reconciled = False
                self.closed = False
                created_tiers.append(self)

            def start_writer(self):
                self.started = True

            def reconcile(self):
                self.reconciled = True

            def close(self):
                self.closed = True

        class FakeMemoryCache:
            def __init__(self):
                self._ssd_tier = None

            def set_ssd_tier(self, tier):
                self._ssd_tier = tier

        monkeypatch.setattr(scheduler_module, "SSDCacheTier", FakeTier)
        scheduler = object.__new__(Scheduler)
        scheduler.config = SimpleNamespace(
            ssd_cache_dir="/cache",
            ssd_cache_max_gb=2.0,
        )
        scheduler.memory_aware_cache = FakeMemoryCache()
        scheduler._ssd_tier = None

        scheduler.ensure_ssd_tier()
        first_tier = scheduler._ssd_tier
        scheduler.close_ssd_tier()

        assert first_tier.closed
        assert scheduler._ssd_tier is None
        assert scheduler.memory_aware_cache._ssd_tier is None

        scheduler.ensure_ssd_tier()

        assert len(created_tiers) == 2
        assert scheduler._ssd_tier is created_tiers[1]
        assert scheduler._ssd_tier is not first_tier
        assert scheduler._ssd_tier.started
        assert scheduler._ssd_tier.reconciled
        assert scheduler.memory_aware_cache._ssd_tier is scheduler._ssd_tier
