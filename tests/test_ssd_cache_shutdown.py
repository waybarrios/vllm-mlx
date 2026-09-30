# SPDX-License-Identifier: Apache-2.0
"""Model-free regression tests for SSD cache shutdown semantics."""

import asyncio
import queue
import threading

import pytest

from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheTier


class TestSSDCacheTierShutdown:
    def test_enqueue_spill_rejects_work_after_close(self, tmp_path):
        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path / "closed_enqueue")))
        tier.close()

        assert tier.enqueue_spill((1,), [], memory_bytes=1) is False

    def test_close_drains_spills_queued_behind_inflight_write(self, tmp_path):
        """Shutdown must process every spill queued before its sentinel."""

        class ObservedQueue(queue.Queue):
            def __init__(self):
                super().__init__(maxsize=2)
                self.shutdown_requested = threading.Event()

            def put(self, item, block=True, timeout=None):
                if item is None:
                    self.shutdown_requested.set()
                return super().put(item, block=block, timeout=timeout)

        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path / "drain_on_close")))
        tier._spill_queue = ObservedQueue()
        first_write_started = threading.Event()
        release_first_write = threading.Event()
        written_tokens = []

        def write_entry(tokens, _layer_snapshots, _memory_bytes):
            if tokens == (1,):
                first_write_started.set()
                assert release_first_write.wait(timeout=2.0)
            written_tokens.append(tokens)

        tier._write_entry = write_entry
        tier.start_writer()
        tier._spill_queue.put(((1,), [], 1))
        assert first_write_started.wait(timeout=2.0)
        tier._spill_queue.put(((2,), [], 1))

        close_thread = threading.Thread(target=tier.close)
        close_thread.start()
        assert tier._spill_queue.shutdown_requested.wait(timeout=2.0)
        release_first_write.set()
        close_thread.join(timeout=2.0)

        assert not close_thread.is_alive()
        assert written_tokens == [(1,), (2,)]
        assert tier._spill_queue.empty()

    def test_close_keeps_index_open_when_writer_does_not_stop(
        self, tmp_path, monkeypatch
    ):
        """A join timeout must preserve resources needed by a live writer."""

        class StillAliveThread:
            def __init__(self):
                self.joined_with = None
                self.alive = True

            def join(self, timeout=None):
                self.joined_with = timeout

            def is_alive(self):
                return self.alive

        class TrackingIndex:
            def __init__(self, index):
                self.index = index
                self.closed = False

            def close(self):
                self.closed = True
                self.index.close()

        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path / "writer_timeout")))
        writer_thread = StillAliveThread()
        tracking_index = TrackingIndex(tier._index)
        tier._writer_thread = writer_thread
        tier._index = tracking_index
        monkeypatch.setattr(tier, "_WRITER_JOIN_TIMEOUT_S", 0.01, raising=False)

        try:
            with pytest.raises(TimeoutError, match="writer thread"):
                tier.close()

            assert 0 < writer_thread.joined_with <= 0.01
            assert tier._writer_thread is writer_thread
            assert tracking_index.closed is False
            assert tier._closed is False

            writer_thread.alive = False
            tier.close()

            assert tier._writer_thread is None
            assert tracking_index.closed is True
            assert tier._closed is True
        finally:
            if not tracking_index.closed:
                tracking_index.index.close()

    def test_close_times_out_if_full_queue_cannot_accept_sentinel(
        self, tmp_path, monkeypatch
    ):
        """The close timeout must also bound sentinel insertion."""

        class FullShutdownQueue:
            def __init__(self):
                self.put_timeout = None

            def put(self, item, block=True, timeout=None):
                assert item is None
                self.put_timeout = timeout
                raise queue.Full

        class LiveThread:
            def join(self, timeout=None):
                raise AssertionError("join must not run without a sentinel")

            def is_alive(self):
                return True

        class TrackingIndex:
            def __init__(self, index):
                self.index = index
                self.closed = False

            def close(self):
                self.closed = True
                self.index.close()

        tier = SSDCacheTier(
            SSDCacheConfig(cache_dir=str(tmp_path / "sentinel_timeout"))
        )
        tracking_index = TrackingIndex(tier._index)
        shutdown_queue = FullShutdownQueue()
        writer_thread = LiveThread()
        tier._index = tracking_index
        tier._spill_queue = shutdown_queue
        tier._writer_thread = writer_thread
        monkeypatch.setattr(tier, "_WRITER_JOIN_TIMEOUT_S", 0.01)

        try:
            with pytest.raises(TimeoutError, match="shutdown sentinel"):
                tier.close()

            assert shutdown_queue.put_timeout == 0.01
            assert tier._writer_thread is writer_thread
            assert tracking_index.closed is False
            assert tier._closed is False
        finally:
            if not tracking_index.closed:
                tracking_index.index.close()

    @pytest.mark.anyio
    async def test_aclose_keeps_event_loop_responsive(self, tmp_path, monkeypatch):
        """Async shutdown must move the blocking writer join off the loop."""
        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path / "async_close")))
        original_close = tier.close
        release_close = threading.Event()
        close_finished = threading.Event()
        heartbeat_ran_during_close = False

        def blocking_close():
            assert release_close.wait(timeout=2.0)
            original_close()
            close_finished.set()

        async def heartbeat():
            nonlocal heartbeat_ran_during_close
            await asyncio.sleep(0)
            heartbeat_ran_during_close = not close_finished.is_set()

        monkeypatch.setattr(tier, "close", blocking_close)
        heartbeat_task = asyncio.create_task(heartbeat())
        release_timer = threading.Timer(0.1, release_close.set)
        release_timer.start()

        try:
            await asyncio.wait_for(tier.aclose(), timeout=1.0)
            await asyncio.wait_for(heartbeat_task, timeout=1.0)
        finally:
            release_close.set()
            release_timer.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()

        assert heartbeat_ran_during_close
