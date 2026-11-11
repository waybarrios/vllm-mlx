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


class TestSSDReaderSerializerContract:
    @pytest.mark.parametrize(
        "layer_type, serializer_name",
        [
            ("KVCache", "KVCacheSerializer"),
            ("RotatingKVCache", "KVCacheSerializer"),
            ("ArraysCache", "ArraysCacheSerializer"),
            ("MambaCache", "ArraysCacheSerializer"),
            ("CacheList", "CacheListSerializer"),
        ],
    )
    def test_known_alias_dispatch_preserves_payload(
        self, tmp_path, monkeypatch, layer_type, serializer_name
    ):
        import json

        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "entry"
        entry.mkdir()
        metadata = {"layer_type": layer_type, "layer_idx": 7, "opaque": [3, None]}
        (entry / "manifest.json").write_text(json.dumps({"layers": [metadata]}))
        calls, quarantines = [], []
        state = {"nested": [object()], "opaque": object()}

        def deserialize(serializer, path, meta):
            calls.append((type(serializer), path, meta))
            return state

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        result = tier._read_entry((4, 5), "entry")

        assert len(result) == 1
        assert result[0] is state
        assert result[0]["nested"] is state["nested"]
        assert calls == [
            (
                getattr(ssd_cache, serializer_name),
                str(entry / "layer_7.safetensors"),
                metadata,
            )
        ]
        assert quarantines == []

    def test_mixed_layers_keep_order_and_returned_dictionary_identity(
        self, tmp_path, monkeypatch
    ):
        import json

        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "mixed"
        entry.mkdir()
        layers = [
            {"layer_type": "MambaCache", "layer_idx": 8},
            {"layer_type": "CacheList", "layer_idx": 2},
            {"layer_type": "RotatingKVCache", "layer_idx": 5},
            {"layer_type": "ArraysCache", "layer_idx": 1},
            {"layer_type": "KVCache", "layer_idx": 9},
        ]
        (entry / "manifest.json").write_text(json.dumps({"layers": layers}))
        states = [{"nested": [object()]} for _ in layers]
        calls, quarantines = [], []

        def deserialize(serializer, path, meta):
            state = states[len(calls)]
            calls.append((type(serializer), path, meta))
            return state

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        result = tier._read_entry((1, 2), "mixed")

        assert len(result) == 5
        assert all(actual is expected for actual, expected in zip(result, states))
        assert all(result[i]["nested"] is states[i]["nested"] for i in range(5))
        assert [cls for cls, _, _ in calls] == [
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.KVCacheSerializer,
        ]
        assert [path for _, path, _ in calls] == [
            str(entry / "layer_8.safetensors"),
            str(entry / "layer_2.safetensors"),
            str(entry / "layer_5.safetensors"),
            str(entry / "layer_1.safetensors"),
            str(entry / "layer_9.safetensors"),
        ]
        assert [meta for _, _, meta in calls] == layers
        assert quarantines == []

    @pytest.mark.parametrize("successful_prefix", [False, True])
    def test_unknown_layer_quarantines_once_and_stops_without_partial_result(
        self, tmp_path, monkeypatch, caplog, successful_prefix
    ):
        import json

        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "unknown"
        entry.mkdir()
        layers = [{"layer_type": "UnknownCache", "layer_idx": 1}]
        if successful_prefix:
            layers.insert(0, {"layer_type": "KVCache", "layer_idx": 0})
        layers.append({"layer_type": "CacheList", "layer_idx": 2})
        (entry / "manifest.json").write_text(json.dumps({"layers": layers}))
        calls, quarantines = [], []

        def deserialize(serializer, path, meta):
            calls.append(meta["layer_idx"])
            return {"opaque": object()}

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        assert tier._read_entry((11,), "unknown") is None
        assert calls == ([0] if successful_prefix else [])
        assert quarantines == [((11,), "unknown")]
        assert len(caplog.records) == 1
        assert "unknown layer type UnknownCache" in caplog.text

    @pytest.mark.parametrize("failing_type", ["KVCache", "ArraysCache", "CacheList"])
    def test_deserializer_error_quarantines_and_discards_earlier_success(
        self, tmp_path, monkeypatch, caplog, failing_type
    ):
        import json

        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "failed"
        entry.mkdir()
        layers = [
            {"layer_type": "KVCache", "layer_idx": 0},
            {"layer_type": failing_type, "layer_idx": 4},
            {"layer_type": "MambaCache", "layer_idx": 9},
        ]
        (entry / "manifest.json").write_text(json.dumps({"layers": layers}))
        calls, quarantines = [], []

        def deserialize(serializer, path, meta):
            calls.append(meta["layer_idx"])
            if meta["layer_idx"] == 4:
                raise ValueError("broken layer payload")
            return {"opaque": object()}

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        assert tier._read_entry((12,), "failed") is None
        assert calls == [0, 4]
        assert quarantines == [((12,), "failed")]
        assert len(caplog.records) == 1
        assert "corrupt layer 4 in failed" in caplog.text
        assert "broken layer payload" in caplog.text

    @pytest.mark.parametrize("manifest_kind", ["invalid_json", "missing_file"])
    def test_manifest_failure_quarantines_before_any_deserialization(
        self, tmp_path, monkeypatch, caplog, manifest_kind
    ):
        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "corrupt"
        entry.mkdir()
        if manifest_kind == "invalid_json":
            (entry / "manifest.json").write_text("{")
        calls, quarantines = [], []

        def deserialize(serializer, path, meta):
            calls.append(meta)
            return {}

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        assert tier._read_entry((13,), "corrupt") is None
        assert calls == []
        assert quarantines == [((13,), "corrupt")]
        assert len(caplog.records) == 1
        assert "corrupt manifest for corrupt" in caplog.text

    def test_empty_manifest_returns_empty_layers_without_callbacks(
        self, tmp_path, monkeypatch
    ):
        from vllm_mlx import ssd_cache

        tier = object.__new__(SSDCacheTier)
        tier._data_dir = str(tmp_path)
        entry = tmp_path / "empty"
        entry.mkdir()
        (entry / "manifest.json").write_text('{"layers": []}')
        calls, quarantines = [], []

        def deserialize(serializer, path, meta):
            calls.append(meta)
            return {}

        for cls in (
            ssd_cache.KVCacheSerializer,
            ssd_cache.ArraysCacheSerializer,
            ssd_cache.CacheListSerializer,
        ):
            monkeypatch.setattr(cls, "deserialize_layer", deserialize)
        monkeypatch.setattr(
            tier, "_quarantine_entry", lambda *args: quarantines.append(args)
        )

        assert tier._read_entry((), "empty") == []
        assert calls == []
        assert quarantines == []
