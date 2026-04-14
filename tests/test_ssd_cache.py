# SPDX-License-Identifier: Apache-2.0
"""Tests for SSD KV cache tiering."""

import pytest

from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheStats


class TestSSDCacheConfig:
    """Tests for SSDCacheConfig."""

    def test_default_config(self):
        config = SSDCacheConfig()
        assert config.cache_dir is None
        assert config.max_size_gb == 10.0
        assert config.max_entries == 10000
        assert config.file_permissions == 0o600
        assert config.dir_permissions == 0o700
        assert config.spill_queue_size == 64
        assert config.retention_seconds is None

    def test_custom_config(self):
        config = SSDCacheConfig(
            cache_dir="/tmp/test-ssd-cache",
            max_size_gb=5.0,
            max_entries=500,
        )
        assert config.cache_dir == "/tmp/test-ssd-cache"
        assert config.max_size_gb == 5.0
        assert config.max_entries == 500

    def test_max_size_bytes(self):
        config = SSDCacheConfig(max_size_gb=2.0)
        assert config.max_size_bytes == 2 * 1024 * 1024 * 1024

    def test_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size_gb"):
            SSDCacheConfig(max_size_gb=0.0)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            SSDCacheConfig(max_entries=0)

    def test_invalid_spill_queue_size(self):
        with pytest.raises(ValueError, match="spill_queue_size"):
            SSDCacheConfig(spill_queue_size=0)


class TestSSDCacheStats:
    """Tests for SSDCacheStats."""

    def test_initial_stats(self):
        stats = SSDCacheStats()
        assert stats.spill_count == 0
        assert stats.spill_bytes == 0
        assert stats.ssd_hits == 0
        assert stats.ssd_misses == 0
        assert stats.reload_latency_sum == 0.0
        assert stats.reload_bytes == 0
        assert stats.promotion_failures == 0

    def test_to_dict(self):
        stats = SSDCacheStats(
            spill_count=10,
            spill_bytes=1024 * 1024,
            ssd_hits=5,
            ssd_misses=3,
            reload_latency_sum=0.5,
            reload_bytes=512 * 1024,
            promotion_failures=1,
        )
        d = stats.to_dict()
        assert d["spill_count"] == 10
        assert d["spill_bytes"] == 1024 * 1024
        assert d["ssd_hits"] == 5
        assert d["ssd_misses"] == 3
        assert d["reload_bytes"] == 512 * 1024
        assert d["promotion_failures"] == 1
        assert d["ssd_hit_rate"] == pytest.approx(5 / 8)
        assert d["avg_reload_latency_ms"] == pytest.approx(100.0)

    def test_hit_rate_no_lookups(self):
        stats = SSDCacheStats()
        d = stats.to_dict()
        assert d["ssd_hit_rate"] == 0.0
        assert d["avg_reload_latency_ms"] == 0.0


import os
import tempfile

from vllm_mlx.ssd_cache import SSDIndex


class TestSSDIndex:
    """Tests for SQLite-backed SSD cache index."""

    @pytest.fixture
    def db_dir(self, tmp_path):
        return str(tmp_path / "ssd_index_test")

    @pytest.fixture
    def index(self, db_dir):
        os.makedirs(db_dir, exist_ok=True)
        idx = SSDIndex(db_dir)
        yield idx
        idx.close()

    def test_create_opens_db(self, index, db_dir):
        assert os.path.exists(os.path.join(db_dir, "index.db"))

    def test_insert_and_lookup_exact(self, index):
        tokens = (1, 2, 3, 4, 5)
        index.insert_entry(
            tokens_key=tokens,
            file_path="entry_abc123.safetensors",
            memory_bytes=4096,
            num_tokens=5,
        )
        result = index.lookup_exact(tokens)
        assert result is not None
        assert result["file_path"] == "entry_abc123.safetensors"
        assert result["memory_bytes"] == 4096
        assert result["num_tokens"] == 5

    def test_lookup_exact_miss(self, index):
        result = index.lookup_exact((99, 98, 97))
        assert result is None

    def test_lookup_prefix(self, index):
        # Insert a few entries
        index.insert_entry((1, 2, 3), "a.safetensors", 1000, 3)
        index.insert_entry((1, 2, 3, 4, 5), "b.safetensors", 2000, 5)
        index.insert_entry((1, 2, 3, 4, 5, 6, 7), "c.safetensors", 3000, 7)
        index.insert_entry((9, 8, 7), "d.safetensors", 1000, 3)

        # Lookup prefix matches for (1,2,3,4,5,6,7,8)
        results = index.lookup_prefix((1, 2, 3, 4, 5, 6, 7, 8))
        # Should return entries whose tokens are a prefix of the query
        file_paths = [r["file_path"] for r in results]
        assert "a.safetensors" in file_paths
        assert "b.safetensors" in file_paths
        assert "c.safetensors" in file_paths
        assert "d.safetensors" not in file_paths

    def test_delete_entry(self, index):
        tokens = (10, 20, 30)
        index.insert_entry(tokens, "x.safetensors", 500, 3)
        assert index.lookup_exact(tokens) is not None

        index.delete_entry(tokens)
        assert index.lookup_exact(tokens) is None

    def test_get_lru(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)  # Ensure different timestamps
        index.insert_entry((2,), "b.safetensors", 200, 1)
        time.sleep(0.01)
        index.insert_entry((3,), "c.safetensors", 300, 1)

        # Get oldest 2 entries
        lru = index.get_lru(limit=2)
        assert len(lru) == 2
        assert lru[0]["file_path"] == "a.safetensors"
        assert lru[1]["file_path"] == "b.safetensors"

    def test_get_total_bytes(self, index):
        index.insert_entry((1,), "a.safetensors", 1000, 1)
        index.insert_entry((2,), "b.safetensors", 2000, 1)
        assert index.get_total_bytes() == 3000

    def test_get_total_bytes_empty(self, index):
        assert index.get_total_bytes() == 0

    def test_get_entry_count(self, index):
        assert index.get_entry_count() == 0
        index.insert_entry((1,), "a.safetensors", 100, 1)
        index.insert_entry((2,), "b.safetensors", 200, 1)
        assert index.get_entry_count() == 2

    def test_insert_duplicate_replaces(self, index):
        tokens = (1, 2, 3)
        index.insert_entry(tokens, "old.safetensors", 100, 3)
        index.insert_entry(tokens, "new.safetensors", 200, 3)
        result = index.lookup_exact(tokens)
        assert result["file_path"] == "new.safetensors"
        assert result["memory_bytes"] == 200
        assert index.get_entry_count() == 1

    def test_touch_updates_access_time(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)
        index.insert_entry((2,), "b.safetensors", 100, 1)

        # Touch the older entry
        index.touch((1,))

        # Now (2,) should be the LRU entry
        lru = index.get_lru(limit=1)
        assert lru[0]["file_path"] == "b.safetensors"


import tempfile
import numpy as np

from vllm_mlx.ssd_cache import (
    LayerSerializer,
    KVCacheSerializer,
    ArraysCacheSerializer,
    get_serializer_for_layer,
    SERIALIZER_SUPPORT_MATRIX,
)


class MockMLXArray:
    """Minimal mock for MLX array with shape, dtype, and numpy conversion."""

    def __init__(self, data):
        self._data = np.array(data)
        self.shape = self._data.shape
        self.dtype = type("dtype", (), {"size": self._data.dtype.itemsize})()

    def __array__(self):
        return self._data


class MockKVCacheLayer:
    """Mock KVCache layer with keys, values, offset."""

    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class MockArraysCacheLayer:
    """Mock ArraysCache layer with state list."""

    def __init__(self, state):
        self.state = state


class TestLayerSerializer:
    """Tests for per-layer serializer interface."""

    def test_support_matrix_has_required_types(self):
        assert "KVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "RotatingKVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "ArraysCache" in SERIALIZER_SUPPORT_MATRIX

    def test_kv_cache_serializer_round_trip(self, tmp_path):
        keys = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=32)

        serializer = KVCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        metadata = serializer.serialize_layer(layer, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "KVCache"
        assert metadata["offset"] == 32

        restored = serializer.deserialize_layer(file_path, metadata)
        assert restored["offset"] == 32
        np.testing.assert_array_almost_equal(
            np.array(restored["keys"]),
            np.array(keys),
        )
        np.testing.assert_array_almost_equal(
            np.array(restored["values"]),
            np.array(values),
        )

    def test_arrays_cache_serializer_round_trip(self, tmp_path):
        arr0 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        arr1 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        layer = MockArraysCacheLayer(state=[arr0, arr1])

        serializer = ArraysCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        metadata = serializer.serialize_layer(layer, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "ArraysCache"
        assert metadata["num_arrays"] == 2

        restored = serializer.deserialize_layer(file_path, metadata)
        assert len(restored["state"]) == 2
        np.testing.assert_array_almost_equal(
            np.array(restored["state"][0]),
            np.array(arr0),
        )

    def test_get_serializer_for_kvcache(self):
        layer = MockKVCacheLayer(keys=None, values=None, offset=0)
        s = get_serializer_for_layer(layer)
        assert isinstance(s, KVCacheSerializer)

    def test_get_serializer_for_arrays_cache(self):
        layer = MockArraysCacheLayer(state=[])
        s = get_serializer_for_layer(layer)
        assert isinstance(s, ArraysCacheSerializer)

    def test_get_serializer_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_serializer_for_layer("not a cache layer")


from vllm_mlx.ssd_cache import SSDCacheTier


class TestSSDCacheTierCore:
    """Tests for SSDCacheTier initialization and directory setup."""

    def test_init_creates_directory(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_test")
        config = SSDCacheConfig(cache_dir=cache_dir)
        tier = SSDCacheTier(config)
        try:
            assert os.path.isdir(cache_dir)
            assert os.path.exists(os.path.join(cache_dir, "data"))
            assert os.path.exists(os.path.join(cache_dir, "index.db"))
        finally:
            tier.close()

    def test_dir_permissions(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_perms")
        config = SSDCacheConfig(cache_dir=cache_dir, dir_permissions=0o700)
        tier = SSDCacheTier(config)
        try:
            stat = os.stat(os.path.join(cache_dir, "data"))
            # Check owner permissions (at least rwx for owner)
            assert stat.st_mode & 0o700 == 0o700
        finally:
            tier.close()

    def test_entry_hash_deterministic(self):
        tokens = (1, 2, 3, 4, 5)
        h1 = SSDCacheTier._entry_hash(tokens)
        h2 = SSDCacheTier._entry_hash(tokens)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_entry_hash_different_for_different_tokens(self):
        h1 = SSDCacheTier._entry_hash((1, 2, 3))
        h2 = SSDCacheTier._entry_hash((1, 2, 4))
        assert h1 != h2

    def test_stats_initial(self, tmp_path):
        config = SSDCacheConfig(cache_dir=str(tmp_path / "stats_test"))
        tier = SSDCacheTier(config)
        try:
            stats = tier.get_stats()
            assert stats["spill_count"] == 0
            assert stats["ssd_hits"] == 0
        finally:
            tier.close()

    def test_close_idempotent(self, tmp_path):
        config = SSDCacheConfig(cache_dir=str(tmp_path / "close_test"))
        tier = SSDCacheTier(config)
        tier.close()
        tier.close()  # Should not raise
