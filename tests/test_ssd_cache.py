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
