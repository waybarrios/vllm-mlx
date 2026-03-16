# SPDX-License-Identifier: Apache-2.0
"""Tests for BlockAwarePrefixCache with hybrid model caches (issues #142, #136)."""

from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from vllm_mlx.prefix_cache import (
    BlockAwarePrefixCache,
    NonKVCacheData,
    _is_kv_layer,
)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_state(seq_len=64, n_kv_heads=4, head_dim=32):
    """Simulate a KVCache layer_state dict (4D tensors)."""
    keys = mx.zeros((1, n_kv_heads, seq_len, head_dim))
    values = mx.zeros((1, n_kv_heads, seq_len, head_dim))
    return {
        "state": (keys, values),
        "meta_state": (str(seq_len),),
        "class_name": "KVCache",
        "class_ref": KVCache,
    }


def _make_arrays_state(seq_len=64, conv_dim=128, ssm_heads=8, ssm_dim=64):
    """Simulate an ArraysCache layer_state dict (conv_3D + recurrent_4D)."""
    conv_state = mx.zeros((1, 3, conv_dim))
    ssm_state = mx.zeros((1, ssm_heads, ssm_dim, ssm_dim))
    return {
        "state": [conv_state, ssm_state],
        "meta_state": "",
        "class_name": "ArraysCache",
        "class_ref": ArraysCache,
    }


def _make_hybrid_cache_data(n_total=48, attn_interval=4, seq_len=128):
    """Simulate extracted cache states for a hybrid model.

    Qwen 3.5 pattern: every 4th layer is attention, rest are GatedDeltaNet.
    """
    cache_data = []
    for i in range(n_total):
        is_attn = (i + 1) % attn_interval == 0
        if is_attn:
            cache_data.append(_make_kv_state(seq_len=seq_len))
        else:
            cache_data.append(_make_arrays_state())
    return cache_data


def _make_pure_kv_cache_data(n_layers=32, seq_len=128):
    """Simulate extracted cache states for a pure attention model."""
    return [_make_kv_state(seq_len=seq_len) for _ in range(n_layers)]


# ---------------------------------------------------------------------------
# Tests: Layer Classification
# ---------------------------------------------------------------------------

class TestIsKVLayer:
    def test_kvcache_is_kv(self):
        assert _is_kv_layer({"class_name": "KVCache"}) is True

    def test_rotating_kvcache_is_kv(self):
        assert _is_kv_layer({"class_name": "RotatingKVCache"}) is True

    def test_quantized_kvcache_is_kv(self):
        assert _is_kv_layer({"class_name": "QuantizedKVCache"}) is True

    def test_batch_kvcache_is_kv(self):
        assert _is_kv_layer({"class_name": "BatchKVCache"}) is True

    def test_arrays_cache_is_not_kv(self):
        assert _is_kv_layer({"class_name": "ArraysCache"}) is False

    def test_cache_list_is_not_kv(self):
        assert _is_kv_layer({"class_name": "CacheList"}) is False

    def test_missing_class_name_is_not_kv(self):
        assert _is_kv_layer({}) is False

    def test_empty_class_name_is_not_kv(self):
        assert _is_kv_layer({"class_name": ""}) is False


class TestExtractBlockTensorSlice:
    """Test _extract_block_tensor_slice with hybrid cache data."""

    @pytest.fixture
    def cache(self):
        mock_model = MagicMock()
        from vllm_mlx.paged_cache import PagedCacheManager
        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(mock_model, paged)

    def test_pure_kv_slicing_unchanged(self, cache):
        """Pure KV model: all layers sliced as before."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=128)
        result = cache._extract_block_tensor_slice(data, 0, 64)
        assert result is not None
        assert len(result) == 4
        for entry in result:
            assert entry is not None
            keys_slice, values_slice = entry
            assert keys_slice.shape == (1, 4, 64, 32)
            assert values_slice.shape == (1, 4, 64, 32)

    def test_hybrid_skips_non_kv_layers(self, cache):
        """Hybrid model: KV layers sliced, ArraysCache layers return None."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        # Layers: 0=Arr, 1=Arr, 2=Arr, 3=KV, 4=Arr, 5=Arr, 6=Arr, 7=KV
        result = cache._extract_block_tensor_slice(data, 0, 64)
        assert result is not None
        assert len(result) == 8
        # Non-KV layers are None
        assert result[0] is None
        assert result[1] is None
        assert result[2] is None
        assert result[4] is None
        assert result[5] is None
        assert result[6] is None
        # KV layers are sliced
        assert result[3] is not None
        keys, values = result[3]
        assert keys.shape == (1, 4, 64, 32)
        assert result[7] is not None

    def test_hybrid_does_not_crash(self, cache):
        """Regression: the original bug — no IndexError on ArraysCache layers."""
        data = _make_hybrid_cache_data(n_total=48, attn_interval=4, seq_len=256)
        result = cache._extract_block_tensor_slice(data, 0, 64)
        assert result is not None

    def test_slice_beyond_seq_len(self, cache):
        """KV slice beyond available seq_len clips correctly."""
        data = _make_pure_kv_cache_data(n_layers=2, seq_len=100)
        result = cache._extract_block_tensor_slice(data, 64, 128)
        assert result is not None
        keys, values = result[0]
        assert keys.shape[2] == 36  # 100 - 64


# ---------------------------------------------------------------------------
# Tests: store_cache with hybrid model cache data
# ---------------------------------------------------------------------------

class TestStoreHybridCache:
    """Test store_cache with hybrid model cache data."""

    @pytest.fixture
    def cache(self):
        mock_model = MagicMock()
        from vllm_mlx.paged_cache import PagedCacheManager
        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(mock_model, paged)

    def test_stores_non_kv_states(self, cache):
        """Hybrid cache data stores non-KV states in _non_kv_states."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        assert len(cache._non_kv_states) == 1
        non_kv = list(cache._non_kv_states.values())[0]
        assert isinstance(non_kv, NonKVCacheData)
        assert non_kv.total_layers == 8
        assert len(non_kv.layer_indices) == 6  # 6 ArraysCache layers
        assert non_kv.layer_indices == [0, 1, 2, 4, 5, 6]

    def test_pure_kv_no_non_kv_states(self, cache):
        """Pure KV model does not create non-KV states."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        assert len(cache._non_kv_states) == 0

    def test_has_non_kv_flag_on_entry(self, cache):
        """BlockCacheEntry gets has_non_kv flag."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        entry = cache._request_tables["req-1"]
        assert entry.has_non_kv is True

    def test_has_non_kv_false_for_pure_kv(self, cache):
        """Pure KV entries have has_non_kv=False."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        entry = cache._request_tables["req-1"]
        assert entry.has_non_kv is False


# ---------------------------------------------------------------------------
# Tests: reconstruct_cache with hybrid model cache data
# ---------------------------------------------------------------------------

class TestReconstructHybridCache:
    """Test reconstruct_cache with hybrid model cache data."""

    @pytest.fixture
    def cache(self):
        mock_model = MagicMock()
        from vllm_mlx.paged_cache import PagedCacheManager
        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(mock_model, paged)

    def test_pure_kv_reconstruct_unchanged(self, cache):
        """Pure KV model: reconstruct works as before."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=128)
        tokens = list(range(128))
        bt = cache.store_cache("req-1", tokens, data)
        result = cache.reconstruct_cache(bt)
        assert result is not None
        assert len(result) == 4
        for c in result:
            assert hasattr(c, "keys")
            assert hasattr(c, "values")

    def test_hybrid_reconstruct_all_layers(self, cache):
        """Hybrid model: reconstructs both KV and non-KV layers."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        bt = cache.store_cache("req-1", tokens, data)
        result = cache.reconstruct_cache(bt)
        assert result is not None
        assert len(result) == 8  # All layers present
        # KV layers (3, 7) should be KVCache
        assert hasattr(result[3], "keys")
        assert hasattr(result[7], "keys")
        # Non-KV layers (0,1,2,4,5,6) should be ArraysCache
        assert isinstance(result[0], ArraysCache)
        assert isinstance(result[4], ArraysCache)

    def test_hybrid_reconstruct_missing_non_kv_returns_none(self, cache):
        """If non-KV states are missing, return None (can't reconstruct)."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        bt = cache.store_cache("req-1", tokens, data)
        # Delete non-KV states to simulate missing data
        cache._non_kv_states.clear()
        result = cache.reconstruct_cache(bt)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: fetch_cache with hybrid model prefix matching
# ---------------------------------------------------------------------------

class TestFetchHybridCache:
    """Test fetch_cache with hybrid model prefix matching."""

    @pytest.fixture
    def cache(self):
        mock_model = MagicMock()
        from vllm_mlx.paged_cache import PagedCacheManager
        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(mock_model, paged)

    def test_full_match_hybrid_no_crash(self, cache):
        """Full prefix match with non-KV states does not crash."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        # Same tokens — should not crash
        bt, remaining = cache.fetch_cache("req-2", tokens)

    def test_pure_kv_partial_match_no_crash(self, cache):
        """Pure KV model: partial prefix does not crash."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=192)
        tokens = list(range(192))
        cache.store_cache("req-1", tokens, data)
        shorter_tokens = list(range(128))
        bt, remaining = cache.fetch_cache("req-2", shorter_tokens)

    def test_cleanup_removes_non_kv_states(self, cache):
        """release_cache cleans up non-KV states when no other request uses same blocks."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        assert len(cache._non_kv_states) == 1
        cache.release_cache("req-1")
        assert len(cache._non_kv_states) == 0

    def test_clear_removes_all_non_kv_states(self, cache):
        """clear() removes all non-KV states."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        cache.store_cache("req-1", list(range(128)), data)
        cache.store_cache("req-2", list(range(64, 192)), data)
        cache.clear()
        assert len(cache._non_kv_states) == 0


# ---------------------------------------------------------------------------
# Tests: scheduler robustness in cache reconstruction
# ---------------------------------------------------------------------------

class TestSchedulerRobustness:
    """Test scheduler cache reconstruction with non-KV layers."""

    def test_from_state_works_for_both_cache_types(self):
        """Both ArraysCache and KVCache can be reconstructed via from_state.

        KVCache.meta_state is always "" (inherits _BaseCache which has no
        meta_state). Only subclasses like RotatingKVCache define meta_state
        as a tuple. The scheduler guard should handle non-4D state tensors.
        """
        arrays_state = {
            "state": [mx.zeros((1, 3, 128)), mx.zeros((1, 8, 64, 64))],
            "meta_state": "",
            "class_name": "ArraysCache",
            "class_ref": ArraysCache,
        }
        # KVCache.meta_state is "" (inherits _BaseCache), NOT a tuple
        kv_state = {
            "state": (mx.zeros((1, 4, 100, 32)), mx.zeros((1, 4, 100, 32))),
            "meta_state": "",
            "class_name": "KVCache",
            "class_ref": KVCache,
        }
        extracted = [arrays_state, kv_state, arrays_state, kv_state]

        for layer_state in extracted:
            cls = layer_state["class_ref"]
            state = layer_state["state"]
            meta = layer_state.get("meta_state", "")
            obj = cls.from_state(state, meta)
            assert obj is not None
