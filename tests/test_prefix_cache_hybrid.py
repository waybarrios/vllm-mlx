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
