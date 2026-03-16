# SPDX-License-Identifier: Apache-2.0
"""
Tests for prefix cache with hybrid models (mixed KV + non-KV cache layers).

Hybrid models like Qwen3.5 use a mix of standard attention layers (KVCache)
and linear attention / SSM layers (ArraysCache). The prefix cache must handle
both cache types correctly:
- KVCache layers: sequence-indexed, can be sliced into blocks
- ArraysCache layers: cumulative state, stored separately per prefix

These tests cover:
- 3D vs 4D KV tensor handling
- Mixed KV/ArraysCache layer detection and storage
- Exact prefix match reconstruction for hybrid models
- Graceful fallback when non-KV states are unavailable
"""

import platform
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


def _make_array(shape, dtype="float16"):
    """Create a mock MLX-like array with shape and ndim."""
    import mlx.core as mx

    return mx.zeros(shape, dtype=getattr(mx, dtype))


def _make_kv_state_4d(seq_len, n_heads=4, head_dim=64):
    """Create a 4D KV cache state: (batch, n_heads, seq_len, head_dim)."""
    keys = _make_array((1, n_heads, seq_len, head_dim))
    values = _make_array((1, n_heads, seq_len, head_dim))
    return (keys, values)


def _make_kv_state_3d(seq_len, n_heads=4, head_dim=64):
    """Create a 3D KV cache state: (n_heads, seq_len, head_dim) — no batch dim."""
    keys = _make_array((n_heads, seq_len, head_dim))
    values = _make_array((n_heads, seq_len, head_dim))
    return (keys, values)


def _make_arrays_cache_state():
    """Create an ArraysCache state: list of [conv_state, ssm_state]."""
    conv_state = _make_array((1, 3, 128))  # (batch, kernel_size-1, conv_dim)
    ssm_state = _make_array((1, 16, 64))  # (batch, state_size, dim)
    return [conv_state, ssm_state]


def _kv_layer_dict(state, class_name="KVCache"):
    """Build a layer state dict for a KV cache layer."""
    from mlx_lm.models.cache import KVCache

    return {
        "state": state,
        "meta_state": (str(state[0].shape[-2]),),
        "class_name": class_name,
        "class_ref": KVCache,
    }


def _arrays_layer_dict(state, class_name="ArraysCache"):
    """Build a layer state dict for an ArraysCache layer."""
    from mlx_lm.models.cache import ArraysCache

    return {
        "state": state,
        "meta_state": "",
        "class_name": class_name,
        "class_ref": ArraysCache,
    }


# ── _is_kv_cache_layer ────────────────────────────────────────────────────


class TestIsKvCacheLayer:
    """Tests for the _is_kv_cache_layer helper."""

    def test_kv_cache_by_class_name(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        layer = {"class_name": "KVCache", "state": (None, None)}
        assert _is_kv_cache_layer(layer) is True

    def test_batch_kv_cache_by_class_name(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        layer = {"class_name": "BatchKVCache", "state": (None, None)}
        assert _is_kv_cache_layer(layer) is True

    def test_arrays_cache_by_class_name(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        layer = {"class_name": "ArraysCache", "state": [None, None]}
        assert _is_kv_cache_layer(layer) is False

    def test_mamba_cache_by_class_name(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        layer = {"class_name": "MambaCache", "state": [None]}
        assert _is_kv_cache_layer(layer) is False

    def test_structural_fallback_kv(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        keys = _make_array((4, 128, 64))
        values = _make_array((4, 128, 64))
        layer = {"class_name": "UnknownCache", "state": (keys, values)}
        assert _is_kv_cache_layer(layer) is True

    def test_structural_fallback_non_kv(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        # List state (not tuple) → not recognized as KV
        layer = {"class_name": "UnknownCache", "state": [_make_array((4, 3, 64))]}
        assert _is_kv_cache_layer(layer) is False

    def test_missing_class_name(self):
        from vllm_mlx.prefix_cache import _is_kv_cache_layer

        # No class_name but structural tuple-of-2 → KV
        keys = _make_array((4, 128, 64))
        values = _make_array((4, 128, 64))
        layer = {"state": (keys, values)}
        assert _is_kv_cache_layer(layer) is True


# ── _extract_block_tensor_slice ────────────────────────────────────────────


class TestExtractBlockTensorSlice:
    """Tests for _extract_block_tensor_slice with mixed layer types."""

    def _make_cache(self):
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    def test_4d_kv_only(self):
        """Standard 4D KV tensors should slice correctly."""
        cache = self._make_cache()
        state = _make_kv_state_4d(seq_len=128)
        cache_data = [_kv_layer_dict(state)]

        result = cache._extract_block_tensor_slice(cache_data, 0, 64)
        assert result is not None
        assert len(result) == 1
        assert result[0] is not None
        k, v = result[0]
        assert k.shape == (1, 4, 64, 64)
        assert v.shape == (1, 4, 64, 64)

    def test_3d_kv_only(self):
        """3D KV tensors (no batch dim) should slice correctly."""
        cache = self._make_cache()
        state = _make_kv_state_3d(seq_len=128)
        cache_data = [_kv_layer_dict(state)]

        result = cache._extract_block_tensor_slice(cache_data, 0, 64)
        assert result is not None
        assert len(result) == 1
        assert result[0] is not None
        k, v = result[0]
        # 3D: (n_heads, seq_len, head_dim) → sliced to (n_heads, 64, head_dim)
        assert k.shape == (4, 64, 64)
        assert v.shape == (4, 64, 64)

    def test_arrays_cache_skipped(self):
        """ArraysCache layers should produce None placeholder."""
        cache = self._make_cache()
        arrays_state = _make_arrays_cache_state()
        cache_data = [_arrays_layer_dict(arrays_state)]

        result = cache._extract_block_tensor_slice(cache_data, 0, 64)
        # All layers are non-KV → returns None (no KV data)
        assert result is None

    def test_mixed_kv_and_arrays(self):
        """Mixed KV + ArraysCache layers: KV sliced, ArraysCache → None."""
        cache = self._make_cache()
        kv_state = _make_kv_state_3d(seq_len=128)
        arrays_state = _make_arrays_cache_state()

        # Qwen3.5-like pattern: 3 linear + 1 attention
        cache_data = [
            _arrays_layer_dict(arrays_state),
            _arrays_layer_dict(arrays_state),
            _arrays_layer_dict(arrays_state),
            _kv_layer_dict(kv_state),
        ]

        result = cache._extract_block_tensor_slice(cache_data, 0, 64)
        assert result is not None
        assert len(result) == 4
        # First 3 layers (ArraysCache) → None
        assert result[0] is None
        assert result[1] is None
        assert result[2] is None
        # Last layer (KVCache) → sliced
        assert result[3] is not None
        k, v = result[3]
        assert k.shape == (4, 64, 64)

    def test_slice_second_block(self):
        """Slicing tokens 64-128 from a 128-token KV cache."""
        cache = self._make_cache()
        state = _make_kv_state_3d(seq_len=128)
        cache_data = [_kv_layer_dict(state)]

        result = cache._extract_block_tensor_slice(cache_data, 64, 128)
        assert result is not None
        k, v = result[0]
        assert k.shape == (4, 64, 64)


# ── Full store + reconstruct roundtrip ─────────────────────────────────────


class TestHybridCacheRoundtrip:
    """End-to-end tests for store → fetch → reconstruct with hybrid caches."""

    def _make_cache(self, block_size=64, max_blocks=100):
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=block_size, max_blocks=max_blocks)
        return BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

    def _make_hybrid_cache_data(self, seq_len=128, pattern=(3, 1)):
        """
        Create Qwen3.5-like cache data: groups of (N_linear, N_attention) layers.

        Args:
            seq_len: Sequence length for KV tensors
            pattern: (num_linear, num_attention) per group, repeated to fill

        Returns:
            List of layer state dicts
        """
        n_linear, n_attn = pattern
        layers = []
        for _ in range(n_linear):
            layers.append(_arrays_layer_dict(_make_arrays_cache_state()))
        for _ in range(n_attn):
            layers.append(_kv_layer_dict(_make_kv_state_3d(seq_len=seq_len)))
        return layers

    def test_store_and_reconstruct_pure_kv_3d(self):
        """Pure KV model with 3D tensors: store + reconstruct roundtrip."""
        cache = self._make_cache()
        seq_len = 128
        tokens = list(range(seq_len))
        cache_data = [_kv_layer_dict(_make_kv_state_3d(seq_len=seq_len))]

        # Store
        block_table = cache.store_cache("req-1", tokens, cache_data)
        assert block_table is not None
        assert block_table.num_tokens == seq_len

        # Reconstruct
        reconstructed = cache.reconstruct_cache(block_table)
        assert reconstructed is not None
        assert len(reconstructed) == 1
        # Verify the reconstructed cache is a KVCache with correct shape
        assert hasattr(reconstructed[0], "keys")
        assert reconstructed[0].keys.shape == (4, seq_len, 64)
        assert reconstructed[0].offset == seq_len

    def test_store_and_reconstruct_hybrid_exact_match(self):
        """Hybrid model: exact block match should reconstruct all layers."""
        cache = self._make_cache()
        seq_len = 128
        tokens = list(range(seq_len))
        cache_data = self._make_hybrid_cache_data(seq_len=seq_len)

        # Store
        block_table = cache.store_cache("req-1", tokens, cache_data)
        assert block_table is not None

        # Verify non-KV states were stored
        block_key = tuple(block_table.block_ids)
        assert block_key in cache._non_kv_layer_states
        assert len(cache._non_kv_layer_states[block_key]) == 3  # 3 ArraysCache layers

        # Reconstruct with same block table
        reconstructed = cache.reconstruct_cache(block_table)
        assert reconstructed is not None
        assert len(reconstructed) == 4  # 3 ArraysCache + 1 KVCache

        # First 3 should be ArraysCache-like (from from_state)
        for i in range(3):
            assert hasattr(reconstructed[i], "state")
        # Last should be KVCache
        assert hasattr(reconstructed[3], "keys")
        assert reconstructed[3].offset == seq_len

    def test_fetch_exact_prefix_hybrid(self):
        """Hybrid model: fetch with exact same tokens should get cache hit."""
        cache = self._make_cache()
        seq_len = 128
        tokens = list(range(seq_len))
        cache_data = self._make_hybrid_cache_data(seq_len=seq_len)

        # Store original
        cache.store_cache("req-1", tokens, cache_data)

        # Fetch for new request with same tokens
        block_table, remaining = cache.fetch_cache("req-2", tokens)
        assert block_table is not None
        assert remaining == [] or len(remaining) == 0

        # Reconstruct should work (exact block match)
        reconstructed = cache.reconstruct_cache(block_table)
        assert reconstructed is not None
        assert len(reconstructed) == 4

    def test_fetch_prefix_match_hybrid_partial_returns_none(self):
        """Hybrid model: partial prefix can't reconstruct (SSM state mismatch)."""
        cache = self._make_cache()
        seq_len = 256  # 4 blocks
        tokens = list(range(seq_len))
        cache_data = self._make_hybrid_cache_data(seq_len=seq_len)

        # Store 256 tokens (4 blocks)
        cache.store_cache("req-1", tokens, cache_data)

        # Fetch with first 128 tokens + extra → shares 2 of 4 blocks
        short_tokens = list(range(128)) + [999, 1000]
        block_table, remaining = cache.fetch_cache("req-2", short_tokens)

        if block_table is not None:
            # Reconstruct should return None for hybrid model partial match:
            # non-KV states stored under (b1,b2,b3,b4) but request has (b1,b2)
            reconstructed = cache.reconstruct_cache(block_table)
            assert reconstructed is None  # Correct: can't reuse SSM state

    def test_pure_kv_partial_prefix(self):
        """Pure KV model: partial prefix should still reconstruct."""
        cache = self._make_cache()
        # Store a 256-token cache
        long_tokens = list(range(256))
        state = _make_kv_state_3d(seq_len=256)
        cache_data = [_kv_layer_dict(state)]
        cache.store_cache("req-1", long_tokens, cache_data)

        # Fetch with first 128 tokens + extra
        short_tokens = list(range(128)) + [999]
        block_table, remaining = cache.fetch_cache("req-2", short_tokens)

        if block_table is not None:
            reconstructed = cache.reconstruct_cache(block_table)
            # Pure KV: should always work (no non-KV state needed)
            assert reconstructed is not None

    def test_clear_resets_non_kv_states(self):
        """clear() should reset all state including non-KV cache."""
        cache = self._make_cache()
        tokens = list(range(128))
        cache_data = self._make_hybrid_cache_data(seq_len=128)

        cache.store_cache("req-1", tokens, cache_data)
        assert len(cache._non_kv_layer_states) > 0

        cache.clear()
        assert len(cache._non_kv_layer_states) == 0

    def test_no_crash_on_empty_arrays_cache(self):
        """ArraysCache with None entries should not crash extraction."""
        cache = self._make_cache()
        # ArraysCache state where entries haven't been initialized
        null_state = [None, None]
        cache_data = [_arrays_layer_dict(null_state)]

        # Should not crash
        result = cache._extract_block_tensor_slice(cache_data, 0, 64)
        # All layers non-KV → None
        assert result is None


# ── Stats ──────────────────────────────────────────────────────────────────


class TestHybridCacheStats:
    """Verify stats work with hybrid cache operations."""

    def test_stats_after_hybrid_store(self):
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        paged = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged)

        tokens = list(range(128))
        cache_data = [
            _arrays_layer_dict(_make_arrays_cache_state()),
            _kv_layer_dict(_make_kv_state_3d(seq_len=128)),
        ]
        cache.store_cache("req-1", tokens, cache_data)

        stats = cache.get_stats()
        assert stats["misses"] == 0
