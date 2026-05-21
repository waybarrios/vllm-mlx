# SPDX-License-Identifier: Apache-2.0
"""Tests for linear-attention (GatedDeltaNet) state quantization in prefix cache.

Qwen3.5/3.6 hybrid models keep GatedDeltaNet recurrent + conv state in an
``ArraysCache``. These tests cover the ``_QuantizedArraysCacheWrapper`` that
quantizes that state on store and reconstructs it on fetch.
"""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from vllm_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _QuantizedArraysCacheWrapper,
    _QuantizedCacheWrapper,
    _dequantize_cache,
    _quantize_cache,
    estimate_kv_cache_memory,
)


def _make_arrays_cache(
    conv_dim: int = 128,
    hv: int = 4,
    dv: int = 16,
    dk: int = 64,
    recurrent_dtype=mx.float32,
):
    """Create an ArraysCache mimicking GatedDeltaNet state: [conv, recurrent].

    cache[0] = conv state (bfloat16, like inputs.dtype)
    cache[1] = recurrent matrix state (float32, as gated_delta.py allocates)
    """
    c = ArraysCache(size=2)
    c.cache[0] = mx.random.normal((1, 3, conv_dim)).astype(mx.bfloat16)
    c.cache[1] = mx.random.normal((1, hv, dv, dk)).astype(recurrent_dtype)
    mx.eval(c.cache[0], c.cache[1])
    return c


def _make_kv_layer(seq_len: int = 64, n_heads: int = 8, head_dim: int = 64):
    kv = KVCache()
    kv.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
    kv.values = mx.random.normal((1, n_heads, seq_len, head_dim))
    kv.offset = seq_len
    mx.eval(kv.keys, kv.values)
    return kv


def _mean_abs_err(a, b) -> float:
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).mean().item()


# ---------------------------------------------------------------------------
# _QuantizedArraysCacheWrapper — quantize / dequantize
# ---------------------------------------------------------------------------


class TestArraysQuantizeDequantize:
    def test_quantize_wraps_arrays_cache(self):
        cache = [_make_arrays_cache()]
        quantized = _quantize_cache(cache, quantize_kv=False, quantize_linear=True)
        assert len(quantized) == 1
        assert isinstance(quantized[0], _QuantizedArraysCacheWrapper)

    def test_quantize_skipped_when_disabled(self):
        cache = [_make_arrays_cache()]
        quantized = _quantize_cache(cache, quantize_kv=False, quantize_linear=False)
        assert quantized[0] is cache[0]  # passed through unchanged

    def test_dequantize_restores_arrays_cache(self):
        original = _make_arrays_cache()
        quantized = _quantize_cache([original], quantize_kv=False, quantize_linear=True)
        restored = _dequantize_cache(quantized)
        assert len(restored) == 1
        assert isinstance(restored[0], ArraysCache)
        assert len(restored[0].cache) == 2

    def test_roundtrip_shape_and_values(self):
        original = _make_arrays_cache()
        quantized = _quantize_cache([original], quantize_kv=False, quantize_linear=True)
        restored = _dequantize_cache(quantized)[0]
        for i in (0, 1):
            assert restored.cache[i].shape == original.cache[i].shape
            # 8-bit affine quant of normal(0,1) data: small mean-abs error.
            assert _mean_abs_err(restored.cache[i], original.cache[i]) < 0.02

    def test_dtype_preservation(self):
        """Recurrent float32 and conv bfloat16 must each round-trip exactly."""
        original = _make_arrays_cache()
        assert original.cache[0].dtype == mx.bfloat16
        assert original.cache[1].dtype == mx.float32
        quantized = _quantize_cache([original], quantize_kv=False, quantize_linear=True)
        restored = _dequantize_cache(quantized)[0]
        assert restored.cache[0].dtype == mx.bfloat16
        assert restored.cache[1].dtype == mx.float32

    def test_subclass_preserved(self):
        """A MambaCache-like subclass must round-trip to the same class."""

        class _MambaLike(ArraysCache):
            pass

        original = _MambaLike(size=2)
        original.cache[0] = mx.zeros((1, 3, 128), dtype=mx.bfloat16)
        original.cache[1] = mx.random.normal((1, 4, 16, 64)).astype(mx.float32)
        quantized = _quantize_cache([original], quantize_kv=False, quantize_linear=True)
        restored = _dequantize_cache(quantized)[0]
        assert type(restored) is _MambaLike

    def test_none_slot_preserved(self):
        c = ArraysCache(size=2)
        c.cache[0] = None
        c.cache[1] = mx.random.normal((1, 4, 16, 64)).astype(mx.float32)
        mx.eval(c.cache[1])
        quantized = _quantize_cache([c], quantize_kv=False, quantize_linear=True)
        restored = _dequantize_cache(quantized)[0]
        assert restored.cache[0] is None
        assert restored.cache[1] is not None

    def test_4bit_roundtrip(self):
        original = _make_arrays_cache()
        quantized = _quantize_cache(
            [original],
            linear_bits=4,
            linear_group_size=64,
            quantize_kv=False,
            quantize_linear=True,
        )
        assert quantized[0].bits == 4
        restored = _dequantize_cache(quantized)[0]
        # 4-bit is lossier but still bounded.
        assert _mean_abs_err(restored.cache[1], original.cache[1]) < 0.15

    def test_raw_fallback_for_non_divisible_last_dim(self):
        """An array whose last dim is not divisible by group_size is stored
        raw and round-trips exactly."""
        c = ArraysCache(size=2)
        c.cache[0] = None
        c.cache[1] = mx.random.normal((1, 4, 16, 100)).astype(mx.float32)  # 100 % 64
        mx.eval(c.cache[1])
        quantized = _quantize_cache(
            [c], linear_group_size=64, quantize_kv=False, quantize_linear=True
        )
        restored = _dequantize_cache(quantized)[0]
        # Raw-stored → exact (only deep-copied).
        assert _mean_abs_err(restored.cache[1], c.cache[1]) == 0.0

    def test_empty_arrays_cache_passes_through(self):
        """An ArraysCache with all-None slots is not wrapped."""
        c = ArraysCache(size=2)  # cache = [None, None]
        quantized = _quantize_cache([c], quantize_kv=False, quantize_linear=True)
        assert quantized[0] is c


# ---------------------------------------------------------------------------
# Hybrid (3 linear + 1 full attention) cache
# ---------------------------------------------------------------------------


class TestHybridCache:
    def test_hybrid_3plus1_pattern(self):
        cache = [
            _make_arrays_cache(),
            _make_arrays_cache(),
            _make_arrays_cache(),
            _make_kv_layer(),
        ]
        quantized = _quantize_cache(cache, quantize_kv=True, quantize_linear=True)
        assert isinstance(quantized[0], _QuantizedArraysCacheWrapper)
        assert isinstance(quantized[1], _QuantizedArraysCacheWrapper)
        assert isinstance(quantized[2], _QuantizedArraysCacheWrapper)
        assert isinstance(quantized[3], _QuantizedCacheWrapper)

        restored = _dequantize_cache(quantized)
        assert isinstance(restored[0], ArraysCache)
        assert isinstance(restored[3], KVCache)

    def test_kv_only_still_deep_copies_linear(self):
        """With only KV quant on, linear ArraysCache layers must still be
        deep-copied on dequantize so model mutations don't corrupt the entry."""
        linear = _make_arrays_cache()
        cache = [linear, _make_kv_layer()]
        quantized = _quantize_cache(cache, quantize_kv=True, quantize_linear=False)
        assert quantized[0] is linear  # not wrapped
        restored = _dequantize_cache(quantized)
        assert isinstance(restored[0], ArraysCache)
        assert restored[0] is not linear  # deep-copied
        assert restored[0].cache[1] is not linear.cache[1]

    def test_independent_bits_for_kv_and_linear(self):
        cache = [_make_arrays_cache(), _make_kv_layer()]
        quantized = _quantize_cache(
            cache,
            bits=8,
            linear_bits=4,
            quantize_kv=True,
            quantize_linear=True,
        )
        assert quantized[1].bits == 8  # KV wrapper
        assert quantized[0].bits == 4  # linear wrapper


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


class TestMemoryAccounting:
    def test_wrapper_counted_and_smaller(self):
        original = _make_arrays_cache(conv_dim=128, hv=8, dv=64, dk=128)
        unquantized_bytes = estimate_kv_cache_memory([original])
        quantized = _quantize_cache([original], quantize_kv=False, quantize_linear=True)
        quantized_bytes = estimate_kv_cache_memory(quantized)
        assert quantized_bytes > 0  # wrapper must be counted, not 0
        # 8-bit + scales/biases of float32 source → well under half.
        assert quantized_bytes < 0.45 * unquantized_bytes


# ---------------------------------------------------------------------------
# MemoryCacheConfig
# ---------------------------------------------------------------------------


class TestMemoryCacheConfig:
    def test_linear_quant_off_by_default(self):
        config = MemoryCacheConfig()
        assert config.linear_state_quantize is False
        assert config.any_quantize is False

    def test_any_quantize_truth_table(self):
        assert MemoryCacheConfig(kv_quantize=True).any_quantize is True
        assert MemoryCacheConfig(linear_state_quantize=True).any_quantize is True
        assert MemoryCacheConfig().any_quantize is False

    def test_invalid_linear_bits_rejected(self):
        try:
            MemoryCacheConfig(linear_state_bits=3)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for linear_state_bits=3")


# ---------------------------------------------------------------------------
# Prefix cache store / fetch integration
# ---------------------------------------------------------------------------


class TestPrefixCacheIntegration:
    def _model(self):
        class FakeModel:
            pass

        return FakeModel()

    def test_store_fetch_with_linear_quantization(self):
        config = MemoryCacheConfig(
            kv_quantize=False,
            linear_state_quantize=True,
            kv_min_quantize_tokens=1,
            max_memory_mb=500,
            min_prefix_tokens=1,
        )
        pc = MemoryAwarePrefixCache(self._model(), config)

        cache = [_make_arrays_cache(), _make_arrays_cache(), _make_kv_layer()]
        tokens = list(range(50))
        assert pc.store(tokens, cache) is True

        fetched, remaining = pc.fetch(tokens)
        assert fetched is not None
        assert remaining == []
        assert isinstance(fetched[0], ArraysCache)
        assert isinstance(fetched[2], KVCache)
        # Linear state survived the quant/dequant roundtrip.
        assert _mean_abs_err(fetched[0].cache[1], cache[0].cache[1]) < 0.02

    def test_fetch_returns_independent_cache(self):
        """Two fetches of the same entry must not alias each other's state."""
        config = MemoryCacheConfig(
            kv_quantize=False,
            linear_state_quantize=True,
            kv_min_quantize_tokens=1,
            max_memory_mb=500,
            min_prefix_tokens=1,
        )
        pc = MemoryAwarePrefixCache(self._model(), config)
        cache = [_make_arrays_cache()]
        tokens = list(range(40))
        pc.store(tokens, cache)

        first, _ = pc.fetch(tokens)
        second, _ = pc.fetch(tokens)
        assert first[0] is not second[0]
        assert first[0].cache[1] is not second[0].cache[1]
