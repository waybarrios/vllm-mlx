# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchQuantizedKVCache and asymmetric KV cache quantization."""

import mlx.core as mx

from vllm_mlx.utils.quantized_kv_cache import (
    BatchQuantizedKVCache,
    asymmetric_quantized_sdpa,
    install_quantized_kv_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv(B=1, H=4, S=8, Dk=64, Dv=64, dtype=mx.float16):
    """Create random key and value tensors."""
    keys = mx.random.normal((B, H, S, Dk)).astype(dtype)
    values = mx.random.normal((B, H, S, Dv)).astype(dtype)
    return keys, values


# ---------------------------------------------------------------------------
# BatchQuantizedKVCache — basic operations
# ---------------------------------------------------------------------------


class TestBatchQuantizedKVCacheBasic:
    def test_init_empty(self):
        cache = BatchQuantizedKVCache([0, 0], k_bits=8, v_bits=4)
        assert cache.empty()
        assert cache.size() == 0
        assert cache.k_bits == 8
        assert cache.v_bits == 4

    def test_update_and_fetch_returns_quantized_tuples(self):
        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=8)
        keys, values = _make_kv(B=1, H=4, S=8, Dk=64, Dv=64)
        q_keys, q_values = cache.update_and_fetch(keys, values)

        # Should return 3-tuples (data, scales, biases)
        assert isinstance(q_keys, tuple) and len(q_keys) == 3
        assert isinstance(q_values, tuple) and len(q_values) == 3

        # data should be uint32
        assert q_keys[0].dtype == mx.uint32
        assert q_values[0].dtype == mx.uint32

        # Sequence dimension should match
        assert q_keys[0].shape[2] == 8
        assert q_values[0].shape[2] == 8
        assert cache.size() == 8
        assert not cache.empty()

    def test_asymmetric_k8_v4_bits(self):
        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=4)
        keys, values = _make_kv(B=1, H=4, S=16, Dk=64, Dv=64)
        q_keys, q_values = cache.update_and_fetch(keys, values)

        # K at 8-bit: 64 / (32/8) = 16 uint32 per head_dim
        assert q_keys[0].shape[-1] == 64 // (32 // 8)  # 16
        # V at 4-bit: 64 / (32/4) = 8 uint32 per head_dim
        assert q_values[0].shape[-1] == 64 // (32 // 4)  # 8

    def test_incremental_update(self):
        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=8)
        keys1, values1 = _make_kv(B=1, H=4, S=4)
        cache.update_and_fetch(keys1, values1)
        assert cache.size() == 4

        keys2, values2 = _make_kv(B=1, H=4, S=3)
        q_keys, q_values = cache.update_and_fetch(keys2, values2)
        assert cache.size() == 7
        assert q_keys[0].shape[2] == 7

    def test_nbytes_smaller_than_fp16(self):
        cache_q = BatchQuantizedKVCache([0], k_bits=8, v_bits=4)
        keys, values = _make_kv(B=1, H=4, S=256, Dk=64, Dv=64)
        cache_q.update_and_fetch(keys, values)

        q_nbytes = cache_q.nbytes
        fp16_nbytes = keys.nbytes + values.nbytes

        # Quantized should be significantly smaller
        assert q_nbytes < fp16_nbytes


# ---------------------------------------------------------------------------
# BatchQuantizedKVCache — batch operations
# ---------------------------------------------------------------------------


class TestBatchQuantizedKVCacheBatch:
    def test_make_mask(self):
        cache = BatchQuantizedKVCache([2, 0])
        keys, values = _make_kv(B=2, H=4, S=8)
        cache.update_and_fetch(keys, values)
        mask = cache.make_mask(1)
        assert mask is not None

    def test_trim(self):
        cache = BatchQuantizedKVCache([0])
        keys, values = _make_kv(B=1, H=4, S=10)
        cache.update_and_fetch(keys, values)
        assert cache.size() == 10

        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.size() == 7

    def test_filter_preserves_quantized_data(self):
        cache = BatchQuantizedKVCache([0, 0, 0], k_bits=8, v_bits=4)
        keys, values = _make_kv(B=3, H=4, S=8)
        cache.update_and_fetch(keys, values)

        cache.filter([0, 2])

        # Should have 2 sequences
        assert cache.keys[0].shape[0] == 2
        assert cache.values[0].shape[0] == 2
        assert cache.offset.shape[0] == 2

    def test_extract_returns_dequantized_kvcache(self):
        from mlx_lm.models.cache import KVCache

        cache = BatchQuantizedKVCache([0, 0], k_bits=8, v_bits=8)
        keys, values = _make_kv(B=2, H=4, S=8)
        cache.update_and_fetch(keys, values)

        extracted = cache.extract(0)
        assert isinstance(extracted, KVCache)
        assert extracted.keys is not None
        assert extracted.keys.dtype != mx.uint32  # dequantized
        assert extracted.keys.shape == (1, 4, 8, 64)
        assert extracted.offset == 8

    def test_merge_from_kvcaches(self):
        from mlx_lm.models.cache import KVCache

        caches = []
        for length in [5, 10, 3]:
            c = KVCache()
            k, v = _make_kv(B=1, H=4, S=length)
            c.keys = k
            c.values = v
            c.offset = length
            caches.append(c)

        merged = BatchQuantizedKVCache.merge(caches, k_bits=8, v_bits=4, group_size=64)
        assert merged.keys[0].shape[0] == 3  # batch size
        assert merged.size() == 10  # max length
        assert merged.k_bits == 8
        assert merged.v_bits == 4

    def test_extend_merges_quantized_batches(self):
        cache1 = BatchQuantizedKVCache([0], k_bits=8, v_bits=8)
        k1, v1 = _make_kv(B=1, H=4, S=8)
        cache1.update_and_fetch(k1, v1)

        cache2 = BatchQuantizedKVCache([0], k_bits=8, v_bits=8)
        k2, v2 = _make_kv(B=1, H=4, S=5)
        cache2.update_and_fetch(k2, v2)

        cache1.extend(cache2)
        assert cache1.keys[0].shape[0] == 2  # 2 sequences
        assert cache1.offset.shape[0] == 2

    def test_prepare_left_padding(self):
        cache = BatchQuantizedKVCache([0, 0])
        cache.prepare(left_padding=[3, 1])
        assert cache.left_padding[0].item() == 3
        assert cache.left_padding[1].item() == 1

    def test_finalize_rolls_quantized_data(self):
        cache = BatchQuantizedKVCache([0, 0])
        keys, values = _make_kv(B=2, H=4, S=8)
        cache.update_and_fetch(keys, values)

        # Set right padding
        cache._right_padding = mx.array([2, 0])
        cache.finalize()

        assert cache._right_padding is None
        # left_padding should have increased for first seq
        assert cache.left_padding[0].item() == 2


# ---------------------------------------------------------------------------
# asymmetric_quantized_sdpa
# ---------------------------------------------------------------------------


class TestAsymmetricQuantizedSDPA:
    def test_basic_sdpa(self):
        B, n_heads, S, D = 1, 4, 16, 64
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
        values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

        q_keys = mx.quantize(keys, group_size=64, bits=8)
        q_values = mx.quantize(values, group_size=64, bits=4)

        out = asymmetric_quantized_sdpa(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=None,
            group_size=64,
            k_bits=8,
            v_bits=4,
        )

        assert out.shape == (B, n_heads, 1, D)

    def test_gqa_repeats(self):
        """Test with grouped query attention (more query heads than KV heads)."""
        B, n_q_heads, n_kv_heads, S, D = 1, 8, 2, 16, 64
        queries = mx.random.normal((B, n_q_heads, 1, D)).astype(mx.float16)
        keys = mx.random.normal((B, n_kv_heads, S, D)).astype(mx.float16)
        values = mx.random.normal((B, n_kv_heads, S, D)).astype(mx.float16)

        q_keys = mx.quantize(keys, group_size=64, bits=8)
        q_values = mx.quantize(values, group_size=64, bits=8)

        out = asymmetric_quantized_sdpa(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=None,
            group_size=64,
            k_bits=8,
            v_bits=8,
        )

        assert out.shape == (B, n_q_heads, 1, D)

    def test_with_mask(self):
        B, n_heads, S, D = 1, 4, 8, 64
        queries = mx.random.normal((B, n_heads, 2, D)).astype(mx.float16)
        keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
        values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

        q_keys = mx.quantize(keys, group_size=64, bits=8)
        q_values = mx.quantize(values, group_size=64, bits=4)

        # Create a simple additive mask
        mask = mx.zeros((2, S))

        out = asymmetric_quantized_sdpa(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=mask,
            group_size=64,
            k_bits=8,
            v_bits=4,
        )

        assert out.shape == (B, n_heads, 2, D)


# ---------------------------------------------------------------------------
# SDPA routing via install patch
# ---------------------------------------------------------------------------


class TestSDPARouting:
    def test_sdpa_routing_asymmetric(self):
        """Verify SDPA dispatches to asymmetric_quantized_sdpa for
        caches with k_bits attribute."""
        import importlib

        # Save originals
        base_module = importlib.import_module("mlx_lm.models.base")
        original_sdpa = base_module.scaled_dot_product_attention

        try:
            install_quantized_kv_cache(k_bits=8, v_bits=4, group_size=64)

            # Create a mock cache with k_bits
            class MockCache:
                k_bits = 8
                v_bits = 4
                group_size = 64

            B, n_heads, S, D = 1, 4, 8, 64
            queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
            keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
            values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

            q_keys = mx.quantize(keys, group_size=64, bits=8)
            q_values = mx.quantize(values, group_size=64, bits=4)

            out = base_module.scaled_dot_product_attention(
                queries, q_keys, q_values, MockCache(), D**-0.5, None
            )
            assert out.shape == (B, n_heads, 1, D)

        finally:
            # Restore original
            base_module.scaled_dot_product_attention = original_sdpa

    def test_sdpa_routing_symmetric_fallback(self):
        """Verify SDPA still handles QuantizedKVCache (symmetric bits)."""
        import importlib

        base_module = importlib.import_module("mlx_lm.models.base")
        original_sdpa = base_module.scaled_dot_product_attention

        try:
            install_quantized_kv_cache(k_bits=8, v_bits=8, group_size=64)

            # Mock QuantizedKVCache (has bits but not k_bits)
            class MockSymCache:
                bits = 8
                group_size = 64

            B, n_heads, S, D = 1, 4, 8, 64
            queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
            keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
            values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

            q_keys = mx.quantize(keys, group_size=64, bits=8)
            q_values = mx.quantize(values, group_size=64, bits=8)

            out = base_module.scaled_dot_product_attention(
                queries, q_keys, q_values, MockSymCache(), D**-0.5, None
            )
            assert out.shape == (B, n_heads, 1, D)

        finally:
            base_module.scaled_dot_product_attention = original_sdpa

    def test_sdpa_routing_fp16_fallback(self):
        """Verify SDPA falls back to mx.fast for non-quantized caches."""
        import importlib

        base_module = importlib.import_module("mlx_lm.models.base")
        original_sdpa = base_module.scaled_dot_product_attention

        try:
            install_quantized_kv_cache(k_bits=8, v_bits=8, group_size=64)

            # Mock normal cache (no bits or k_bits)
            class MockNormalCache:
                pass

            B, n_heads, S, D = 1, 4, 8, 64
            queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
            keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
            values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

            out = base_module.scaled_dot_product_attention(
                queries, keys, values, MockNormalCache(), D**-0.5, None
            )
            assert out.shape == (B, n_heads, 1, D)

        finally:
            base_module.scaled_dot_product_attention = original_sdpa


# ---------------------------------------------------------------------------
# Memory cache wrapper asymmetric bits
# ---------------------------------------------------------------------------


class TestMemoryCacheAsymmetric:
    def test_quantized_cache_wrapper_asymmetric(self):
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _QuantizedCacheWrapper

        cache = KVCache()
        k, v = _make_kv(B=1, H=4, S=32)
        cache.keys = k
        cache.values = v
        cache.offset = 32

        wrapper = _QuantizedCacheWrapper(
            cache, bits=8, group_size=64, k_bits=8, v_bits=4
        )
        assert wrapper.k_bits == 8
        assert wrapper.v_bits == 4
        # K at 8-bit
        assert wrapper.keys[0].dtype == mx.uint32
        # V at 4-bit (different packed size)
        assert wrapper.values[0].dtype == mx.uint32
        # V should have fewer uint32 elements per head_dim
        assert wrapper.values[0].shape[-1] < wrapper.keys[0].shape[-1]

    def test_quantize_dequantize_roundtrip_asymmetric(self):
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _dequantize_cache, _quantize_cache

        cache = KVCache()
        k, v = _make_kv(B=1, H=4, S=64)
        cache.keys = k
        cache.values = v
        cache.offset = 64

        quantized = _quantize_cache([cache], bits=8, group_size=64, k_bits=8, v_bits=4)
        dequantized = _dequantize_cache(quantized)

        assert isinstance(dequantized[0], KVCache)
        assert dequantized[0].keys.shape == k.shape
        assert dequantized[0].values.shape == v.shape
        assert dequantized[0].offset == 64

    def test_memory_cache_config_asymmetric(self):
        from vllm_mlx.memory_cache import MemoryCacheConfig

        config = MemoryCacheConfig(
            kv_quantize=True,
            kv_bits=8,
            kv_k_bits=8,
            kv_v_bits=4,
        )
        assert config.kv_k_bits == 8
        assert config.kv_v_bits == 4


# ---------------------------------------------------------------------------
# Hadamard rotation (QuaRot)
# ---------------------------------------------------------------------------


class TestHadamardQuaRot:
    def test_hadamard_update_and_fetch(self):
        """Hadamard rotation should produce quantized output."""
        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=4, use_hadamard=True)
        keys, values = _make_kv(B=1, H=4, S=8, Dk=128, Dv=128)
        q_keys, q_values = cache.update_and_fetch(keys, values)

        assert isinstance(q_keys, tuple) and len(q_keys) == 3
        assert isinstance(q_values, tuple) and len(q_values) == 3
        assert q_keys[0].dtype == mx.uint32
        assert cache.use_hadamard is True

    def test_hadamard_sdpa_quality(self):
        """QuaRot SDPA should be closer to FP16 reference than plain."""
        B, n_heads, S, D = 1, 4, 32, 128
        mx.random.seed(42)
        queries = mx.random.normal((B, n_heads, 1, D)).astype(mx.float16)
        keys = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)
        values = mx.random.normal((B, n_heads, S, D)).astype(mx.float16)

        # FP16 reference
        ref = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=D**-0.5, mask=None
        )
        mx.eval(ref)

        # Plain quantized SDPA (no Hadamard)
        q_keys_plain = mx.quantize(keys, group_size=64, bits=8)
        q_values_plain = mx.quantize(values, group_size=64, bits=4)
        out_plain = asymmetric_quantized_sdpa(
            queries,
            q_keys_plain,
            q_values_plain,
            scale=D**-0.5,
            mask=None,
            group_size=64,
            k_bits=8,
            v_bits=4,
            use_hadamard=False,
        )
        mx.eval(out_plain)

        # QuaRot SDPA (with Hadamard)
        keys_rot = mx.hadamard_transform(keys)
        values_rot = mx.hadamard_transform(values)
        q_keys_rot = mx.quantize(keys_rot, group_size=64, bits=8)
        q_values_rot = mx.quantize(values_rot, group_size=64, bits=4)
        out_quarot = asymmetric_quantized_sdpa(
            queries,
            q_keys_rot,
            q_values_rot,
            scale=D**-0.5,
            mask=None,
            group_size=64,
            k_bits=8,
            v_bits=4,
            use_hadamard=True,
        )
        mx.eval(out_quarot)

        # QuaRot should be closer to reference
        err_plain = mx.abs(ref - out_plain).mean().item()
        err_quarot = mx.abs(ref - out_quarot).mean().item()
        assert err_quarot < err_plain, (
            f"QuaRot error {err_quarot:.6f} should be less than "
            f"plain error {err_plain:.6f}"
        )

    def test_hadamard_extract_unrotates(self):
        """extract() should return un-rotated (original-space) KVCache."""
        from mlx_lm.models.cache import KVCache

        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=8, use_hadamard=True)
        keys, values = _make_kv(B=1, H=4, S=16, Dk=128, Dv=128)
        cache.update_and_fetch(keys, values)

        extracted = cache.extract(0)
        assert isinstance(extracted, KVCache)
        # Extracted keys should be close to original (un-rotated),
        # within quantization error
        mx.eval(extracted.keys, keys)
        err = mx.abs(extracted.keys - keys).mean().item()
        # 8-bit quantization + Hadamard roundtrip should have small error
        assert err < 0.1, f"Extract un-rotation error too large: {err:.4f}"

    def test_hadamard_merge_rotates(self):
        """merge() with use_hadamard should rotate before quantization."""
        from mlx_lm.models.cache import KVCache

        caches = []
        for length in [8, 12]:
            c = KVCache()
            k, v = _make_kv(B=1, H=4, S=length, Dk=128, Dv=128)
            c.keys = k
            c.values = v
            c.offset = length
            caches.append(c)

        merged = BatchQuantizedKVCache.merge(
            caches, k_bits=8, v_bits=8, group_size=64, use_hadamard=True
        )
        assert merged.use_hadamard is True
        assert merged.keys[0].dtype == mx.uint32
        assert merged.size() == 12  # max length

    def test_hadamard_roundtrip(self):
        """extract(merge(caches)) should approximate original data."""
        from mlx_lm.models.cache import KVCache

        original_k, original_v = _make_kv(B=1, H=4, S=16, Dk=128, Dv=128)
        c = KVCache()
        c.keys = original_k
        c.values = original_v
        c.offset = 16

        merged = BatchQuantizedKVCache.merge(
            [c], k_bits=8, v_bits=8, group_size=64, use_hadamard=True
        )
        extracted = merged.extract(0)

        mx.eval(extracted.keys, extracted.values, original_k, original_v)
        k_err = mx.abs(extracted.keys - original_k).mean().item()
        v_err = mx.abs(extracted.values - original_v).mean().item()
        assert k_err < 0.1, f"Key roundtrip error too large: {k_err:.4f}"
        assert v_err < 0.1, f"Value roundtrip error too large: {v_err:.4f}"

    def test_hadamard_disabled_by_default(self):
        """Default cache should not use Hadamard."""
        cache = BatchQuantizedKVCache([0], k_bits=8, v_bits=8)
        assert cache.use_hadamard is False


# ---------------------------------------------------------------------------
# max_kv_size interaction with quantization
# ---------------------------------------------------------------------------


class TestMaxKvSizeQuantizationInteraction:
    def test_make_cache_with_max_kv_size_logs_warning(self, caplog):
        """When max_kv_size is set and model lacks make_cache(),
        _patched_make_cache should return BatchRotatingKVCache
        and log a warning about skipped quantization."""
        import importlib
        import logging

        from mlx_lm.generate import BatchRotatingKVCache

        gen_module = importlib.import_module("mlx_lm.generate")
        original_make_cache = gen_module._make_cache

        try:
            install_quantized_kv_cache(k_bits=8, v_bits=4, group_size=64)

            # Mock model WITHOUT make_cache() method
            class MockModel:
                layers = [None, None, None, None]

            with caplog.at_level(
                logging.WARNING, logger="vllm_mlx.utils.quantized_kv_cache"
            ):
                caches = gen_module._make_cache(MockModel(), [0, 0], max_kv_size=1024)

            # All layers should be BatchRotatingKVCache (unquantized)
            assert len(caches) == 4
            for c in caches:
                assert isinstance(c, BatchRotatingKVCache)
                assert not hasattr(c, "k_bits")

            # Warning should mention quantization bypass
            assert any("max_kv_size" in rec.message for rec in caplog.records)

        finally:
            gen_module._make_cache = original_make_cache

    def test_make_cache_model_with_make_cache_uses_quantization(self):
        """When model has make_cache(), KVCache layers get quantized
        regardless of max_kv_size parameter."""
        import importlib

        from mlx_lm.models.cache import KVCache as _KVCache

        gen_module = importlib.import_module("mlx_lm.generate")
        original_make_cache = gen_module._make_cache

        try:
            install_quantized_kv_cache(k_bits=8, v_bits=4, group_size=64)

            # Mock model WITH make_cache() returning KVCache layers
            class MockModel:
                layers = [None, None]

                def make_cache(self):
                    return [_KVCache(), _KVCache()]

            caches = gen_module._make_cache(MockModel(), [0, 0], max_kv_size=1024)

            # KVCache layers should be converted to BatchQuantizedKVCache
            assert len(caches) == 2
            for c in caches:
                assert isinstance(c, BatchQuantizedKVCache)
                assert c.k_bits == 8
                assert c.v_bits == 4

        finally:
            gen_module._make_cache = original_make_cache
