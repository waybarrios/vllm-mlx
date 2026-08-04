# SPDX-License-Identifier: Apache-2.0
"""MLX-dependent regression tests for the LCP trim contamination fix (#384).

These tests use real ``mlx_lm.models.cache.KVCache`` / ``RotatingKVCache``
objects backed by ``mlx.core`` arrays.  They run only on the apple-silicon
CI matrix (and locally on M-series hardware); the Linux ``test-matrix``
job excludes this file because MLX has no Linux distribution.
"""

from unittest.mock import MagicMock


class TestTrimCacheOffset:
    """Tests for ``_trim_cache_offset``, focused on the LCP contamination fix.

    Regression: when the LCP fetch path trimmed a cache entry by shrinking
    the offset while still sharing the underlying (oversized) key/value
    arrays, downstream attention layers that read ``cache.state`` directly
    (e.g. Gemma 4 KV-shared layers) could see stale tokens from the previous
    owner of the entry.  See issue #384.  The fix slices the arrays down to
    new_offset so no memory beyond the new boundary remains accessible.
    """

    def test_plain_kv_cache_array_sliced_to_new_offset(self):
        """Plain-KVCache-like layer: after trim, keys.shape[-2] == new_offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        # Pretend a previous request wrote 500 tokens worth of data
        layer.keys = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(1, 4, 500, 8)
        layer.values = mx.arange(1 * 4 * 500 * 8, dtype=mx.float32).reshape(
            1, 4, 500, 8
        )
        layer.offset = 500

        # New request shares only the first 60 tokens as prefix
        trim_by = 500 - 60
        trimmed = _trim_cache_offset([layer], trim_by)
        tc = trimmed[0]

        assert tc.offset == 60
        # The underlying array MUST be shrunk, not just the offset pointer.
        # Otherwise Gemma 4's cache.state-reading layers would see positions
        # 60..500 filled with the previous request's tokens.
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_plain_kv_cache_no_stale_tokens_visible_via_state(self):
        """A layer that reads the full cache.state must not see tokens past new_offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        # Positions 0..60: shared prefix (same for everyone).  Positions 60..500:
        # private content from a previous request that must NOT leak.
        shared = mx.ones((1, 4, 60, 8), dtype=mx.float32)
        private = mx.full((1, 4, 440, 8), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = layer.keys
        layer.offset = 500

        tc = _trim_cache_offset([layer], 500 - 60)[0]

        # cache.state is what KV-shared layers read directly.
        keys_view, _ = tc.state
        assert keys_view.shape[-2] == 60
        # No "7.0" tokens anywhere — private content was excluded.
        assert float(mx.max(keys_view).item()) == 1.0

    def test_plain_kv_cache_no_trim_preserves_array(self):
        """If trim_by == 0 or offset already equals shape, array is untouched."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 100, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 100, 8), dtype=mx.float32)
        layer.offset = 100

        tc = _trim_cache_offset([layer], 0)[0]

        assert tc.offset == 100
        assert tc.keys.shape[-2] == 100

    def test_plain_kv_cache_trim_by_exceeds_offset_clamps_to_zero(self):
        """trim_by larger than offset yields an empty-but-valid trimmed cache."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 80, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 80, 8), dtype=mx.float32)
        layer.offset = 80

        tc = _trim_cache_offset([layer], 1000)[0]

        assert tc.offset == 0
        assert tc.keys.shape[-2] == 0
        assert tc.values.shape[-2] == 0

    def test_plain_kv_cache_stored_entry_unaffected_after_trim(self):
        """Calling _trim_cache_offset must not mutate the source layer in place.

        The stored prefix-cache entry is the source here; a later lookup for
        a different request should get the same pristine data.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = KVCache()
        full = mx.arange(1 * 2 * 200 * 4, dtype=mx.float32).reshape(1, 2, 200, 4)
        layer.keys = full
        layer.values = full
        layer.offset = 200
        original_shape = layer.keys.shape

        _trim_cache_offset([layer], 150)

        # Source entry keeps its full shape and offset.
        assert layer.keys.shape == original_shape
        assert layer.values.shape == original_shape
        assert layer.offset == 200

    def test_plain_kv_cache_in_place_write_does_not_corrupt_source(self):
        """After trim, writing through the returned cache must not leak into
        the stored entry.  This is the direct semantics of the fix: the stored
        prefix-cache entry has to survive concurrent use by other requests.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        # Source stored entry: positions 0..300 holding 5.0.
        layer = KVCache()
        layer.keys = mx.full((1, 2, 300, 4), 5.0, dtype=mx.float32)
        layer.values = mx.full((1, 2, 300, 4), 5.0, dtype=mx.float32)
        layer.offset = 300

        # New request shares first 50 tokens.
        tc = _trim_cache_offset([layer], 300 - 50)[0]

        # The trimmed cache now only has 50 tokens.  Writing new tokens via
        # update_and_fetch allocates a new array (because prev + N > current
        # shape) and does not touch the source.
        new_keys = mx.zeros((1, 2, 10, 4), dtype=mx.float32)
        new_values = mx.zeros((1, 2, 10, 4), dtype=mx.float32)
        tc.update_and_fetch(new_keys, new_values)

        # Source remains untouched (all 5.0 values preserved across full range).
        assert layer.keys.shape[-2] == 300
        assert float(mx.min(layer.keys).item()) == 5.0
        assert float(mx.max(layer.keys).item()) == 5.0

    def test_plain_kv_cache_multiple_layers_all_sliced(self):
        """Caches with several KVCache layers: every layer gets sliced."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layers = []
        for _ in range(5):
            layer = KVCache()
            layer.keys = mx.ones((1, 4, 200, 8), dtype=mx.float32)
            layer.values = mx.ones((1, 4, 200, 8), dtype=mx.float32)
            layer.offset = 200
            layers.append(layer)

        trimmed = _trim_cache_offset(layers, 150)

        assert len(trimmed) == 5
        for tc in trimmed:
            assert tc.offset == 50
            assert tc.keys.shape[-2] == 50
            assert tc.values.shape[-2] == 50

    def test_plain_kv_cache_slice_works_for_float16_and_bfloat16(self):
        """Fix must be dtype-agnostic so quantized / mixed-precision KV caches
        receive the same treatment as fp32.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        for dtype in (mx.float16, mx.bfloat16):
            layer = KVCache()
            layer.keys = mx.ones((1, 2, 120, 4), dtype=dtype)
            layer.values = mx.ones((1, 2, 120, 4), dtype=dtype)
            layer.offset = 120

            tc = _trim_cache_offset([layer], 80)[0]

            assert tc.offset == 40, f"dtype={dtype}"
            assert tc.keys.shape[-2] == 40, f"dtype={dtype}"
            assert tc.keys.dtype == dtype, f"dtype={dtype}"

    def test_plain_kv_cache_rotating_layers_unchanged_behavior(self):
        """RotatingKVCache was already trimming correctly before this fix.
        The plain-KVCache branch is the only one that changed; the rotating
        branch is exercised here to catch regressions.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        from vllm_mlx.memory_cache import _trim_cache_offset

        layer = RotatingKVCache(max_size=128, keep=0)
        # Layer already rotated once: offset=200, buffer holds max_size entries.
        layer.keys = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 128, 8), dtype=mx.float32)
        layer.offset = 200
        layer._idx = 128

        tc = _trim_cache_offset([layer], 100)[0]

        # Offset dropped by trim_by, clamped at >= 0.
        assert tc.offset == 100
        # Rotating path materialises a buffer whose shape matches new_offset
        # (padding with zeros if needed).  It must not come back as None.
        assert tc.keys is not None
        assert tc.values is not None
        # Dtype preserved through trim.
        assert tc.keys.dtype == mx.float32
        # Type-specific attrs preserved.
        assert hasattr(tc, "max_size")
        assert tc.max_size == 128

    def test_fetch_returns_sliced_cache_on_lcp_match(self):
        """End-to-end: MemoryAwarePrefixCache.fetch on a request that shares
        only a prefix with a longer stored entry must return a cache whose
        arrays are already sliced down.  This is the full regression of the
        #384 scenario above the unit level.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        model = MagicMock()
        cache = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1),
        )

        # Stored: tokens [1..120] with 120 positions of KV data, the first 60
        # tokens being the shared prefix (all 1.0), the last 60 private (7.0).
        stored_layer = KVCache()
        shared = mx.ones((1, 2, 60, 4), dtype=mx.float32)
        private = mx.full((1, 2, 60, 4), 7.0, dtype=mx.float32)
        stored_layer.keys = mx.concatenate([shared, private], axis=2)
        stored_layer.values = stored_layer.keys
        stored_layer.offset = 120
        cache.store(list(range(1, 121)), [stored_layer])

        # New request: tokens [1..59] + [999, 1000, 1001] — first 59 tokens
        # match, then diverge.  LCP is 59.
        new_tokens = list(range(1, 60)) + [999, 1000, 1001]
        fetched, remaining = cache.fetch(new_tokens)

        assert fetched is not None
        tc = fetched[0]
        # LCP of 59 (the divergent tokens are stripped).
        assert tc.offset == 59
        assert tc.keys.shape[-2] == 59
        # Critical: the "7.0" private content from the stored entry must NOT
        # be visible anywhere in the returned cache (this is what caused the
        # cross-request contamination in #384).
        assert float(mx.max(tc.keys).item()) == 1.0
        assert remaining == [999, 1000, 1001]


class TestDequantizeCacheSlice:
    """Tests for _dequantize_cache slicing after dequantization.

    When KV cache quantization is enabled (--kv-cache-quantization), the
    prefix cache stores _QuantizedCacheWrapper layers.  After LCP trim
    reduces the offset, _dequantize_cache must slice the dequantized arrays
    down to offset to prevent readers that bypass offset (e.g. Gemma 4's
    KV-shared layers reading cache.state) from seeing stale tokens.

    This is the quantized-cache counterpart of the plain-KVCache fix
    tested in TestTrimCacheOffset above.
    """

    def test_dequantize_slices_to_offset(self):
        """After trim + dequantize, keys/values shape[-2] == offset."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        # Build a KVCache with 500 tokens, quantize it, then trim to 60.
        layer = KVCache()
        layer.keys = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 512, 64), dtype=mx.float32)
        layer.offset = 512
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        trimmed = _trim_cache_offset([qw], 512 - 60)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        assert tc.values.shape[-2] == 60

    def test_dequantize_no_stale_tokens_via_state(self):
        """Stale tokens from a previous request must not be visible via cache.state."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        layer = KVCache()
        # First 64 positions: shared prefix (1.0), next 448: private (7.0)
        shared = mx.ones((1, 4, 64, 64), dtype=mx.float32)
        private = mx.full((1, 4, 448, 64), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = mx.concatenate([shared, private], axis=2)
        layer.offset = 512
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        trimmed = _trim_cache_offset([qw], 512 - 64)
        result = _dequantize_cache(trimmed)

        tc = result[0]
        keys_view, _ = tc.state
        assert keys_view.shape[-2] == 64
        # Dequantized values are approximate (quantization error), but should
        # be close to 1.0 (the shared prefix), never near 7.0 (the private data).
        assert float(mx.max(keys_view).item()) < 2.0

    def test_dequantize_no_trim_preserves_full_array(self):
        """When offset == shape[-2], no slicing occurs."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
        )

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 128, 64), dtype=mx.float32)
        layer.offset = 128
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        result = _dequantize_cache([qw])

        tc = result[0]
        assert tc.offset == 128
        assert tc.keys.shape[-2] == 128
        assert tc.values.shape[-2] == 128

    def test_dequantize_source_unaffected(self):
        """Dequantizing must not mutate the stored quantized wrapper."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _QuantizedCacheWrapper,
            _dequantize_cache,
            _trim_cache_offset,
        )

        layer = KVCache()
        layer.keys = mx.ones((1, 4, 256, 64), dtype=mx.float32)
        layer.values = mx.ones((1, 4, 256, 64), dtype=mx.float32)
        layer.offset = 256
        mx.eval(layer.keys, layer.values)

        qw = _QuantizedCacheWrapper(layer, bits=8, group_size=64)
        original_offset = qw.offset
        original_keys_shape = qw.keys[0].shape  # quantized data tuple

        trimmed = _trim_cache_offset([qw], 192)
        _dequantize_cache(trimmed)

        # Source wrapper unchanged
        assert qw.offset == original_offset
        assert qw.keys[0].shape == original_keys_shape

    def test_dequantize_end_to_end_fetch_with_quantization(self):
        """End-to-end: store with kv_quantize=True, fetch with LCP, verify no stale data."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            MemoryAwarePrefixCache,
            MemoryCacheConfig,
        )

        model = MagicMock()
        pc = MemoryAwarePrefixCache(
            model,
            MemoryCacheConfig(
                max_memory_mb=64,
                max_entries=10,
                kv_quantize=True,
                kv_bits=8,
                kv_group_size=64,
                kv_min_quantize_tokens=0,
                min_prefix_tokens=1,
            ),
        )

        # Store a KVCache with 128 tokens — store() quantizes automatically.
        layer = KVCache()
        shared = mx.ones((1, 2, 64, 64), dtype=mx.float32)
        private = mx.full((1, 2, 64, 64), 7.0, dtype=mx.float32)
        layer.keys = mx.concatenate([shared, private], axis=2)
        layer.values = mx.concatenate([shared, private], axis=2)
        layer.offset = 128
        mx.eval(layer.keys, layer.values)

        pc.store(list(range(1, 129)), [layer])

        # Fetch with partial match (first 60 tokens match, then diverge).
        # fetch() dequantizes automatically when kv_quantize=True.
        new_tokens = list(range(1, 61)) + [999, 1000]
        fetched, remaining = pc.fetch(new_tokens)

        assert fetched is not None
        tc = fetched[0]
        assert tc.offset == 60
        assert tc.keys.shape[-2] == 60
        # No private data (7.0) visible — only shared prefix (~1.0 with quantization noise)
        assert float(mx.max(tc.keys).item()) < 2.0
        assert remaining == [999, 1000]


class TestDetachCacheForStorage:
    """Tests for ``_detach_cache_for_storage``: stored entries must not alias
    live batch state or retain lazy computation graphs.

    Regression: ``MemoryAwarePrefixCache.store()`` stored per-request cache
    layers by reference.  For hybrid models (e.g. Qwen3.5/3.6, whose
    ``ArraysCache`` layers expose their mutable ``cache`` list via
    ``.state``), the stored entry aliased the live state container (same
    class of bug as the SimpleEngine snapshot aliasing, #575) and — because
    the extracted per-request arrays are lazy slices of batch-wide arrays —
    retained the entire upstream computation graph.  Under sustained traffic
    each stored entry pinned Metal buffer handles roughly proportional to
    generated tokens, until the process hit the device resource limit
    (``[metal::malloc] Resource limit (N) exceeded``) and aborted.
    """

    def test_arrays_cache_container_not_aliased(self):
        """The stored ArraysCache snapshot must not follow later mutation."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        parent = mx.arange(32, dtype=mx.float32).reshape(4, 8)
        mx.eval(parent)
        lazy = parent[1:2]
        for _ in range(5):
            lazy = lazy + 1
        expected = [[float(v) + 5 for v in range(8, 16)]]

        layer = ArraysCache(size=1)
        layer[0] = lazy

        detached = _detach_cache_for_storage([layer])

        assert detached[0] is not layer
        assert detached[0].cache is not layer.cache
        assert detached[0][0].tolist() == expected

        # Simulate the batch generator advancing the live state after the
        # snapshot was stored — the stored copy must not change.
        layer[0] = mx.zeros((1, 8))
        assert detached[0][0].tolist() == expected

    def test_kv_cache_layer_snapshotted_with_equal_arrays(self):
        """KVCache layers are snapshotted; array contents are preserved."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        layer = KVCache()
        layer.keys = mx.arange(1 * 2 * 4 * 3, dtype=mx.float32).reshape(1, 2, 4, 3)
        layer.values = mx.ones((1, 2, 4, 3))
        layer.offset = 4

        detached = _detach_cache_for_storage([layer])

        assert detached[0] is not layer
        assert detached[0].offset == 4
        assert detached[0].keys.tolist() == layer.keys.tolist()
        assert detached[0].values.tolist() == layer.values.tolist()

    def test_lazy_graph_is_cut(self):
        """Detaching must not retain the upstream lazy graph / parent buffers.

        Builds a long lazy op chain rooted at a large parent array, detaches,
        then drops every reference except the detached copy.  If the graph
        were retained, active memory would still include the parent (8MB+);
        the detached copy itself is only 16KB.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        mx.eval(mx.zeros((1,)))
        base = mx.get_active_memory()

        big_parent = mx.random.normal((512, 4096))  # ~8MB
        chain = big_parent[0:1]
        for _ in range(200):
            chain = chain * 1.0001
        layer = ArraysCache(size=1)
        layer[0] = chain

        detached = _detach_cache_for_storage([layer])

        del big_parent, chain, layer
        mx.eval(mx.zeros((1,)))

        retained = mx.get_active_memory() - base
        assert retained < 1_000_000, (
            f"lazy graph retained: {retained / 1e6:.1f}MB still active "
            f"after dropping originals (expected < 1MB)"
        )
        assert detached[0][0].shape == (1, 4096)

    def test_cache_list_children_snapshotted_without_write_through(self):
        """CacheList's ``state`` setter writes through to child caches in
        place; the detach path must snapshot children recursively instead."""
        import mlx.core as mx
        from mlx_lm.models.cache import CacheList, KVCache

        from vllm_mlx.memory_cache import _detach_cache_for_storage

        inner = KVCache()
        inner.keys = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)
        inner.values = mx.ones((1, 2, 3, 4))
        inner.offset = 3
        original_keys = inner.keys

        cl = CacheList(inner)

        detached = _detach_cache_for_storage([cl])

        assert detached[0] is not cl
        assert detached[0].caches is not cl.caches
        assert detached[0].caches[0] is not inner
        assert detached[0].caches[0].keys.tolist() == original_keys.tolist()
        # The live child must be untouched (no setter write-through)
        assert cl.caches[0] is inner
        assert inner.keys is original_keys

    def test_slots_class_snapshotted(self):
        """Layers using ``__slots__`` (no ``__dict__``), like
        ``_QuantizedCacheWrapper``, must snapshot without crashing."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _detach_cache_for_storage,
            _QuantizedCacheWrapper,
        )

        kv = KVCache()
        kv.keys = mx.ones((1, 2, 4, 64))
        kv.values = mx.ones((1, 2, 4, 64))
        kv.offset = 4
        mx.eval(kv.keys, kv.values)
        wrapper = _QuantizedCacheWrapper(kv, bits=8, group_size=32)

        detached = _detach_cache_for_storage([wrapper])

        assert detached[0] is not wrapper
        assert isinstance(detached[0], _QuantizedCacheWrapper)
        assert detached[0].offset == wrapper.offset
        # mx.quantize returns a (data, scales, biases) container; the
        # snapshot must preserve its type and contents.
        assert type(detached[0].keys) is type(wrapper.keys)
        assert len(detached[0].keys) == len(wrapper.keys)
        for got, orig in zip(detached[0].keys, wrapper.keys):
            assert got.tolist() == orig.tolist()

    def test_undetachable_layer_falls_back_to_reference(self):
        """A layer that cannot be snapshotted is stored by reference with a
        warning instead of failing the whole store()."""
        from vllm_mlx.memory_cache import _detach_cache_for_storage

        class Undetachable:
            def __init__(self):
                self.values = None
                self.offset = 0

            @property
            def keys(self):  # read-only: snapshot assignment must fail
                return None

        layer = Undetachable()

        detached = _detach_cache_for_storage([layer])

        assert detached[0] is layer
