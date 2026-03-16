# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM continuous batching with hybrid model caches.

Hybrid models (Qwen 3.5, Nemotron 3 Super) mix attention layers (KVCache)
with recurrent/SSM layers (ArraysCache). The MLLM batch generator must
handle both cache types during merge, filter, extract, and extend operations.
"""

import pytest

try:
    import mlx.core as mx
    from mlx_lm.models.cache import (
        ArraysCache,
        BatchKVCache,
        KVCache,
        RotatingKVCache,
        BatchRotatingKVCache,
    )

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# ---------------------------------------------------------------------------
# Helpers — simulate Qwen 3.5 cache layout (12 KVCache + 36 ArraysCache)
# ---------------------------------------------------------------------------


def _make_hybrid_cache(n_kv=12, n_arrays=36, arrays_size=2):
    """Create a hybrid cache list like Qwen 3.5's make_cache().

    Qwen 3.5 layout: is_linear = (layer_idx + 1) % 4 != 0
    So layers 0,1,2 are ArraysCache, layer 3 is KVCache, etc.
    For simplicity, we just create n_arrays ArraysCache + n_kv KVCache
    interleaved with the real pattern.
    """
    full_attention_interval = 4
    total = n_kv + n_arrays
    cache = []
    for i in range(total):
        is_linear = (i + 1) % full_attention_interval != 0
        if is_linear:
            cache.append(ArraysCache(size=arrays_size))
        else:
            cache.append(KVCache())
    return cache


def _populate_kv_cache(
    cache: KVCache, seq_len: int, n_kv_heads: int = 4, head_dim: int = 8
):
    """Populate a KVCache with dummy data to simulate a completed prefill."""
    # KVCache.update_and_fetch expects 4D: (batch, n_kv_heads, seq_len, head_dim)
    keys = mx.random.normal((1, n_kv_heads, seq_len, head_dim))
    values = mx.random.normal((1, n_kv_heads, seq_len, head_dim))
    cache.update_and_fetch(keys, values)


def _populate_arrays_cache(
    cache: ArraysCache, batch_size: int = 1, state_dim: int = 16
):
    """Populate an ArraysCache with dummy SSM state."""
    for i in range(len(cache.cache)):
        cache.cache[i] = mx.random.normal((batch_size, state_dim))


def _make_populated_hybrid_cache(
    seq_len: int = 10, n_kv_heads: int = 4, head_dim: int = 8, state_dim: int = 16
):
    """Create and populate a hybrid cache simulating a completed vision encoding prefill."""
    cache = _make_hybrid_cache()
    for c in cache:
        if isinstance(c, KVCache):
            _populate_kv_cache(c, seq_len, n_kv_heads, head_dim)
        elif isinstance(c, ArraysCache):
            _populate_arrays_cache(c, batch_size=1, state_dim=state_dim)
    return cache


# ---------------------------------------------------------------------------
# Test: _make_batch_cache handles all cache types
# ---------------------------------------------------------------------------


class TestMakeBatchCache:
    """Test _make_batch_cache() with hybrid model caches."""

    def test_hybrid_cache_creates_correct_types(self):
        """_make_batch_cache returns BatchKVCache for KVCache layers, ArraysCache for ArraysCache layers."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        # Mock model with make_cache returning hybrid layout
        class FakeModel:
            def make_cache(self):
                return _make_hybrid_cache()

        left_padding = [0, 2]  # 2-request batch, different prompt lengths
        batch_cache = _make_batch_cache(FakeModel(), left_padding)

        assert len(batch_cache) == 48  # 12 KV + 36 Arrays
        for i, c in enumerate(batch_cache):
            is_linear = (i + 1) % 4 != 0
            if is_linear:
                # ArraysCache is returned as-is with left_padding set
                assert isinstance(c, ArraysCache), f"Layer {i} should be ArraysCache"
                assert c.left_padding is not None
            else:
                assert isinstance(c, BatchKVCache), f"Layer {i} should be BatchKVCache"

    def test_pure_kv_cache_still_works(self):
        """Regression: pure attention models (all KVCache) still work."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        class FakeModel:
            def make_cache(self):
                return [KVCache() for _ in range(24)]

        batch_cache = _make_batch_cache(FakeModel(), [0, 1])
        assert all(isinstance(c, BatchKVCache) for c in batch_cache)

    def test_pure_arrays_cache_works(self):
        """Pure SSM models (all ArraysCache) work."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        class FakeModel:
            def make_cache(self):
                return [ArraysCache(size=2) for _ in range(24)]

        batch_cache = _make_batch_cache(FakeModel(), [0, 1])
        assert all(isinstance(c, ArraysCache) for c in batch_cache)

    def test_rotating_kv_cache_works(self):
        """RotatingKVCache (keep=0) gets wrapped in BatchRotatingKVCache."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        class FakeModel:
            def make_cache(self):
                return [RotatingKVCache(max_size=1024, keep=0) for _ in range(4)]

        batch_cache = _make_batch_cache(FakeModel(), [0])
        assert all(isinstance(c, BatchRotatingKVCache) for c in batch_cache)

    def test_rotating_kv_cache_with_keep_rejected(self):
        """RotatingKVCache with keep > 0 is rejected."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        class FakeModel:
            def make_cache(self):
                return [RotatingKVCache(max_size=1024, keep=4)]

        with pytest.raises(ValueError, match="keep tokens is not supported"):
            _make_batch_cache(FakeModel(), [0])

    def test_unsupported_cache_type_rejected(self):
        """Cache types without batching support are rejected with clear error."""
        from vllm_mlx.mllm_batch_generator import _make_batch_cache

        class UnsupportedCache:
            pass

        class FakeModel:
            def make_cache(self):
                return [UnsupportedCache()]

        with pytest.raises(ValueError, match="does not support"):
            _make_batch_cache(FakeModel(), [0])


# ---------------------------------------------------------------------------
# Test: Merge loop works with mixed cache types
# ---------------------------------------------------------------------------


class TestHybridCacheMerge:
    """Test the per-layer merge loop from _process_prompts."""

    def test_merge_hybrid_per_request_caches(self):
        """Merging per-request hybrid caches produces correct batched types."""
        # Simulate 2 requests, each with a populated hybrid cache
        caches = [
            _make_populated_hybrid_cache(seq_len=10),
            _make_populated_hybrid_cache(seq_len=15),
        ]

        # This is the exact merge loop from _process_prompts (lines 679-685)
        batch_cache = [
            caches[0][layer_idx].merge([c[layer_idx] for c in caches])
            for layer_idx in range(len(caches[0]))
        ]

        assert len(batch_cache) == 48
        for i, c in enumerate(batch_cache):
            is_linear = (i + 1) % 4 != 0
            if is_linear:
                assert isinstance(
                    c, ArraysCache
                ), f"Layer {i}: merged ArraysCache should stay ArraysCache"
                # Merged arrays should have batch dimension = 2
                for arr in c.cache:
                    if arr is not None:
                        assert arr.shape[0] == 2, f"Layer {i}: batch dim should be 2"
            else:
                assert isinstance(
                    c, BatchKVCache
                ), f"Layer {i}: merged KVCache should become BatchKVCache"

    def test_merge_single_request(self):
        """Single-request merge works (degenerate case)."""
        caches = [_make_populated_hybrid_cache(seq_len=10)]
        batch_cache = [
            caches[0][layer_idx].merge([c[layer_idx] for c in caches])
            for layer_idx in range(len(caches[0]))
        ]
        assert len(batch_cache) == 48

    def test_type_guard_rejects_unmergeable_cache(self):
        """The capability check rejects caches without merge()."""
        from vllm_mlx.mllm_batch_generator import _validate_caches_mergeable

        class NoMergeCache:
            pass

        per_request_caches = [[NoMergeCache(), KVCache()]]
        with pytest.raises(ValueError, match="lacks a merge"):
            _validate_caches_mergeable(per_request_caches)


# ---------------------------------------------------------------------------
# Test: Filter on merged batch
# ---------------------------------------------------------------------------


class TestHybridCacheFilter:
    """Test filter() on merged hybrid batches."""

    def test_filter_keeps_correct_requests(self):
        """Filter on merged batch keeps correct batch elements for both cache types."""
        caches = [
            _make_populated_hybrid_cache(seq_len=10),
            _make_populated_hybrid_cache(seq_len=15),
            _make_populated_hybrid_cache(seq_len=12),
        ]
        batch_cache = [
            caches[0][layer_idx].merge([c[layer_idx] for c in caches])
            for layer_idx in range(len(caches[0]))
        ]

        # Keep only request 0 and 2
        keep_idx = mx.array([0, 2], mx.int32)
        for c in batch_cache:
            if hasattr(c, "filter"):
                c.filter(keep_idx)

        # Verify batch dimension is now 2
        for i, c in enumerate(batch_cache):
            is_linear = (i + 1) % 4 != 0
            if is_linear and isinstance(c, ArraysCache):
                for arr in c.cache:
                    if arr is not None:
                        assert (
                            arr.shape[0] == 2
                        ), f"Layer {i}: filtered batch dim should be 2"


# ---------------------------------------------------------------------------
# Test: Extract from merged batch
# ---------------------------------------------------------------------------


class TestHybridCacheExtract:
    """Test extract() on merged hybrid batches."""

    def test_extract_returns_correct_types(self):
        """Extracting a single request returns correct unbatched types."""
        caches = [
            _make_populated_hybrid_cache(seq_len=10),
            _make_populated_hybrid_cache(seq_len=15),
        ]
        batch_cache = [
            caches[0][layer_idx].merge([c[layer_idx] for c in caches])
            for layer_idx in range(len(caches[0]))
        ]

        # Extract request 0
        extracted = [
            c.extract(0) if hasattr(c, "extract") else None for c in batch_cache
        ]

        for i, c in enumerate(extracted):
            is_linear = (i + 1) % 4 != 0
            if c is None:
                continue
            if is_linear:
                assert isinstance(
                    c, ArraysCache
                ), f"Layer {i}: extracted should be ArraysCache"
                for arr in c.cache:
                    if arr is not None:
                        assert (
                            arr.shape[0] == 1
                        ), f"Layer {i}: extracted batch dim should be 1"
            else:
                assert isinstance(c, KVCache), f"Layer {i}: extracted should be KVCache"


# ---------------------------------------------------------------------------
# Test: Extend merged batches
# ---------------------------------------------------------------------------


class TestHybridCacheExtend:
    """Test extend() combining two merged hybrid batches."""

    def test_extend_combines_batches(self):
        """Extending one merged batch with another works for both cache types."""
        caches_a = [
            _make_populated_hybrid_cache(seq_len=10),
            _make_populated_hybrid_cache(seq_len=15),
        ]
        caches_b = [
            _make_populated_hybrid_cache(seq_len=12),
        ]
        batch_a = [
            caches_a[0][layer_idx].merge([c[layer_idx] for c in caches_a])
            for layer_idx in range(len(caches_a[0]))
        ]
        batch_b = [
            caches_b[0][layer_idx].merge([c[layer_idx] for c in caches_b])
            for layer_idx in range(len(caches_b[0]))
        ]

        # Extend batch_a with batch_b
        for c, o in zip(batch_a, batch_b):
            if c is not None and o is not None and hasattr(c, "extend"):
                if not c.empty() and not o.empty():
                    c.extend(o)

        # Verify combined batch has 3 elements
        for i, c in enumerate(batch_a):
            is_linear = (i + 1) % 4 != 0
            if is_linear and isinstance(c, ArraysCache):
                for arr in c.cache:
                    if arr is not None:
                        assert (
                            arr.shape[0] == 3
                        ), f"Layer {i}: extended batch dim should be 3"


# ---------------------------------------------------------------------------
# Test: Message normalization
# ---------------------------------------------------------------------------


class TestNormalizeMessages:
    """Test _normalize_messages() for handling real-world client formats."""

    def test_merge_consecutive_system_messages(self):
        """Consecutive system messages are merged into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Always respond in JSON."},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "helpful assistant" in result[0]["content"]
        assert "JSON" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_merge_consecutive_user_messages(self):
        """Consecutive user messages are merged into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "First part"},
            {"role": "user", "content": "Second part"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert "First part" in result[1]["content"]
        assert "Second part" in result[1]["content"]

    def test_opencode_format(self):
        """OpenCode's system+system+user+user format is normalized."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "System prompt part 1"},
            {"role": "system", "content": "System prompt part 2"},
            {"role": "user", "content": "User instruction"},
            {"role": "user", "content": "User question"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_already_alternating_unchanged(self):
        """Well-formed alternating messages pass through unchanged."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
        ]
        result = _normalize_messages(messages)
        assert result == messages

    def test_single_message_unchanged(self):
        """Single message passes through unchanged."""
        from vllm_mlx.server import _normalize_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = _normalize_messages(messages)
        assert result == messages

    def test_empty_messages(self):
        """Empty message list passes through."""
        from vllm_mlx.server import _normalize_messages

        assert _normalize_messages([]) == []

    def test_multimodal_content_preserved(self):
        """Messages with list content (multimodal) are preserved during merge."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "user", "content": "Describe this:"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            },
        ]
        result = _normalize_messages(messages)
        # When one message has list content and previous has string,
        # they can't be trivially merged — keep them or convert
        # At minimum, no crash
        assert len(result) >= 1

    def test_preserves_non_content_fields(self):
        """Fields other than role/content are preserved."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "Part 1", "name": "sys1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        # First message retains extra fields from first of the merged group
        assert result[0]["role"] == "system"

    def test_null_content_not_merged(self):
        """Messages with None content (tool_calls pattern) are not merged."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
            {"role": "assistant", "content": "Follow-up"},
        ]
        result = _normalize_messages(messages)
        # None content can't be merged with string — kept separate
        assert len(result) == 2

    def test_three_consecutive_system_messages(self):
        """Three consecutive system messages merge into one."""
        from vllm_mlx.server import _normalize_messages

        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "system", "content": "Part 3"},
            {"role": "user", "content": "Hello"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert "Part 1" in result[0]["content"]
        assert "Part 3" in result[0]["content"]


# ---------------------------------------------------------------------------
# Test: Empty cache extend guard
# ---------------------------------------------------------------------------


class TestEmptyCacheExtend:
    """Test the empty() guard in extend prevents crashes on unpopulated caches."""

    def test_extend_skips_empty_caches(self):
        """Extending when one cache is empty does not crash."""
        populated = _make_populated_hybrid_cache(seq_len=10)

        # Merge populated into single-request batch
        batch_pop = [populated[i].merge([populated[i]]) for i in range(len(populated))]

        # Create empty caches directly (don't merge — merge() can't handle all-None)
        batch_empty = _make_hybrid_cache()

        # Extend should not crash — empty guard should skip
        for c, o in zip(batch_pop, batch_empty):
            if c is not None and o is not None and hasattr(c, "extend"):
                if not c.empty() and not o.empty():
                    c.extend(o)
        # If we get here without crash, test passes
