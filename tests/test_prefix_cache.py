# SPDX-License-Identifier: Apache-2.0
"""
Tests for prefix cache functionality.

These tests verify the PrefixCacheManager for KV cache reuse
to speed up inference with repeated prompts.
"""

import threading
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, get_args, get_type_hints
from unittest.mock import MagicMock

import pytest

from vllm_mlx.prefix_cache import (
    CacheEntry,
    PrefixCacheManager,
    PrefixCacheStats,
)


class TestPrefixCacheStats:
    """Tests for PrefixCacheStats class."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = PrefixCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0
        assert stats.total_queries == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = PrefixCacheStats(hits=3, misses=7, total_queries=10)
        assert stats.hit_rate == 0.3

    def test_hit_rate_zero_queries(self):
        """Test hit rate with zero queries."""
        stats = PrefixCacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PrefixCacheStats(hits=5, misses=5, tokens_saved=100, total_queries=10)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["hit_rate"] == 0.5
        assert d["tokens_saved"] == 100
        assert d["total_queries"] == 10


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        cache = ["mock_kv_cache"]
        entry = CacheEntry(prompt_cache=cache, count=1)
        assert entry.prompt_cache == ["mock_kv_cache"]
        assert entry.count == 1

    def test_cache_entry_count_increment(self):
        """Test incrementing reference count."""
        entry = CacheEntry(prompt_cache=["cache"], count=1)
        entry.count += 1
        assert entry.count == 2


class TestPrefixCacheManager:
    """Tests for PrefixCacheManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    @pytest.fixture
    def cache_manager(self, mock_model):
        """Create a cache manager with default settings."""
        return PrefixCacheManager(mock_model, max_entries=10)

    def test_initialization(self, mock_model):
        """Test cache manager initialization."""
        manager = PrefixCacheManager(mock_model, max_entries=50)
        assert manager.max_size == 50
        assert manager.model_key == id(mock_model)
        assert len(manager) == 0

    def test_fetch_empty_cache(self, cache_manager):
        """Test fetching from empty cache returns miss."""
        tokens = [1, 2, 3, 4, 5]
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is None
        assert remaining == tokens
        assert cache_manager.stats.misses == 1
        assert cache_manager.stats.hits == 0

    def test_store_and_fetch_exact_match(self, cache_manager):
        """Test storing and fetching exact match."""
        tokens = [1, 2, 3, 4, 5]
        mock_cache = ["kv_layer_1", "kv_layer_2"]

        # Store cache
        cache_manager.store_cache(tokens, mock_cache)
        assert len(cache_manager) == 1

        # Fetch exact match
        cache, remaining = cache_manager.fetch_cache(tokens)

        assert cache is not None
        assert remaining == []
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(tokens)

    def test_fetch_shorter_prefix(self, cache_manager):
        """Test fetching when a shorter prefix is cached."""
        # Store short prefix
        short_tokens = [1, 2, 3]
        mock_cache = ["short_cache"]
        cache_manager.store_cache(short_tokens, mock_cache)

        # Fetch longer sequence
        long_tokens = [1, 2, 3, 4, 5, 6]
        cache, remaining = cache_manager.fetch_cache(long_tokens)

        assert cache is not None
        assert remaining == [4, 5, 6]
        assert cache_manager.stats.hits == 1
        assert cache_manager.stats.tokens_saved == len(short_tokens)

    def test_lru_eviction(self, mock_model):
        """Test LRU eviction when cache is full."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])
        assert len(manager) == 3

        # Add one more - should evict oldest
        manager.store_cache([4], ["cache4"])
        assert len(manager) == 3
        assert manager.stats.evictions == 1

        # Token [1] should be evicted
        cache, _ = manager.fetch_cache([1])
        assert cache is None

    def test_lru_touch_on_access(self, mock_model):
        """Test that accessing a cache updates LRU order."""
        manager = PrefixCacheManager(mock_model, max_entries=3)

        # Fill cache
        manager.store_cache([1], ["cache1"])
        manager.store_cache([2], ["cache2"])
        manager.store_cache([3], ["cache3"])

        # Access [1] to make it most recently used
        manager.fetch_cache([1])

        # Add new entry - should evict [2] (oldest untouched)
        manager.store_cache([4], ["cache4"])

        # [1] should still be there
        cache, _ = manager.fetch_cache([1])
        assert cache is not None

        # [2] should be evicted
        cache, _ = manager.fetch_cache([2])
        assert cache is None

    def test_store_empty_tokens(self, cache_manager):
        """Test that empty tokens are not stored."""
        cache_manager.store_cache([], ["empty_cache"])
        assert len(cache_manager) == 0

    def test_get_stats(self, cache_manager):
        """Test getting statistics."""
        # Generate some activity
        cache_manager.store_cache([1, 2, 3], ["cache1"])
        cache_manager.fetch_cache([1, 2, 3])  # Hit
        cache_manager.fetch_cache([4, 5, 6])  # Miss

        stats = cache_manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["total_queries"] == 2

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics."""
        cache_manager.stats.hits = 10
        cache_manager.stats.misses = 5
        cache_manager.reset_stats()

        assert cache_manager.stats.hits == 0
        assert cache_manager.stats.misses == 0

    def test_clear(self, cache_manager):
        """Test clearing the cache."""
        cache_manager.store_cache([1, 2], ["cache1"])
        cache_manager.store_cache([3, 4], ["cache2"])
        assert len(cache_manager) == 2

        cache_manager.clear()
        assert len(cache_manager) == 0

        # Stats should also be reset
        assert cache_manager.stats.hits == 0

    def test_cache_no_copy(self, cache_manager):
        """Test that fetched cache is a reference (no copy) — MLX arrays are immutable."""
        original = [[1, 2, 3]]
        cache_manager.store_cache([1, 2], original)

        cache, _ = cache_manager.fetch_cache([1, 2])

        # Returns the same object (no deep copy overhead)
        assert cache is original

    def test_multiple_prefixes(self, cache_manager):
        """Test multiple different prefixes."""
        cache_manager.store_cache([1, 2], ["cache_a"])
        cache_manager.store_cache([3, 4], ["cache_b"])
        cache_manager.store_cache([1, 2, 3], ["cache_c"])

        # Fetch each
        cache_a, _ = cache_manager.fetch_cache([1, 2])
        cache_b, _ = cache_manager.fetch_cache([3, 4])
        cache_c, _ = cache_manager.fetch_cache([1, 2, 3])

        assert cache_a == ["cache_a"]
        assert cache_b == ["cache_b"]
        assert cache_c == ["cache_c"]

    def test_trie_structure(self, cache_manager):
        """Test trie correctly handles branching prefixes."""
        # Store two prefixes with common start
        cache_manager.store_cache([1, 2, 3], ["cache_123"])
        cache_manager.store_cache([1, 2, 4], ["cache_124"])

        # Fetch the common prefix should return shorter match
        cache, remaining = cache_manager.fetch_cache([1, 2])
        # No exact match for [1,2], but [1,2,3] is longer - behavior depends on implementation
        # In our implementation, we find shorter prefix if available, otherwise return miss

        # Fetch exact matches
        cache_123, _ = cache_manager.fetch_cache([1, 2, 3])
        cache_124, _ = cache_manager.fetch_cache([1, 2, 4])

        assert cache_123 == ["cache_123"]
        assert cache_124 == ["cache_124"]


class TestPrefixTrieInferenceContract:
    @pytest.mark.parametrize(
        "method, annotation",
        [
            ("_search", "List[int]"),
            ("_delete_cache", "List[Tuple[Dict, Optional[int]]]"),
        ],
    )
    def test_local_paths_describe_tokens_and_root_sentinel(self, method, annotation):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(getattr(PrefixCacheManager, method)))
        )
        declarations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "path"
        ]
        assert len(declarations) == 1, "trie path needs its actual local element type"
        assert ast.dump(declarations[0].annotation) == ast.dump(
            ast.parse(annotation, mode="eval").body
        )

    def test_cleanup_key_does_not_rebind_integer_navigation_token(self):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(PrefixCacheManager._delete_cache))
        )
        navigation = next(node for node in ast.walk(tree) if isinstance(node, ast.For))
        cleanup = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "path"
        )
        assert cleanup.targets[0].elts[1].id != navigation.target.id

    @pytest.mark.parametrize(
        "tokens, expected",
        [
            ([1, 2], ([1, 2], None, None, 0)),
            ([1, 2, 9], (None, [1, 2], None, 0)),
            ([1, 4], (None, None, [1, 4, 5], 2)),
            ([1, 9], (None, None, None, 0)),
            ([8], (None, None, None, 0)),
        ],
    )
    def test_search_results_are_independent_without_mutating_shared_branches(
        self, tokens, expected
    ):
        import copy

        manager = PrefixCacheManager(None)
        manager.store_cache([1, 2], ["prefix"])
        manager.store_cache([1, 2, 3], ["descendant"])
        manager.store_cache([1, 4, 5], ["sibling"])
        original_tokens = list(tokens)
        before = copy.deepcopy(manager._cache)
        root = manager._cache[manager.model_key]
        branch = root[1]
        lru = list(manager._lru)
        result = manager._search(tokens)
        assert result == expected
        assert tokens == original_tokens
        for path in result[:3]:
            if path is not None:
                assert path is not tokens
                path.append(99)
        assert manager._search(tokens) == expected
        assert manager._cache == before
        assert manager._cache[manager.model_key] is root
        assert root[1] is branch
        assert list(manager._lru) == lru
        assert manager.stats == PrefixCacheStats()

    def test_search_absent_and_empty_roots_and_empty_query(self):
        manager = PrefixCacheManager(None)
        assert manager._search([]) == (None, None, None, 0)
        assert manager._search([0]) == (None, None, None, 0)
        manager.store_cache([], ["not-stored"])
        assert manager._cache == {}
        manager._cache[manager.model_key] = {}
        assert manager._search([]) == (None, None, None, 0)
        manager.store_cache([0, -2], ["only-extension"])
        assert manager._search([]) == (None, None, [0, -2], 0)
        root = manager._cache[manager.model_key]
        root["cache"] = CacheEntry(["root"], 1)
        assert manager._search([]) == ([], None, None, 0)
        assert manager.fetch_cache([]) == (None, [])
        assert manager.stats.misses == manager.stats.total_queries == 1

    @pytest.mark.parametrize("token", [0, -7])
    def test_delete_prunes_only_empty_branches_and_retains_root_and_survivors(
        self, token
    ):
        manager = PrefixCacheManager(None)
        prefix, leaf, sibling, other = ["prefix"], ["leaf"], ["sibling"], ["other"]
        for tokens, payload in [
            ([token], prefix),
            ([token, 2], leaf),
            ([token, 3], sibling),
            ([9], other),
        ]:
            manager.store_cache(tokens, payload)
        root = manager._cache[manager.model_key]
        branch = root[token]
        surviving_entry = manager._get_cache_entry([token, 3])
        other_entry = manager._get_cache_entry([9])
        lru = list(manager._lru)
        manager._delete_cache(manager.model_key, [token, 2])
        assert 2 not in branch
        assert manager._get_cache_entry([token]).prompt_cache is prefix
        manager._delete_cache(manager.model_key, [token])
        assert "cache" not in branch
        assert root[token] is branch
        assert manager._get_cache_entry([token, 3]) is surviving_entry
        assert surviving_entry.prompt_cache is sibling
        assert manager._get_cache_entry([9]) is other_entry
        assert other_entry.prompt_cache is other
        manager._delete_cache(manager.model_key, [token, 3])
        assert token not in root
        assert manager._get_cache_entry([9]) is other_entry
        manager._delete_cache(manager.model_key, [9])
        assert manager._cache[manager.model_key] is root
        assert root == {}
        assert None not in root
        assert list(manager._lru) == lru
        assert manager.stats == PrefixCacheStats()

    def test_delete_missing_empty_paths_and_other_namespace_preserve_identity(self):
        manager = PrefixCacheManager(None)
        manager._delete_cache(manager.model_key, [])
        assert manager._cache == {}
        payload = ["survivor"]
        manager.store_cache([0, -1], payload)
        root = manager._cache[manager.model_key]
        branch = root[0]
        entry = manager._get_cache_entry([0, -1])
        other_key = object()
        foreign_root = {0: {"cache": CacheEntry(["foreign"], 1)}}
        manager._cache[other_key] = foreign_root
        for key, tokens in [
            (object(), [0]),
            (manager.model_key, [9]),
            (manager.model_key, [0, 9]),
            (manager.model_key, []),
        ]:
            manager._delete_cache(key, tokens)
            assert manager._cache[manager.model_key] is root
            assert root[0] is branch
            assert manager._get_cache_entry([0, -1]) is entry
            assert entry.prompt_cache is payload
            assert manager._cache[other_key] is foreign_root
        root["cache"] = CacheEntry(["root"], 1)
        manager._delete_cache(manager.model_key, [])
        assert "cache" not in root
        assert root[0] is branch
        manager._delete_cache(other_key, [0])
        assert manager._cache[other_key] is foreign_root
        assert foreign_root == {}
        assert manager._get_cache_entry([0, -1]) is entry

    def test_actual_lru_touch_eviction_duplicate_store_and_query_counters(self):
        manager = PrefixCacheManager(None, max_entries=2)
        first, second, third = ["first"], ["second"], ["third"]
        manager.store_cache([0], first)
        manager.store_cache([-1], second)
        entry = manager._get_cache_entry([0])
        fetched, remaining = manager.fetch_cache([0, 9])
        assert fetched is first
        assert remaining == [9]
        assert list(manager._lru) == [
            (manager.model_key, (-1,)),
            (manager.model_key, (0,)),
        ]
        manager.store_cache([2], third)
        assert manager._get_cache_entry([-1]) is None
        assert manager._get_cache_entry([0]) is entry
        assert manager.stats.evictions == 1
        manager.store_cache([0], ["replacement-not-used"])
        assert entry.count == 2
        assert entry.prompt_cache is first
        assert list(manager._lru) == [
            (manager.model_key, (2,)),
            (manager.model_key, (0,)),
        ]
        fetched, remaining = manager.fetch_cache([0])
        assert fetched is first and remaining == []
        missing = [-1]
        fetched, remaining = manager.fetch_cache(missing)
        assert fetched is None and remaining is missing
        assert manager.get_stats() == {
            "hits": 2,
            "misses": 1,
            "hit_rate": 2 / 3,
            "tokens_saved": 2,
            "total_queries": 3,
            "evictions": 1,
        }


class TestBlockLayerStorageContract:
    def test_block_annotation_matches_normalized_producer_schema(self):
        from vllm_mlx.paged_cache import CacheBlock
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        stored = get_type_hints(CacheBlock)["cache_data"]
        produced = get_type_hints(BlockAwarePrefixCache._extract_block_tensor_slice)[
            "return"
        ]
        assert stored == produced

        from vllm_mlx.paged_cache import (
            BlockCacheData,
            BlockLayerState,
            ConcatBlockLayerState,
            LatestBlockLayerState,
        )

        assert stored == Optional[BlockCacheData]
        assert BlockCacheData == List[Optional[BlockLayerState]]
        assert set(get_args(BlockLayerState)) == {
            ConcatBlockLayerState,
            LatestBlockLayerState,
        }
        metadata = {
            "state": Any,
            "meta_state": object,
            "class_ref": Optional[type],
            "class_name": Optional[str],
        }
        concat = {**metadata, "storage": Literal["concat"], "seq_axis": int}
        latest = {**metadata, "storage": Literal["latest"]}
        assert get_type_hints(ConcatBlockLayerState) == concat
        assert get_type_hints(LatestBlockLayerState) == latest
        assert ConcatBlockLayerState.__required_keys__ == frozenset(concat)
        assert LatestBlockLayerState.__required_keys__ == frozenset(latest)
        assert ConcatBlockLayerState.__optional_keys__ == frozenset()
        assert LatestBlockLayerState.__optional_keys__ == frozenset()
        assert CacheBlock(1).cache_data is None

    @pytest.mark.parametrize("legacy", [["legacy-reference"], [("keys", "values")]])
    def test_legacy_request_entries_keep_original_reference_and_cow(self, legacy):
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(None, manager)
        table = cache.store_cache("legacy", list(range(4)), legacy)
        assert table is not None
        source = manager.allocated_blocks[table.block_ids[0]]
        assert source.cache_data is None
        assert cache._request_tables["legacy"].cache_data is legacy
        returned, copied = cache.get_cache_for_generation("legacy")
        assert returned is legacy
        assert copied is False
        assert cache.reconstruct_cache(table) is None

        fork = cache.fork_cache("legacy", "fork")
        assert fork is not None
        assert cache._request_tables["fork"].cache_data is legacy
        returned, copied = cache.get_cache_for_generation("fork")
        assert copied is True
        assert returned == legacy
        assert returned is not legacy
        assert cache._request_tables["legacy"].cache_data is legacy
        assert source.cache_data is None
        assert manager.allocated_blocks[fork.block_ids[0]].cache_data is None

    def test_normalized_cow_shares_payload_and_clear_recreates_empty_blocks(self):
        from vllm_mlx.paged_cache import PagedCacheManager

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        source = manager.allocate_block()
        assert source is not None
        state, metadata = object(), {"opaque": object()}
        payload = [
            None,
            {
                "state": state,
                "meta_state": metadata,
                "class_ref": None,
                "class_name": None,
                "storage": "latest",
            },
        ]
        source.cache_data = payload
        table = manager.create_block_table("source")
        manager.add_block_to_table(table, source, 4)
        fork = manager.fork_block_table(table, "fork")
        blocks, copied = manager.get_blocks_for_generation(fork)
        assert copied is True
        assert blocks[0] is not source
        assert blocks[0].cache_data is payload
        assert source.cache_data is payload
        assert payload[0] is None
        assert payload[1]["state"] is state
        assert payload[1]["meta_state"] is metadata
        assert source.ref_count == blocks[0].ref_count == 1

        manager.clear()
        assert all(block.cache_data is None for block in manager.blocks)
        assert manager.request_tables == {}
        assert list(manager.allocated_blocks) == [manager.null_block.block_id]
        assert source.cache_data is payload

    @pytest.mark.parametrize("shape, seq_axis", [((1, 2, 8, 3), 2), ((2, 8, 3), 1)])
    @pytest.mark.parametrize("container", [tuple, list])
    def test_concat_envelope_slices_and_reconstructs_values(
        self, shape, seq_axis, container
    ):
        mx = pytest.importorskip("mlx.core")
        cache_types = pytest.importorskip("mlx_lm.models.cache")
        from vllm_mlx.paged_cache import BlockTable, PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(None, manager)
        keys = mx.arange(48).reshape(shape)
        values = mx.arange(1000, 1048).reshape(shape)
        state = container([keys, values])
        extracted = [
            {
                "state": state,
                "meta_state": "",
                "class_ref": cache_types.KVCache,
                "class_name": "KVCache",
            }
        ]
        table = cache.store_cache("tensor", list(range(8)), extracted)
        assert table is not None
        assert cache._request_tables["tensor"].cache_data is extracted
        for index, block_id in enumerate(table.block_ids):
            payload = manager.allocated_blocks[block_id].cache_data
            assert isinstance(payload, list) and len(payload) == 1
            layer = payload[0]
            assert set(layer) == {
                "state",
                "meta_state",
                "class_ref",
                "class_name",
                "storage",
                "seq_axis",
            }
            assert layer["storage"] == "concat"
            assert layer["seq_axis"] == seq_axis
            assert layer["class_ref"] is cache_types.KVCache
            assert layer["class_name"] == "KVCache"
            assert layer["meta_state"] == ""
            assert isinstance(layer["state"], container)
            start, end = index * 4, (index + 1) * 4
            for actual, original in zip(layer["state"], [keys, values]):
                expected = (
                    original[:, :, start:end, :]
                    if seq_axis == 2
                    else original[:, start:end, :]
                )
                assert actual.shape == expected.shape
                assert actual.tolist() == expected.tolist()

        reconstructed = cache.reconstruct_cache(table)
        assert reconstructed is not None
        assert reconstructed[0].keys.shape == shape
        assert reconstructed[0].keys.tolist() == keys.tolist()
        assert reconstructed[0].values.tolist() == values.tolist()
        prefix = BlockTable("prefix", block_ids=table.block_ids[:1], num_tokens=4)
        rebuilt_prefix = cache.reconstruct_cache(prefix)
        assert rebuilt_prefix is not None
        expected_keys = keys[:, :, :4, :] if seq_axis == 2 else keys[:, :4, :]
        expected_values = values[:, :, :4, :] if seq_axis == 2 else values[:, :4, :]
        assert rebuilt_prefix[0].keys.shape == expected_keys.shape
        assert rebuilt_prefix[0].keys.tolist() == expected_keys.tolist()
        assert rebuilt_prefix[0].values.tolist() == expected_values.tolist()
        if seq_axis == 2:
            assert reconstructed[0].offset == 8
            assert rebuilt_prefix[0].offset == 4

    def test_hybrid_latest_none_slots_and_missing_boundary_fail_closed(self):
        mx = pytest.importorskip("mlx.core")
        cache_types = pytest.importorskip("mlx_lm.models.cache")
        from vllm_mlx.paged_cache import BlockTable, PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(None, manager)
        keys = mx.arange(48).reshape(1, 2, 8, 3)
        recurrent = [mx.ones((1, 3, 8)), mx.full((1, 2, 4, 4), 7)]
        extracted = [
            {"state": (keys, keys), "class_ref": cache_types.KVCache, "meta_state": ""},
            {
                "state": recurrent,
                "class_ref": cache_types.ArraysCache,
                "meta_state": "",
            },
        ]
        table = cache.store_cache("hybrid", list(range(8)), extracted)
        assert table is not None
        first, last = [manager.allocated_blocks[bid] for bid in table.block_ids]
        assert len(first.cache_data) == len(last.cache_data) == 2
        assert first.cache_data[1] is None
        latest = last.cache_data[1]
        assert set(latest) == {
            "state",
            "meta_state",
            "class_ref",
            "class_name",
            "storage",
        }
        assert latest["storage"] == "latest"
        assert latest["state"] is recurrent
        assert latest["class_name"] is None
        prefix = BlockTable("prefix", block_ids=table.block_ids[:1], num_tokens=4)
        assert cache.reconstruct_cache(prefix) is None
        rebuilt = cache.reconstruct_cache(table)
        assert rebuilt is not None
        assert isinstance(rebuilt[1], cache_types.ArraysCache)
        assert rebuilt[1].state[0].tolist() == recurrent[0].tolist()
        assert rebuilt[1].state[1].tolist() == recurrent[1].tolist()
        assert cache._extract_block_tensor_slice([{}, {}], 0, 4, 8) is None
        hole = cache._extract_block_tensor_slice([extracted[0], {}], 0, 4, 8)
        assert hole is not None
        assert len(hole) == 2 and hole[1] is None

    def test_opaque_latest_state_and_metadata_keep_identity(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm.models.cache")
        from vllm_mlx.paged_cache import BlockTable, PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        class OpaqueCache:
            @classmethod
            def from_state(cls, state, meta):
                restored = cls()
                restored.state, restored.meta = state, meta
                return restored

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(None, manager)
        state, metadata = object(), {"opaque": object()}
        table = cache.store_cache(
            "opaque",
            list(range(8)),
            [{"state": state, "meta_state": metadata, "class_ref": OpaqueCache}],
        )
        assert table is not None
        first, last = [manager.allocated_blocks[bid] for bid in table.block_ids]
        assert first.cache_data is None
        assert last.cache_data[0]["state"] is state
        assert last.cache_data[0]["meta_state"] is metadata
        assert cache.reconstruct_cache(table) is None
        terminal = BlockTable("last", block_ids=table.block_ids[-1:], num_tokens=4)
        rebuilt = cache.reconstruct_cache(terminal)
        assert rebuilt is not None
        assert rebuilt[0].state is state
        assert rebuilt[0].meta is metadata

    def test_batch_state_envelope_cow_and_clear_keep_payload_identity(self):
        mx = pytest.importorskip("mlx.core")
        cache_types = pytest.importorskip("mlx_lm.models.cache")
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        cache = BlockAwarePrefixCache(None, manager)
        keys = mx.arange(24).reshape(1, 2, 4, 3)
        batch = cache_types.BatchKVCache(left_padding=[0])
        batch.update_and_fetch(keys, keys)
        state, metadata = batch.state, batch.meta_state
        table = cache.store_cache(
            "batch",
            list(range(4)),
            [
                {
                    "state": state,
                    "meta_state": metadata,
                    "class_ref": cache_types.BatchKVCache,
                }
            ],
        )
        assert table is not None
        source = manager.allocated_blocks[table.block_ids[0]]
        assert source.cache_data[0]["storage"] == "latest"
        assert source.cache_data[0]["state"] is state
        assert source.cache_data[0]["meta_state"] is metadata
        assert "seq_axis" not in source.cache_data[0]
        fork = cache.fork_cache("batch", "fork")
        assert fork is not None
        blocks, copied = manager.get_blocks_for_generation(fork)
        assert copied is True
        assert blocks[0].cache_data is source.cache_data
        cache.clear()
        assert all(block.cache_data is None for block in manager.blocks)
        assert len(cache) == 0


class TestSchedulerIntegration:
    """Test integration with scheduler."""

    def test_request_cache_fields(self):
        """Test that Request has cache fields."""
        from vllm_mlx.request import Request, SamplingParams

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        # Check cache fields exist
        assert hasattr(request, "prompt_cache")
        assert hasattr(request, "cached_tokens")
        assert hasattr(request, "remaining_tokens")

        # Check defaults
        assert request.prompt_cache is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens is None

    def test_scheduler_config_cache_options(self):
        """Test scheduler config has cache options."""
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(
            enable_prefix_cache=True,
            prefix_cache_size=200,
        )

        assert config.enable_prefix_cache is True
        assert config.prefix_cache_size == 200

        # Test defaults
        default_config = SchedulerConfig()
        assert default_config.enable_prefix_cache is True
        assert default_config.prefix_cache_size == 100


class TestTrimRotatingCaches:
    """Regression tests for restored rotating-cache validation."""

    def test_saturated_offset_is_preserved_while_buffer_is_trimmed(self):
        """The absolute position must survive window normalization."""
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        # A large first prefill can leave an oversized buffer. The next
        # single-token update normally reduces it to the rotating window.
        rc = RotatingKVCache(max_size=128, keep=0)
        rc.offset = 500
        rc.keys = mx.zeros((1, 4, 500, 64))
        rc.values = mx.zeros((1, 4, 500, 64))
        rc._idx = 500

        # Also include a regular KVCache (should be unaffected)
        kv = KVCache()
        kv.offset = 200
        kv.keys = mx.zeros((1, 4, 200, 64))
        kv.values = mx.zeros((1, 4, 200, 64))

        cache_list = [rc, kv]
        assert MLLMBatchGenerator._prepare_rotating_caches(cache_list) is True

        assert rc.offset == 500
        assert rc.keys.shape[2] == 128
        assert rc.values.shape[2] == 128
        assert rc._idx == 128

        # KVCache should be untouched
        assert kv.offset == 200

    def test_inconsistent_undersized_buffer_is_rejected(self):
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        rc = RotatingKVCache(max_size=128, keep=0)
        rc.offset = 500
        rc.keys = mx.zeros((1, 1, 64, 4))
        rc.values = mx.zeros((1, 1, 64, 4))
        rc._idx = 64

        assert MLLMBatchGenerator._prepare_rotating_caches([rc]) is False
        assert rc.offset == 500


class TestCopyPrefixCache:
    """Tests for _copy_prefix_cache preventing mutation of stored entries."""

    def test_copy_prevents_offset_mutation(self):
        """Modifying copied cache offset should not affect original."""
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        KVCache = mlx_lm_cache.KVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        original = KVCache()
        original.offset = 100

        copies = MLLMBatchGenerator._copy_prefix_cache([original])
        copies[0].offset = 999

        assert original.offset == 100

    def test_copy_rotating_prevents_idx_mutation(self):
        """Modifying copied RotatingKVCache _idx should not affect original."""
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        original = RotatingKVCache(max_size=128, keep=0)
        original.offset = 50
        original._idx = 50
        original.keys = mx.zeros((1, 4, 50, 64))
        original.values = mx.zeros((1, 4, 50, 64))

        copies = MLLMBatchGenerator._copy_prefix_cache([original])
        copies[0].offset = 999
        copies[0]._idx = 999

        assert original.offset == 50
        assert original._idx == 50

    def test_copy_recurses_through_cache_list(self):
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        kv = KVCache()
        kv.update_and_fetch(mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
        rotating = RotatingKVCache(max_size=4)
        rotating.update_and_fetch(mx.zeros((1, 1, 3, 4)), mx.zeros((1, 1, 3, 4)))
        original = CacheList(kv, rotating)

        copied = MLLMBatchGenerator._copy_prefix_cache([original])[0]

        assert copied is not original
        assert all(a is not b for a, b in zip(copied.caches, original.caches))
        assert copied.caches[0].keys is original.caches[0].keys
        copied.caches[0].offset = 1
        copied.caches[1]._idx = 1
        assert original.caches[0].offset == 3
        assert original.caches[1]._idx == 3

    def test_copy_recurses_through_mlx_vlm_cache_list(self):
        mx = pytest.importorskip("mlx.core")
        mlx_vlm_cache = pytest.importorskip("mlx_vlm.models.cache")

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        child = mlx_vlm_cache.KVCache()
        child.update_and_fetch(mx.zeros((1, 1, 3, 2)), mx.zeros((1, 1, 3, 2)))
        original = mlx_vlm_cache.CacheList(child)

        copied = MLLMBatchGenerator._copy_prefix_cache([original])[0]

        assert type(copied) is mlx_vlm_cache.CacheList
        assert copied is not original
        assert copied.caches[0] is not child
        copied.caches[0].offset = 1
        assert child.offset == 3


class TestRewindPrefixCache:
    def test_nested_plain_cache_rewinds_without_mutating_storage(self):
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        children = []
        for _ in range(2):
            cache = KVCache()
            cache.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
            children.append(cache)
        stored = CacheList(*children)

        rewound = MLLMBatchGenerator._rewind_prefix_cache([stored], 1)[0]

        assert [c.offset for c in rewound.caches] == [3, 3]
        assert [c.offset for c in stored.caches] == [4, 4]
        assert rewound is not stored

    def test_saturated_rotating_child_fails_closed(self):
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        kv = KVCache()
        kv.update_and_fetch(mx.zeros((1, 1, 6, 2)), mx.zeros((1, 1, 6, 2)))
        rotating = RotatingKVCache(max_size=4)
        rotating.keys = mx.zeros((1, 1, 4, 2))
        rotating.values = mx.zeros((1, 1, 4, 2))
        rotating.offset = 6
        rotating._idx = 4

        assert (
            MLLMBatchGenerator._rewind_prefix_cache([CacheList(kv, rotating)], 1)
            is None
        )

    @pytest.mark.parametrize(
        "cache_module", ["mlx_lm.models.cache", "mlx_vlm.models.cache"]
    )
    def test_chunked_cache_rewind_preserves_type_metadata(self, cache_module):
        mx = pytest.importorskip("mlx.core")
        cache_types = pytest.importorskip(cache_module)
        ChunkedKVCache = cache_types.ChunkedKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        stored = ChunkedKVCache(chunk_size=8)
        stored.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))

        rewound = MLLMBatchGenerator._rewind_prefix_cache([stored], 1)[0]

        assert type(rewound) is ChunkedKVCache
        assert rewound.chunk_size == 8
        assert rewound.start_position == 0
        assert rewound.offset == 3
        assert stored.offset == 4
        rewound.update_and_fetch(mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2)))
        assert rewound.offset == 4

    def test_chunked_cache_rewind_fails_when_front_was_discarded(self):
        mx = pytest.importorskip("mlx.core")
        mlx_vlm_cache = pytest.importorskip("mlx_vlm.models.cache")

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        stored = mlx_vlm_cache.ChunkedKVCache(chunk_size=4)
        stored.update_and_fetch(mx.zeros((1, 1, 4, 2)), mx.zeros((1, 1, 4, 2)))
        stored.start_position = 3
        stored.offset = 4

        assert MLLMBatchGenerator._rewind_prefix_cache([stored], 2) is None


class TestHasEmptyRotatingCache:
    """Tests for _has_empty_rotating_cache detection."""

    def test_detects_empty_rotating_cache(self):
        """Should return True when any RotatingKVCache has keys=None."""
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        kv = KVCache()
        rc = RotatingKVCache(max_size=128, keep=0)
        rc.keys = None

        assert MLLMBatchGenerator._has_empty_rotating_cache([kv, rc]) is True

    def test_returns_false_when_all_populated(self):
        """Should return False when all caches have data."""
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        kv = KVCache()
        rc = RotatingKVCache(max_size=128, keep=0)
        rc.keys = mx.zeros((1, 4, 50, 64))

        assert MLLMBatchGenerator._has_empty_rotating_cache([kv, rc]) is False

    def test_detects_empty_rotating_child_in_nested_cache_list(self):
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        nested = CacheList(KVCache(), CacheList(RotatingKVCache(max_size=4)))
        assert MLLMBatchGenerator._has_empty_rotating_cache([nested]) is True

    def test_detects_empty_mlx_vlm_rotating_child(self):
        mlx_vlm_cache = pytest.importorskip("mlx_vlm.models.cache")

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        nested = mlx_vlm_cache.CacheList(
            mlx_vlm_cache.KVCache(), mlx_vlm_cache.RotatingKVCache(max_size=4)
        )
        assert MLLMBatchGenerator._has_empty_rotating_cache([nested]) is True


class TestMLLMCompletionCacheStore:
    @staticmethod
    def _filled(cache, length, mx):
        cache.update_and_fetch(mx.zeros((1, 1, length, 2)), mx.zeros((1, 1, length, 2)))
        return cache

    def test_saturated_nested_rotating_cache_is_not_stored(self):
        from types import SimpleNamespace

        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache
        RotatingKVCache = mlx_lm_cache.RotatingKVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        full = self._filled(KVCache(), 8, mx)
        rotating = RotatingKVCache(max_size=4)
        rotating.keys = mx.zeros((1, 1, 4, 2))
        rotating.values = mx.zeros((1, 1, 4, 2))
        rotating.offset = 8
        rotating._idx = 4

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()
        generator._think_suffix_len = 0
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        request = SimpleNamespace(
            request_id="saturated",
            input_ids=mx.array([[1, 2, 3, 4]]),
            is_text_only=True,
        )
        batch = SimpleNamespace(
            requests=[request],
            num_tokens=[4],
            extract_cache=lambda _: [CacheList(full, rotating)],
        )

        generator._maybe_store_prefix_cache(batch, [0])

        generator.prefix_cache.store.assert_not_called()

    def test_saturated_flat_rotating_cache_is_not_stored(self):
        """Exercise #689's completion rewind without the container guard."""
        from types import SimpleNamespace

        mx = pytest.importorskip("mlx.core")
        mlx_vlm_cache = pytest.importorskip("mlx_vlm.models.cache")

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        full = self._filled(mlx_vlm_cache.KVCache(), 8, mx)
        rotating = mlx_vlm_cache.RotatingKVCache(max_size=4)
        rotating.keys = mx.zeros((1, 1, 4, 2))
        rotating.values = mx.zeros((1, 1, 4, 2))
        rotating.offset = 8
        rotating._idx = 4
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()
        generator._think_suffix_len = 0
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        request = SimpleNamespace(
            request_id="flat",
            input_ids=mx.array([[1, 2, 3, 4]]),
            is_text_only=True,
        )
        batch = SimpleNamespace(
            requests=[request],
            num_tokens=[4],
            extract_cache=lambda _: [full, rotating],
        )

        generator._maybe_store_prefix_cache(batch, [0])

        generator.prefix_cache.store.assert_not_called()

    def test_plain_nested_cache_is_stored_with_recursive_accounting(self):
        from types import SimpleNamespace

        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        CacheList = mlx_lm_cache.CacheList
        KVCache = mlx_lm_cache.KVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator
        from vllm_mlx.memory_cache import estimate_kv_cache_memory

        children = [self._filled(KVCache(), 4, mx) for _ in range(2)]
        extracted = CacheList(*children)
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()
        generator._think_suffix_len = 0
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        request = SimpleNamespace(
            request_id="plain",
            input_ids=mx.array([[1, 2, 3]]),
            is_text_only=True,
        )
        batch = SimpleNamespace(
            requests=[request],
            num_tokens=[1],
            extract_cache=lambda _: [extracted],
        )

        generator._maybe_store_prefix_cache(batch, [0])

        generator.prefix_cache.store.assert_called_once()
        _, stored = generator.prefix_cache.store.call_args.args
        assert estimate_kv_cache_memory(stored) > 0
        assert [child.offset for child in stored[0].caches] == [3, 3]

    def test_media_cache_is_never_stored_under_dense_prompt_tokens(self):
        from vllm_mlx.memory_cache import is_text_only_prefix_cache_request

        request = SimpleNamespace(
            request_id="media",
            input_ids=object(),
            is_text_only=False,
        )

        assert is_text_only_prefix_cache_request(request) is False

    def test_zero_trim_flat_snapshot_does_not_alias_live_cache(self):
        mx = pytest.importorskip("mlx.core")
        mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
        KVCache = mlx_lm_cache.KVCache

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        live = self._filled(KVCache(), 3, mx)
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()

        assert generator._store_prefix_snapshot(
            [1, 2, 3], [live], 0, "interleaved", "interleaved prefill"
        )

        _, stored = generator.prefix_cache.store.call_args.args
        assert stored[0] is not live
        live.offset = 99
        assert stored[0].offset == 3


class TestMLLMHybridPrefillCheckpoint:
    @pytest.fixture(autouse=True)
    def _require_full_mlx_runtime(self):
        pytest.importorskip("mlx.nn")

    class _HybridCache:
        def __init__(self, position=0):
            self.position = position

        @property
        def state(self):
            return [self.position]

        @state.setter
        def state(self, value):
            self.position = value[0]

        @property
        def meta_state(self):
            return []

        @classmethod
        def from_state(cls, state, _meta_state):
            return cls(state[0])

        def is_trimmable(self):
            return False

    class _KVCache(_HybridCache):
        def is_trimmable(self):
            return True

    @staticmethod
    def _request(input_ids):
        return SimpleNamespace(
            request_id="hybrid-prefill",
            input_ids=input_ids,
            vision_encoded=False,
            pixel_values=None,
            attention_mask=None,
            image_grid_thw=None,
            extra_kwargs={},
        )

    def test_nonrewindable_prompt_defers_store_until_full_prompt(self, monkeypatch):
        import numpy as np

        import vllm_mlx.mllm_batch_generator as module
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MemoryAwarePrefixCache(
            MagicMock(),
            MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1),
        )
        generator.prefill_step_size = 32
        # The prompt does not end with the think suffix (thinking disabled),
        # so no conversation-boundary checkpoint is planned; see
        # tests/test_mllm_boundary_checkpoint.py for the suffix case.
        generator._think_suffix_len = 2
        generator._think_suffixes = ((98, 99),)
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        cache = [self._HybridCache()]
        calls = []

        def language_model(tokens, cache):
            calls.append(tokens.tolist())
            cache[0].position += tokens.shape[1]
            return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

        generator.language_model = language_model
        monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)
        monkeypatch.setattr(module.mx, "eval", lambda *_args: None)

        request = self._request(np.array([[10, 11, 12, 13, 14]]))
        generator._run_chunked_text_prefill(request, cache)

        assert calls == [[[10, 11, 12, 13, 14]]]
        assert generator.prefix_cache.get_stats()["entry_count"] == 0
        assert cache[0].position == 5

    def test_rewindable_prompt_keeps_single_forward(self, monkeypatch):
        import numpy as np

        import vllm_mlx.mllm_batch_generator as module
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()
        generator.prefill_step_size = 32
        generator._think_suffix_len = 2
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        cache = [self._KVCache()]
        calls = []

        def language_model(tokens, cache):
            calls.append(tokens.tolist())
            cache[0].position += tokens.shape[1]
            return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

        generator.language_model = language_model
        monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)

        request = self._request(np.array([[10, 11, 12, 13, 14]]))
        generator._run_chunked_text_prefill(request, cache)

        assert calls == [[[10, 11, 12, 13, 14]]]
        generator.prefix_cache.prepare_store.assert_not_called()
        generator.prefix_cache.commit_prepared.assert_not_called()

    def test_no_think_hybrid_defers_store_until_final_logits_exist(self, monkeypatch):
        import numpy as np

        import vllm_mlx.mllm_batch_generator as module
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MemoryAwarePrefixCache(
            MagicMock(),
            MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1),
        )
        generator.prefill_step_size = 32
        generator._think_suffix_len = 0
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        cache = [self._HybridCache()]

        def language_model(tokens, cache):
            cache[0].position += tokens.shape[1]
            return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

        generator.language_model = language_model
        monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)
        monkeypatch.setattr(module.mx, "eval", lambda *_args: None)

        request = self._request(np.array([[10, 11, 12, 13, 14]]))
        generator._run_chunked_text_prefill(request, cache)

        stored, remaining = generator.prefix_cache.fetch([10, 11, 12, 13, 14])
        assert stored is None
        assert remaining == [10, 11, 12, 13, 14]

    def test_full_prompt_store_does_not_replace_existing_prefix_during_prefill(
        self, monkeypatch
    ):
        import numpy as np

        import vllm_mlx.mllm_batch_generator as module
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        prefix_cache = MemoryAwarePrefixCache(
            MagicMock(),
            MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1),
        )
        assert prefix_cache.store([10, 11], [self._HybridCache(2)])

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = prefix_cache
        generator.prefill_step_size = 32
        # The prompt does not end with the think suffix (thinking disabled),
        # so no conversation-boundary checkpoint is planned; see
        # tests/test_mllm_boundary_checkpoint.py for the suffix case.
        generator._think_suffix_len = 2
        generator._think_suffixes = ((98, 99),)
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        cache = [self._HybridCache()]

        def language_model(tokens, cache):
            cache[0].position += tokens.shape[1]
            return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

        generator.language_model = language_model
        monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)
        monkeypatch.setattr(module.mx, "eval", lambda *_args: None)
        request = self._request(np.array([[10, 11, 12, 13, 14]]))
        generator._run_chunked_text_prefill(request, cache)

        assert (10, 11) in prefix_cache._entries
        assert (10, 11, 12) not in prefix_cache._entries

        original_entry = prefix_cache._entries[(10, 11)]
        second = self._request(np.array([[10, 11, 12, 13, 14]]))
        second.request_id = "duplicate-checkpoint"
        generator._run_chunked_text_prefill(second, [self._HybridCache()])
        generator.abort_prefill(second.request_id)

        assert prefix_cache._entries[(10, 11)] is original_entry

    def test_abort_after_checkpoint_does_not_publish_entry(self, monkeypatch):
        import numpy as np

        import vllm_mlx.mllm_batch_generator as module
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            PrefillAbortedError,
        )

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = MagicMock()
        generator.prefill_step_size = 32
        # The prompt does not end with the think suffix (thinking disabled),
        # so no conversation-boundary checkpoint is planned; see
        # tests/test_mllm_boundary_checkpoint.py for the suffix case.
        generator._think_suffix_len = 2
        generator._think_suffixes = ((98, 99),)
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        cache = [self._HybridCache()]

        def language_model(tokens, cache):
            cache[0].position += tokens.shape[1]
            generator._aborted_request_ids.add("hybrid-prefill")
            return np.zeros((1, tokens.shape[1], 4), dtype=np.float32)

        generator.language_model = language_model
        monkeypatch.setattr(module, "_eval_prompt_cache", lambda _cache: None)

        request = self._request(np.array([[10, 11, 12, 13, 14]]))
        with pytest.raises(PrefillAbortedError):
            generator._run_chunked_text_prefill(request, cache)

        generator.prefix_cache.prepare_store.assert_not_called()
        generator.prefix_cache.commit_prepared.assert_not_called()

    def test_abort_during_checkpoint_commit_rejects_publication(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        store_started = threading.Event()
        release_store = threading.Event()
        prefix_cache = MagicMock()
        prefix_cache._config = SimpleNamespace(min_prefix_tokens=1)

        prepared_entry = SimpleNamespace(tokens=(10, 11, 12))
        prefix_cache.prepare_store.return_value = prepared_entry

        def commit_prepared(_entry, **kwargs):
            store_started.set()
            assert release_store.wait(timeout=5)
            with kwargs["commit_lock"]:
                return kwargs["commit_guard"]()

        prefix_cache.commit_prepared.side_effect = commit_prepared
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator.prefix_cache = prefix_cache
        generator.prefill_step_size = 32
        generator._think_suffix_len = 2
        generator._aborted_request_ids = set()
        generator._prefill_progress = {}
        generator._prefix_checkpoint_lock = threading.Lock()
        generator._request_prefix_checkpoints = {}
        request_id = "hybrid-prefill"

        prefill_errors = []

        def run_prefill():
            try:
                generator._publish_prefill_checkpoint(request_id, prepared_entry)
            except Exception as exc:
                prefill_errors.append(exc)

        prefill = threading.Thread(target=run_prefill)
        prefill.start()
        assert store_started.wait(timeout=5)
        abort = threading.Thread(
            target=generator.abort_prefill,
            args=(request_id,),
        )
        abort.start()
        release_store.set()
        prefill.join(timeout=5)
        abort.join(timeout=5)

        assert not prefill.is_alive()
        assert not abort.is_alive()
        assert len(prefill_errors) == 1
        assert type(prefill_errors[0]).__name__ == "PrefillAbortedError"
        assert prefix_cache.commit_prepared.call_count == 1
        assert request_id not in generator._request_prefix_checkpoints


if __name__ == "__main__":
    # Verbose standalone test with real model
    import argparse
    import asyncio
    import os
    import time

    parser = argparse.ArgumentParser(description="LLM prefix cache benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLLM_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-8bit"),
        help="Model to benchmark",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    def print_header(title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_subheader(title):
        print("\n" + "-" * 70)
        print(f"  {title}")
        print("-" * 70)

    def print_table(headers, rows):
        """Print a formatted table."""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)
        print(f"    {header_line}")
        print(f"    {separator}")

        # Print rows
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            print(f"    {row_line}")

    def print_stats_table(stats, title="Cache Statistics"):
        """Print cache stats as a table."""
        print(f"\n    {title}:")
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        total_queries = stats.get("total_queries", hits + misses)
        hit_rate = stats.get("hit_rate")
        if hit_rate is None:
            hit_rate = (hits / total_queries) if total_queries > 0 else 0.0

        headers = ["Metric", "Value"]
        rows = [
            ["Hits", hits],
            ["Misses", misses],
            ["Hit Rate", f"{hit_rate*100:.1f}%"],
            ["Tokens Saved", stats.get("tokens_saved", 0)],
            ["Total Queries", total_queries],
        ]
        print_table(headers, rows)

    async def run_cache_test():
        from mlx_lm import load

        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        print_header("LLM PREFIX CACHE TEST")
        print(f"\n  Model: {MODEL_NAME}")
        print("  Test: Verify KV cache reuse for repeated prompts")
        print("  Expected behavior:")
        print("    - Same prompt → cache HIT (skip prompt processing)")
        print(
            "    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)"
        )

        print_subheader("Loading Model")
        load_start = time.perf_counter()
        model, tokenizer = load(MODEL_NAME)
        load_time = time.perf_counter() - load_start
        print(f"    Model loaded in {load_time:.2f}s")

        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                enable_prefix_cache=True,
                prefix_cache_size=100,
            ),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            # Test prompts
            prompt1 = "What is 2+2?"
            prompt2 = "What is the capital of France?"

            formatted1 = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt1}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted2 = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt2}],
                tokenize=False,
                add_generation_prompt=True,
            )

            tokens1 = len(tokenizer.encode(formatted1))
            tokens2 = len(tokenizer.encode(formatted2))

            params = SamplingParams(max_tokens=20, temperature=0.0)

            # Collect test results for final table
            test_results = []

            # ============================================================
            # TEST 1: First request - should be cache MISS
            # ============================================================
            print_subheader("TEST 1: First Request (Cache Miss Expected)")
            print(f'    Prompt: "{prompt1}"')
            print(f"    Tokens: {tokens1}")

            start = time.perf_counter()
            rid1 = await engine.add_request(formatted1, params)
            response1 = ""
            async for out in engine.stream_outputs(rid1, timeout=30):
                if out.output_text:
                    response1 = out.output_text
                if out.finished:
                    break
            t1 = time.perf_counter() - start

            stats1 = engine.get_cache_stats()
            test1_pass = stats1["misses"] == 1 and stats1["hits"] == 0
            test_results.append(
                [
                    "TEST 1",
                    "First request",
                    "MISS",
                    "MISS" if stats1["hits"] == 0 else "HIT",
                    f"{t1*1000:.1f}ms",
                    "PASS" if test1_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response1.strip()[:50]}..."')
            print_stats_table(stats1)

            # ============================================================
            # TEST 2: Same prompt again - should be cache HIT
            # ============================================================
            print_subheader("TEST 2: Same Prompt Again (Cache Hit Expected)")
            print(f'    Prompt: "{prompt1}" (same as TEST 1)')
            print(f"    Tokens: {tokens1}")

            start = time.perf_counter()
            rid2 = await engine.add_request(formatted1, params)
            response2 = ""
            async for out in engine.stream_outputs(rid2, timeout=30):
                if out.output_text:
                    response2 = out.output_text
                if out.finished:
                    break
            t2 = time.perf_counter() - start

            stats2 = engine.get_cache_stats()
            test2_pass = stats2["hits"] == 1
            test_results.append(
                [
                    "TEST 2",
                    "Same prompt (cached)",
                    "HIT",
                    "HIT" if stats2["hits"] > stats1["hits"] else "MISS",
                    f"{t2*1000:.1f}ms",
                    "PASS" if test2_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response2.strip()[:50]}..."')
            speedup = t1 / t2 if t2 > 0 else 0
            print(f"    Speedup: {speedup:.2f}x faster")
            print_stats_table(stats2)

            # ============================================================
            # TEST 3: Different prompt - should be cache MISS or PREFIX_HIT
            # ============================================================
            print_subheader(
                "TEST 3: Different Prompt (Cache Miss or Prefix Hit Expected)"
            )
            print(f'    Prompt: "{prompt2}" (different from TEST 1)')
            print(f"    Tokens: {tokens2}")

            start = time.perf_counter()
            rid3 = await engine.add_request(formatted2, params)
            response3 = ""
            async for out in engine.stream_outputs(rid3, timeout=30):
                if out.output_text:
                    response3 = out.output_text
                if out.finished:
                    break
            t3 = time.perf_counter() - start

            stats3 = engine.get_cache_stats()
            hits_delta = stats3["hits"] - stats2["hits"]
            misses_delta = stats3["misses"] - stats2["misses"]
            tokens_saved_delta = stats3.get("tokens_saved", 0) - stats2.get(
                "tokens_saved", 0
            )

            # Different prompts may still share template/system prefix tokens.
            # Treat either a true miss OR any prefix-hit reuse as valid behavior.
            actual3 = "HIT"
            if misses_delta > 0:
                actual3 = "MISS"
                test3_pass = True
            elif hits_delta > 0:
                actual3 = (
                    f"PREFIX_HIT({tokens_saved_delta} tok)"
                    if tokens_saved_delta > 0
                    else "PREFIX_HIT"
                )
                test3_pass = True
            else:
                test3_pass = False

            test_results.append(
                [
                    "TEST 3",
                    "Different prompt",
                    "MISS or PREFIX_HIT",
                    actual3,
                    f"{t3*1000:.1f}ms",
                    "PASS" if test3_pass else "FAIL",
                ]
            )

            print(f'    Response: "{response3.strip()[:50]}..."')
            print_stats_table(stats3)

            # ============================================================
            # SUMMARY TABLE
            # ============================================================
            print_header("TEST RESULTS SUMMARY")

            # Test results table
            print("\n    Test Results:")
            print_table(
                ["Test", "Description", "Expected", "Actual", "Time", "Status"],
                test_results,
            )

            # Final stats table
            final_stats = engine.get_cache_stats()
            print("\n    Final Cache Statistics:")
            print_table(
                ["Metric", "Value"],
                [
                    ["Total Requests", 3],
                    ["Cache Hits", final_stats["hits"]],
                    ["Cache Misses", final_stats["misses"]],
                    ["Hit Rate", f"{final_stats['hit_rate']*100:.1f}%"],
                    ["Tokens Saved", final_stats["tokens_saved"]],
                    ["Speedup (cached)", f"{speedup:.2f}x"],
                ],
            )

            all_passed = test1_pass and test2_pass and test3_pass

            print("\n" + "=" * 70)
            if all_passed:
                print("  [OK] ALL TESTS PASSED - Prefix cache working correctly")
            else:
                print("  [FAILED] SOME TESTS FAILED - Check results above")
            print("=" * 70)

    asyncio.run(run_cache_test())
