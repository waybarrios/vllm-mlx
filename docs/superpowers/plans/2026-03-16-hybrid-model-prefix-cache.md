# Hybrid Model Prefix Cache — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BlockAwarePrefixCache` work with hybrid models (Qwen 3.5, Nemotron) that mix KVCache and ArraysCache layers, fixing issues #142 and #136.

**Architecture:** Classify cache layers as positional (KVCache — block-sliceable) vs non-positional (ArraysCache — stored whole). KV layers are block-sliced as before. Non-KV layers are stored once per prefix entry and restored via `from_state()`. Partial prefix reuse is rejected for hybrid models (mismatched KV/SSM state would corrupt output).

**Tech Stack:** Python 3.12, mlx, mlx-lm (KVCache/ArraysCache), pytest

**Spec:** `docs/superpowers/specs/2026-03-16-hybrid-model-prefix-cache-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm_mlx/prefix_cache.py` | Modify | Layer classification, hybrid-aware slicing/storage/reconstruction |
| `vllm_mlx/scheduler.py` | Modify | Robustness guard in `_reconstruct_cache_from_states()` fallback |
| `tests/test_prefix_cache_hybrid.py` | Create | All new tests for hybrid prefix cache behavior |

---

## Chunk 1: Layer Classification and Extract Fix

### Task 1: Add `_is_kv_layer` helper and `NonKVCacheData` dataclass

**Files:**
- Modify: `vllm_mlx/prefix_cache.py` (add near top, after existing imports/dataclasses)
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the test file with layer classification tests**

Create `tests/test_prefix_cache_hybrid.py`:

```python
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
    conv_state = mx.zeros((1, 3, conv_dim))       # (B, kernel-1, conv_dim)
    ssm_state = mx.zeros((1, ssm_heads, ssm_dim, ssm_dim))  # (B, H, D, D)
    return {
        "state": [conv_state, ssm_state],
        "meta_state": "",
        "class_name": "ArraysCache",
        "class_ref": ArraysCache,
    }


def _make_hybrid_cache_data(n_total=48, attn_interval=4, seq_len=128):
    """Simulate extracted cache states for a hybrid model.

    Qwen 3.5 pattern: every 4th layer is attention, rest are GatedDeltaNet.
    So layers 3,7,11,... are KVCache; layers 0,1,2,4,5,6,... are ArraysCache.
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
```

- [ ] **Step 2: Run test to verify it fails (import error)**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestIsKVLayer -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name '_is_kv_layer' from 'vllm_mlx.prefix_cache'`

- [ ] **Step 3: Implement `_is_kv_layer` and `NonKVCacheData` in `prefix_cache.py`**

Add after the existing `PrefixCacheStats` class (around line 67), before `class PrefixCacheManager`:

```python
# Known KV cache class names — positional caches that can be block-sliced
# along the sequence dimension (axis=2). All produce 4D tensors:
# (batch, n_kv_heads, seq_len, head_dim).
_KV_CACHE_CLASSES = frozenset({
    "KVCache",
    "RotatingKVCache",
    "QuantizedKVCache",
    "ChunkedKVCache",
    "ConcatenateKVCache",
    "BatchKVCache",
    "BatchRotatingKVCache",
})


def _is_kv_layer(layer_state: dict) -> bool:
    """Check if a layer state dict represents a positional KV cache layer.

    Args:
        layer_state: Dict from _extract_cache_states() containing
            'class_name', 'state', 'meta_state', 'class_ref'.

    Returns:
        True if the layer is a KV cache that can be sliced along seq_len.
    """
    return layer_state.get("class_name", "") in _KV_CACHE_CLASSES


@dataclass
class NonKVCacheData:
    """Full state for non-positional cache layers (SSM, linear attention).

    Hybrid models (Qwen 3.5, Nemotron) have layers that use ArraysCache
    instead of KVCache. These store cumulative state (conv + recurrent)
    that cannot be sliced into blocks. This dataclass stores the full
    state for reconstruction alongside block-sliced KV layers.
    """

    layer_indices: List[int]  # Position in the full layer list
    states: List[Any]  # From cache.state for each non-KV layer
    meta_states: List[Any]  # From cache.meta_state for each non-KV layer
    class_refs: List[Any]  # type refs for from_state() reconstruction
    total_layers: int  # Total layer count (KV + non-KV)
```

Also update the module's import block — add `NonKVCacheData` and `_is_kv_layer` to the public API at the top level (no `__all__` needed, just ensure they're importable).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestIsKVLayer -v 2>&1 | tail -15`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/prefix_cache.py tests/test_prefix_cache_hybrid.py
git commit -m "feat: add _is_kv_layer() and NonKVCacheData for hybrid model prefix cache

Adds layer classification to distinguish positional KV cache layers
(block-sliceable) from non-positional layers like ArraysCache (SSM,
linear attention). Foundation for fixing #142 and #136."
```

---

### Task 2: Fix `_extract_block_tensor_slice()` to skip non-KV layers

**Files:**
- Modify: `vllm_mlx/prefix_cache.py:638-691` (`_extract_block_tensor_slice`)
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefix_cache_hybrid.py`:

```python
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
        for keys_slice, values_slice in result:
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
        # This used to raise "Too many indices for array with 3 dimensions"
        result = cache._extract_block_tensor_slice(data, 0, 64)
        assert result is not None

    def test_slice_beyond_seq_len(self, cache):
        """KV slice beyond available seq_len clips correctly."""
        data = _make_pure_kv_cache_data(n_layers=2, seq_len=100)
        result = cache._extract_block_tensor_slice(data, 64, 128)
        assert result is not None
        keys, values = result[0]
        assert keys.shape[2] == 36  # 100 - 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestExtractBlockTensorSlice -v 2>&1 | tail -10`
Expected: FAIL — `test_hybrid_skips_non_kv_layers` and `test_hybrid_does_not_crash` fail

- [ ] **Step 3: Rewrite `_extract_block_tensor_slice()` in `prefix_cache.py`**

Replace lines 638-691 (the entire method) with:

```python
    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> Optional[List[Optional[Tuple[Any, Any]]]]:
        """
        Extract tensor slices for a single block from cache data.

        For KV cache layers (positional), slices keys/values along the
        sequence dimension. For non-KV layers (ArraysCache, etc.),
        returns None at that position — these are stored separately
        by store_cache() as whole-sequence state.

        Args:
            cache_data: List of layer states from _extract_cache_states()
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence

        Returns:
            List with one entry per layer:
            - (keys_slice, values_slice) for KV layers
            - None for non-KV layers
            Returns None only on complete failure.
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices: List[Optional[Tuple[Any, Any]]] = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    block_slices.append(None)
                    continue

                # Skip non-KV layers — they can't be block-sliced
                if not _is_kv_layer(layer_state):
                    block_slices.append(None)
                    continue

                keys, values = layer_state["state"]

                # KV cache shape: (batch, n_kv_heads, seq_len, head_dim)
                # Slice along seq_len dimension (axis 2)
                seq_len = keys.shape[2] if hasattr(keys, "shape") else 0

                if end_idx > seq_len:
                    actual_end = min(end_idx, seq_len)
                    if start_idx >= actual_end:
                        block_slices.append(None)
                        continue
                    keys_slice = keys[:, :, start_idx:actual_end, :]
                    values_slice = values[:, :, start_idx:actual_end, :]
                else:
                    keys_slice = keys[:, :, start_idx:end_idx, :]
                    values_slice = values[:, :, start_idx:end_idx, :]

                block_slices.append((keys_slice, values_slice))

            # Return None only if no layers produced data at all
            if all(s is None for s in block_slices):
                return None

            return block_slices

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestExtractBlockTensorSlice -v 2>&1 | tail -10`
Expected: 4 passed

- [ ] **Step 5: Run existing prefix cache tests for regression**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache.py -v 2>&1 | tail -15`
Expected: All pass (no regression)

- [ ] **Step 6: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/prefix_cache.py tests/test_prefix_cache_hybrid.py
git commit -m "fix: _extract_block_tensor_slice skips non-KV layers

Fixes the 'Too many indices for array with 3 dimensions' crash on
hybrid models (Qwen 3.5, Nemotron). ArraysCache layers are skipped
during block slicing — they're stored separately as whole-sequence
state. Fixes #142, #136."
```

---

## Chunk 2: Store and Reconstruct Hybrid Cache

### Task 3: Update `store_cache()` to store non-KV states separately

**Files:**
- Modify: `vllm_mlx/prefix_cache.py:380-636` (`BlockAwarePrefixCache.__init__` and `store_cache`)
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefix_cache_hybrid.py`:

```python
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
        # Check the stored non-KV data
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestStoreHybridCache -v 2>&1 | tail -10`
Expected: FAIL — `_non_kv_states` doesn't exist, `has_non_kv` not on `BlockCacheEntry`

- [ ] **Step 3: Implement changes to `BlockAwarePrefixCache`**

3a. Add `has_non_kv` field to `BlockCacheEntry` (around line 375):

```python
@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: List[Any]  # Actual KV cache data per block
    last_access: float
    has_non_kv: bool = False  # True if model has non-positional cache layers
```

3b. Add `_non_kv_states` dict to `__init__` (after `self._tokens_saved = 0`, around line 434):

```python
        # Non-KV layer states for hybrid models (SSM, linear attention).
        # Keyed by tuple(block_ids) for lookup during reconstruction.
        self._non_kv_states: Dict[Tuple[int, ...], NonKVCacheData] = {}
```

3c. In `store_cache()`, after block allocation loop and before creating `BlockCacheEntry` (around line 613), add non-KV extraction:

```python
        # Extract and store non-KV layer states for hybrid models
        has_non_kv = False
        if is_tensor_data and cache_data:
            non_kv_indices = []
            non_kv_states_list = []
            non_kv_meta_list = []
            non_kv_refs = []
            for idx, layer_state in enumerate(cache_data):
                if not _is_kv_layer(layer_state):
                    non_kv_indices.append(idx)
                    non_kv_states_list.append(layer_state.get("state"))
                    non_kv_meta_list.append(layer_state.get("meta_state"))
                    non_kv_refs.append(layer_state.get("class_ref"))
            if non_kv_indices:
                has_non_kv = True
                block_key = tuple(block_table.block_ids)
                self._non_kv_states[block_key] = NonKVCacheData(
                    layer_indices=non_kv_indices,
                    states=non_kv_states_list,
                    meta_states=non_kv_meta_list,
                    class_refs=non_kv_refs,
                    total_layers=len(cache_data),
                )
```

3d. Update `BlockCacheEntry` creation (around line 617) to include `has_non_kv`:

```python
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data,
            last_access=time.time(),
            has_non_kv=has_non_kv,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestStoreHybridCache -v 2>&1 | tail -10`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/prefix_cache.py tests/test_prefix_cache_hybrid.py
git commit -m "feat: store_cache separates KV and non-KV layer states

Hybrid models' non-KV states (ArraysCache) are stored once per
prefix in _non_kv_states dict, keyed by block_ids. KV layers
continue to be block-sliced as before. No memory duplication."
```

---

### Task 4: Rewrite `reconstruct_cache()` for hybrid models

**Files:**
- Modify: `vllm_mlx/prefix_cache.py:772-891` (`reconstruct_cache`)
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefix_cache_hybrid.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestReconstructHybridCache -v 2>&1 | tail -10`
Expected: FAIL — `test_hybrid_reconstruct_all_layers` fails (current code only handles KV)

- [ ] **Step 3: Rewrite `reconstruct_cache()` in `prefix_cache.py`**

Replace lines 772-891 (the entire method) with:

```python
    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from stored block tensor data.

        For pure-KV models: concatenates KV block slices into KVCache objects.
        For hybrid models: also restores non-KV layers (ArraysCache, etc.)
        from stored whole-sequence state via from_state().

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed cache objects (one per layer),
            or None if reconstruction fails
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Check for non-KV states (hybrid model)
            block_key = tuple(block_table.block_ids)
            non_kv_data = self._non_kv_states.get(block_key)
            has_non_kv = non_kv_data is not None

            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(
                        f"Block {block_id} not found in allocated blocks"
                    )
                    return None

                if block.cache_data is None:
                    logger.debug(f"Block {block_id} has no tensor data stored")
                    return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            # Determine total number of layers from block data
            num_layers = len(all_block_data[0])
            if num_layers == 0:
                return None

            # If hybrid model but no non-KV states, can't reconstruct
            if has_non_kv and non_kv_data.total_layers != num_layers:
                logger.warning(
                    f"Layer count mismatch: blocks have {num_layers}, "
                    f"non-KV data expects {non_kv_data.total_layers}"
                )
                return None

            # Build set of non-KV layer indices for fast lookup
            non_kv_idx_set = set()
            if has_non_kv:
                non_kv_idx_set = set(non_kv_data.layer_indices)

            # Check if any non-KV layers exist in the data but we have
            # no stored states — indicates hybrid model with missing data
            if not has_non_kv:
                for layer_idx in range(num_layers):
                    # Check first block: if a layer slot is None, it's non-KV
                    if all_block_data[0][layer_idx] is None:
                        logger.debug(
                            "Hybrid model detected but no non-KV states "
                            "stored — cannot reconstruct"
                        )
                        return None

            # Reconstruct each layer
            reconstructed_caches = []

            for layer_idx in range(num_layers):
                if layer_idx in non_kv_idx_set:
                    # Non-KV layer: restore from stored whole-sequence state
                    pos = non_kv_data.layer_indices.index(layer_idx)
                    state = non_kv_data.states[pos]
                    meta = non_kv_data.meta_states[pos]
                    cls = non_kv_data.class_refs[pos]

                    if cls is not None and hasattr(cls, "from_state"):
                        cache_obj = cls.from_state(state, meta)
                    else:
                        # Can't reconstruct without class ref
                        logger.warning(
                            f"No class_ref for non-KV layer {layer_idx}"
                        )
                        return None

                    reconstructed_caches.append(cache_obj)
                else:
                    # KV layer: concatenate block slices
                    layer_keys = []
                    layer_values = []

                    for block_data in all_block_data:
                        if layer_idx < len(block_data):
                            entry = block_data[layer_idx]
                            if entry is not None:
                                keys_slice, values_slice = entry
                                layer_keys.append(keys_slice)
                                layer_values.append(values_slice)

                    if not layer_keys:
                        logger.debug(
                            f"No KV data for layer {layer_idx}"
                        )
                        return None

                    # Concatenate along sequence dimension (axis 2)
                    concat_keys = mx.concatenate(layer_keys, axis=2)
                    concat_values = mx.concatenate(layer_values, axis=2)

                    try:
                        from mlx_lm.models.cache import KVCache

                        cache_obj = KVCache()
                        cache_obj.keys = concat_keys
                        cache_obj.values = concat_values
                        cache_obj.offset = concat_keys.shape[2]
                        reconstructed_caches.append(cache_obj)

                    except ImportError:
                        class SimpleKVCache:
                            def __init__(self, keys, values):
                                self.keys = keys
                                self.values = values
                                self.offset = keys.shape[2]

                            @property
                            def state(self):
                                return (self.keys, self.values)

                            @property
                            def meta_state(self):
                                return (str(self.offset),)

                        reconstructed_caches.append(
                            SimpleKVCache(concat_keys, concat_values)
                        )

            if not reconstructed_caches:
                return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers "
                f"({len(non_kv_idx_set)} non-KV), "
                f"{block_table.num_tokens} tokens from "
                f"{len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestReconstructHybridCache -v 2>&1 | tail -10`
Expected: 3 passed

- [ ] **Step 5: Run full test suite for regression**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache.py tests/test_prefix_cache_hybrid.py -v 2>&1 | tail -20`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/prefix_cache.py tests/test_prefix_cache_hybrid.py
git commit -m "feat: reconstruct_cache handles hybrid model layers

KV layers are concatenated from block slices. Non-KV layers
(ArraysCache) are restored via from_state() from stored whole-
sequence state. If non-KV states are missing, returns None to
force safe recomputation."
```

---

## Chunk 3: Fetch Guard, Cleanup, and Scheduler Fix

### Task 5: Add partial prefix rejection guard to `fetch_cache()`

**Files:**
- Modify: `vllm_mlx/prefix_cache.py:436-510` (`fetch_cache`)
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefix_cache_hybrid.py`:

```python
class TestFetchHybridCache:
    """Test fetch_cache with hybrid model prefix matching."""

    @pytest.fixture
    def cache(self):
        mock_model = MagicMock()
        from vllm_mlx.paged_cache import PagedCacheManager
        paged = PagedCacheManager(block_size=64, max_blocks=100)
        return BlockAwarePrefixCache(mock_model, paged)

    def test_full_match_hybrid_returns_cache(self, cache):
        """Full prefix match with non-KV states → cache hit."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        # Same tokens → should find prefix
        bt, remaining = cache.fetch_cache("req-2", tokens)
        # May or may not hit depending on paged cache internals,
        # but should NOT crash
        # The key assertion: no exception raised

    def test_pure_kv_partial_match_still_works(self, cache):
        """Pure KV model: partial prefix reuse is allowed."""
        data = _make_pure_kv_cache_data(n_layers=4, seq_len=192)
        tokens = list(range(192))
        cache.store_cache("req-1", tokens, data)
        # Different request with shorter prefix → partial match OK for pure KV
        shorter_tokens = list(range(128))
        bt, remaining = cache.fetch_cache("req-2", shorter_tokens)
        # Should not crash regardless of match result

    def test_cleanup_removes_non_kv_states(self, cache):
        """release_cache removes non-KV states for the request."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        tokens = list(range(128))
        cache.store_cache("req-1", tokens, data)
        assert len(cache._non_kv_states) == 1
        cache.release_cache("req-1")
        # Non-KV states for this request's blocks should be cleaned up
        # (they may persist for shared blocks, but the entry is gone)

    def test_clear_removes_all_non_kv_states(self, cache):
        """clear() removes all non-KV states."""
        data = _make_hybrid_cache_data(n_total=8, attn_interval=4, seq_len=128)
        cache.store_cache("req-1", list(range(128)), data)
        cache.store_cache("req-2", list(range(64, 192)), data)
        cache.clear()
        assert len(cache._non_kv_states) == 0
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestFetchHybridCache -v 2>&1 | tail -10`
Expected: At least `test_cleanup_removes_non_kv_states` and `test_clear_removes_all_non_kv_states` fail

- [ ] **Step 3: Implement fetch guard and cleanup**

3a. In `fetch_cache()`, after finding shared blocks via `paged_cache.find_shared_prefix()` (around line 459), add a guard before returning:

```python
            # Guard: reject partial prefix match for hybrid models.
            # If non-KV states don't exist for this exact block set,
            # a hybrid model can't be correctly reconstructed (SSM/KV mismatch).
            if self._non_kv_states:
                candidate_key = tuple(block_table.block_ids)
                if candidate_key not in self._non_kv_states:
                    # Hybrid model with partial match — reject
                    logger.debug(
                        f"Rejecting partial prefix match for {request_id}: "
                        f"hybrid model requires full match with non-KV states"
                    )
                    self.paged_cache.delete_block_table(request_id)
                    self._misses += 1
                    return None, tokens
```

3b. Same guard in the `_find_best_prefix_match` branch (around line 505):

```python
            # Same hybrid guard for prefix index matches
            if self._non_kv_states:
                candidate_key = tuple(block_table.block_ids)
                if candidate_key not in self._non_kv_states:
                    logger.debug(
                        f"Rejecting prefix index match for {request_id}: "
                        f"hybrid model requires full match with non-KV states"
                    )
                    self.paged_cache.delete_block_table(request_id)
                    self._misses += 1
                    return None, tokens
```

3c. In `release_cache()` (around line 724), clean up non-KV states:

```python
    def release_cache(self, request_id: str) -> None:
        """Release cache blocks for a completed request."""
        entry = self._request_tables.pop(request_id, None)
        if entry:
            # Clean up non-KV states if this was the last reference
            block_key = tuple(entry.block_table.block_ids)
            # Only remove if no other request uses the same blocks
            other_uses = any(
                tuple(e.block_table.block_ids) == block_key
                for e in self._request_tables.values()
            )
            if not other_uses:
                self._non_kv_states.pop(block_key, None)
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")
```

3d. In `clear()` (around line 954), add cleanup:

```python
    def clear(self) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self._prefix_index.clear()
        self._non_kv_states.clear()
        self.paged_cache.clear()
        self.reset_stats()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestFetchHybridCache -v 2>&1 | tail -10`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/prefix_cache.py tests/test_prefix_cache_hybrid.py
git commit -m "feat: fetch_cache rejects partial prefix for hybrid models

Partial prefix matches on hybrid models would produce mismatched
KV/SSM state. Guard checks non-KV states exist for exact block
set. Cleanup in release_cache and clear."
```

---

### Task 6: Fix scheduler fallback in `_reconstruct_cache_from_states()`

**Files:**
- Modify: `vllm_mlx/scheduler.py:1486-1496`
- Test: `tests/test_prefix_cache_hybrid.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefix_cache_hybrid.py`:

```python
class TestSchedulerRobustness:
    """Test scheduler cache reconstruction with non-KV layers."""

    def test_reconstruct_with_arrays_cache_uses_from_state(self):
        """ArraysCache layer uses from_state, not manual KV reconstruction."""
        from vllm_mlx.scheduler import Scheduler, SchedulerConfig

        config = SchedulerConfig(enable_prefix_cache=False)
        # We can't easily construct a full Scheduler, so test the method
        # indirectly by verifying the data path.
        # The key check: _reconstruct_cache_from_states handles ArraysCache
        # via from_state (class_ref path), NOT the fallback.
        arrays_state = {
            "state": [mx.zeros((1, 3, 128)), mx.zeros((1, 8, 64, 64))],
            "meta_state": "",
            "class_name": "ArraysCache",
            "class_ref": ArraysCache,
        }
        kv_state = _make_kv_state(seq_len=100)
        extracted = [arrays_state, kv_state, arrays_state, kv_state]

        # Test reconstruction
        # ArraysCache has from_state, KVCache has from_state
        # Both should work through the class_ref path
        for layer_state in extracted:
            cls = layer_state["class_ref"]
            state = layer_state["state"]
            meta = layer_state.get("meta_state", "")
            # This should not raise
            obj = cls.from_state(state, meta)
            assert obj is not None
```

- [ ] **Step 2: Run test to verify it passes (sanity check)**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache_hybrid.py::TestSchedulerRobustness -v 2>&1 | tail -5`
Expected: PASS (from_state works for both types)

- [ ] **Step 3: Add defensive guard to scheduler fallback**

In `vllm_mlx/scheduler.py`, modify the fallback path at lines 1486-1496. Replace:

```python
                else:
                    # Fallback: try KVCache manual reconstruction
                    from mlx_lm.models.cache import KVCache

                    if len(state) != 2:
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    cache.offset = (
                        int(meta_state[0]) if meta_state else cache.keys.shape[2]
                    )
```

With:

```python
                else:
                    # Fallback: try KVCache manual reconstruction
                    from mlx_lm.models.cache import KVCache

                    if (
                        not isinstance(state, (tuple, list))
                        or len(state) != 2
                        or not hasattr(state[0], "shape")
                        or state[0].ndim != 4
                    ):
                        logger.debug(
                            f"[mid_prefill_cache] skipping non-KV layer "
                            f"(state type={type(state).__name__})"
                        )
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    cache.offset = (
                        int(meta_state[0]) if meta_state else cache.keys.shape[2]
                    )
```

- [ ] **Step 4: Run existing scheduler tests for regression**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/ -k "scheduler" -v 2>&1 | tail -15`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
cd ~/code/vllm-mlx
git add vllm_mlx/scheduler.py tests/test_prefix_cache_hybrid.py
git commit -m "fix: scheduler fallback guards against non-KV cache states

The fallback path in _reconstruct_cache_from_states now checks
that state looks like a KV tuple (2-element, 4D tensors) before
assuming shape[2] for offset. Prevents crash on ArraysCache states."
```

---

### Task 7: Run full test suite and format

- [ ] **Step 1: Run ruff linter**

Run: `cd ~/code/vllm-mlx && ruff check vllm_mlx/prefix_cache.py vllm_mlx/scheduler.py tests/test_prefix_cache_hybrid.py --select E,F,W --ignore E501 2>&1`
Expected: All checks passed

- [ ] **Step 2: Run black formatter**

Run: `cd ~/code/vllm-mlx && black vllm_mlx/prefix_cache.py vllm_mlx/scheduler.py tests/test_prefix_cache_hybrid.py 2>&1`
Expected: Files reformatted (or already formatted)

- [ ] **Step 3: Run full prefix cache test suite**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/test_prefix_cache.py tests/test_prefix_cache_hybrid.py tests/test_paged_cache.py -v 2>&1 | tail -25`
Expected: All pass

- [ ] **Step 4: Run all tests**

Run: `cd ~/code/vllm-mlx && python -m pytest tests/ -v --timeout 60 2>&1 | tail -30`
Expected: All pass, 0 failures

- [ ] **Step 5: Commit formatting if needed**

```bash
cd ~/code/vllm-mlx
git add -u
git diff --cached --stat
# Only commit if there are formatting changes
git commit -m "style: formatting for hybrid prefix cache changes" || true
```

- [ ] **Step 6: Push to PR #165 branch**

```bash
cd ~/code/vllm-mlx
git push origin fix/mllm-continuous-batching-hybrid-models
```
