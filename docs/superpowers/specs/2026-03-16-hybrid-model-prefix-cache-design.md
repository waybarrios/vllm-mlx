# Hybrid Model Prefix Cache — Design Spec

**Date:** 2026-03-16
**Issues:** #142, #136
**Branch:** `fix/mllm-continuous-batching-hybrid-models` (PR #165)

## Problem

`BlockAwarePrefixCache` crashes on hybrid models (Qwen 3.5 MoE, Nemotron) with:
```
WARNING: Failed to extract block tensor slice: Too many indices for array with 3 dimensions.
```
Prefix cache silently degrades to no-op. No TTFT benefit on repeated contexts.

## Root Cause

The issue reporters hypothesized that MoE models produce 3D KV tensors. **This is wrong.** All KV cache tensors in mlx-lm are always 4D `(B, n_kv_heads, seq_len, head_dim)` regardless of model architecture.

The actual cause: hybrid models have two cache layer types:

| Layer Type | Cache Class | `.state` Format | Positional? |
|-----------|-------------|-----------------|-------------|
| Attention | KVCache | `(keys_4D, values_4D)` | Yes — indexed by token position |
| GatedDeltaNet / Mamba | ArraysCache | `[conv_3D, recurrent_4D]` | No — cumulative summary |

`_extract_cache_states()` extracts state from ALL layers. `_extract_block_tensor_slice()` then tries to slice every layer as 4D KV:

```python
keys, values = layer_state["state"]  # Unpacks ArraysCache [conv_3D, ssm_4D]
keys[:, :, start:end, :]             # 4 indices on 3D conv_state → IndexError
```

## Design

### Layer Classification

New helper function identifies cache layer types by `class_name` (already stored by `_extract_cache_states()`):

```python
_KV_CACHE_CLASSES = frozenset({
    "KVCache", "RotatingKVCache", "QuantizedKVCache",
    "ChunkedKVCache", "ConcatenateKVCache", "BatchKVCache",
    "BatchRotatingKVCache",
})

def _is_kv_layer(layer_state: dict) -> bool:
    return layer_state.get("class_name", "") in _KV_CACHE_CLASSES
```

### Separate Storage Model

KV and non-KV layers are stored differently — no duplication:

- **KV layers** → block-sliced along seq_dim (axis=2), stored per-block in `KVCacheBlock.cache_data`
- **Non-KV layers** → stored once as whole-sequence state in `BlockAwarePrefixCache._non_kv_states`, keyed by `tuple(block_ids)`

```python
@dataclass
class NonKVCacheData:
    """Full state for non-positional cache layers (SSM, linear attention)."""
    layer_indices: List[int]       # Position in the full layer list
    states: List[Any]              # From cache.state (list/tuple of arrays)
    meta_states: List[Any]         # From cache.meta_state
    class_refs: List[type]         # For from_state() reconstruction
    total_layers: int              # Total layer count (KV + non-KV)
```

### Partial Prefix Rejection

For hybrid models, partial prefix reuse (matching some but not all blocks) would restore KV cache but NOT SSM state. The model would generate with mismatched attention/SSM state, producing incorrect output.

Guard: when `fetch_cache()` finds a partial match and the model has non-KV layers, return cache miss instead.

Full matches work because non-KV states are stored keyed by exact `tuple(block_ids)`.

### Changes

#### `prefix_cache.py`

1. **`_is_kv_layer()`** — classify layers by class_name

2. **`_extract_block_tensor_slice()`**:
   - Skip non-KV layers (append `None` placeholder)
   - KV layers: slice `keys[:,:,start:end,:]` as before
   - Return list with `None` gaps at non-KV positions

3. **`store_cache()`**:
   - Classify layers into KV and non-KV
   - KV: block-slice via `_extract_block_tensor_slice()`
   - Non-KV: extract states, store in `self._non_kv_states[tuple(block_ids)]`
   - Set `has_non_kv` flag on `BlockCacheEntry` for fast checks

4. **`reconstruct_cache()`**:
   - Look up non-KV states via `tuple(block_table.block_ids)`
   - KV layers: concatenate block slices → KVCache (as before)
   - Non-KV layers: `class_ref.from_state(state, meta_state)`
   - Interleave in correct layer order using stored indices
   - If non-KV states missing for hybrid model → return `None`

5. **`fetch_cache()` partial match guard**:
   - After `find_shared_prefix()` or `_find_best_prefix_match()`
   - Check if non-KV states exist for the matched block set
   - Missing → return `(None, tokens)` (force recomputation)

6. **Cleanup**: `release_cache()`, `clear()` clean `_non_kv_states`

#### `scheduler.py`

- `_reconstruct_cache_from_states()` fallback path (L1490): guard `shape[2]` access with `hasattr(state[0], 'shape') and len(state) == 2` before assuming KV format

### What This Does NOT Change

- `PrefixCacheManager` (trie-based) — already works for hybrid models (stores whole cache objects)
- `_extract_cache_states()` — already correctly stores `class_name` and `class_ref`
- `_can_trim_cache()` / `_trim_cache()` — already handles hybrid models (checks all layers)
- KVCache tensor shape handling — always 4D, no ndim checks needed
- Pure-KV model behavior — unchanged, no non-KV layers detected

### Tests

New test file: `tests/test_prefix_cache_hybrid.py`

1. **`test_is_kv_layer`** — classification of KVCache vs ArraysCache
2. **`test_extract_block_tensor_slice_hybrid`** — mixed KV/ArraysCache layers, KV sliced, ArraysCache skipped
3. **`test_store_and_reconstruct_hybrid`** — full roundtrip with simulated hybrid model cache
4. **`test_partial_prefix_rejected_for_hybrid`** — partial match → cache miss
5. **`test_full_prefix_hit_for_hybrid`** — exact match → correct reconstruction
6. **`test_pure_kv_model_unchanged`** — regression: existing behavior for pure attention models
7. **`test_cleanup_non_kv_states`** — release_cache and clear remove non-KV data
