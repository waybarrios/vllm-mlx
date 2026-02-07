# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware prefix cache for vllm-mlx.

This module provides a prefix cache implementation that tracks memory usage
and evicts entries based on memory pressure rather than entry count.

Key features:
- Automatic memory limit detection based on available system RAM
- Accurate memory tracking for MLX array caches
- LRU eviction triggered by memory thresholds
- No unnecessary deep copies (MLX arrays are immutable)

Example:
    config = MemoryCacheConfig(max_memory_percent=0.25)
    cache = MemoryAwarePrefixCache(model, config)

    # Fetch returns reference (no copy) - safe because MLX arrays are immutable
    kv_cache, remaining = cache.fetch(tokens)

    # Store tracks memory automatically
    cache.store(tokens, kv_cache)
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Constants
_BYTES_PER_MB = 1024 * 1024
_DEFAULT_MEMORY_PERCENT = 0.20  # 20% of available RAM
_MIN_MEMORY_BYTES = 100 * _BYTES_PER_MB  # Minimum 100MB
_MAX_ENTRIES_FALLBACK = 50  # Fallback if memory detection fails


def _get_available_memory() -> int:
    """
    Get available system memory in bytes.

    Returns:
        Available memory in bytes, or 0 if detection fails.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        logger.warning("psutil not installed, using fallback memory limit")
        return 0
    except Exception as e:
        logger.warning(f"Failed to detect available memory: {e}")
        return 0


def _array_memory(arr) -> int:
    """
    Estimate array memory from shape+dtype without triggering lazy eval.

    Accessing .nbytes on a lazy MLX array forces evaluation of the entire
    computation graph, causing a VRAM spike. This function uses shape and
    dtype metadata (which are always available without eval) to compute
    the same value.

    Args:
        arr: An MLX array or similar object.

    Returns:
        Estimated memory in bytes.
    """
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
        dtype = arr.dtype
        if hasattr(dtype, "size"):
            return math.prod(arr.shape) * dtype.size
    # Fallback for non-MLX arrays or objects without shape/dtype
    if hasattr(arr, "nbytes"):
        return arr.nbytes
    return 0


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint using shape+dtype metadata to avoid triggering
    lazy evaluation (which would cause a VRAM spike).

    Args:
        cache: List of layer cache objects, each containing keys/values tensors.

    Returns:
        Estimated memory usage in bytes.
    """
    if not cache:
        return 0

    total_bytes = 0

    for layer_cache in cache:
        # Handle different cache object types
        # Check dict first since dicts have .keys() method that would match below
        if isinstance(layer_cache, dict) and "state" in layer_cache:
            # Extracted state dict
            keys, values = layer_cache["state"]
            total_bytes += _array_memory(keys)
            total_bytes += _array_memory(values)
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with state property returning (keys, values)
            try:
                keys, values = layer_cache.state
                total_bytes += _array_memory(keys)
                total_bytes += _array_memory(values)
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr):
                total_bytes += _array_memory(keys_attr)
            if not callable(values_attr):
                total_bytes += _array_memory(values_attr)

    return total_bytes


@dataclass(frozen=True)
class MemoryCacheConfig:
    """
    Configuration for memory-aware prefix cache.

    Attributes:
        max_memory_mb: Maximum memory in MB. If None, auto-detects.
        max_memory_percent: Fraction of available RAM to use (0.0-1.0).
        max_entries: Hard limit on number of entries (safety net).
        enable_memory_tracking: Whether to track per-entry memory.
    """

    max_memory_mb: int | None = None
    max_memory_percent: float = _DEFAULT_MEMORY_PERCENT
    max_entries: int = 1000  # Safety limit
    enable_memory_tracking: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.max_memory_percent <= 1.0:
            raise ValueError(
                f"max_memory_percent must be in (0, 1], got {self.max_memory_percent}"
            )
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")

    def compute_memory_limit(self) -> int:
        """
        Compute the memory limit in bytes.

        Returns:
            Memory limit in bytes.
        """
        if self.max_memory_mb is not None:
            return self.max_memory_mb * _BYTES_PER_MB

        available = _get_available_memory()
        if available > 0:
            limit = int(available * self.max_memory_percent)
            return max(limit, _MIN_MEMORY_BYTES)

        # Fallback: assume 8GB system, use configured percent
        fallback_total = 8 * 1024 * _BYTES_PER_MB
        return int(fallback_total * self.max_memory_percent)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    current_memory_bytes: int = 0
    max_memory_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "current_memory_mb": round(self.current_memory_bytes / _BYTES_PER_MB, 2),
            "max_memory_mb": round(self.max_memory_bytes / _BYTES_PER_MB, 2),
            "memory_utilization": round(self.memory_utilization, 4),
            "entry_count": self.entry_count,
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with memory tracking."""

    tokens: tuple[int, ...]
    cache: list[Any]
    memory_bytes: int

    @classmethod
    def create(cls, tokens: list[int], cache: list[Any]) -> _CacheEntry:
        """Create a cache entry with memory estimation."""
        memory = estimate_kv_cache_memory(cache)
        return cls(
            tokens=tuple(tokens),
            cache=cache,
            memory_bytes=memory,
        )


def _trim_cache_offset(cache: list[Any], trim_by: int) -> list[Any]:
    """Create shallow copies of KVCache layers with offset reduced by *trim_by*.

    This is used when returning a cached KV state to the scheduler so that
    the last N positions are "freed" and the model will recompute them on the
    next forward pass (preventing duplicate KV entries).
    """
    from mlx_lm.models.cache import KVCache

    trimmed: list[Any] = []
    for layer_cache in cache:
        if hasattr(layer_cache, "offset") and hasattr(layer_cache, "keys"):
            tc = KVCache.__new__(KVCache)
            tc.keys = layer_cache.keys
            tc.values = layer_cache.values
            tc.offset = max(layer_cache.offset - trim_by, 0)
            trimmed.append(tc)
        else:
            trimmed.append(layer_cache)
    return trimmed


class MemoryAwarePrefixCache:
    """
    Prefix cache with memory-based eviction.

    This cache tracks memory usage per entry and evicts based on memory
    pressure rather than entry count. It uses LRU (Least Recently Used)
    ordering for eviction decisions.

    Key design decisions:
    - No deep copies on fetch: MLX arrays are immutable, so sharing is safe
    - Memory tracking per entry: Accurate accounting for eviction
    - Auto-detection of available RAM: Adapts to different systems
    - OrderedDict for O(1) LRU operations

    Thread Safety:
        This class is NOT thread-safe. Use external locking if needed.
    """

    def __init__(
        self,
        model: Any,
        config: MemoryCacheConfig | None = None,
    ) -> None:
        """
        Initialize the memory-aware prefix cache.

        Args:
            model: The MLX model (used for identification).
            config: Cache configuration. Uses defaults if None.
        """
        self._model_id = id(model)
        self._config = config or MemoryCacheConfig()

        # OrderedDict maintains insertion order for LRU
        # Key: tuple(tokens), Value: _CacheEntry
        self._entries: OrderedDict[tuple[int, ...], _CacheEntry] = OrderedDict()

        # Memory tracking
        self._max_memory = self._config.compute_memory_limit()
        self._current_memory = 0

        # Statistics
        self._stats = CacheStats(max_memory_bytes=self._max_memory)

        logger.info(
            f"MemoryAwarePrefixCache initialized: "
            f"max_memory={self._max_memory / _BYTES_PER_MB:.1f}MB, "
            f"max_entries={self._config.max_entries}"
        )

    def fetch(self, tokens: list[int]) -> tuple[list[Any] | None, list[int]]:
        """
        Find cached KV state for the given tokens.

        This method searches for exact matches and prefix matches.
        Returns the cached KV state directly (no copy) since MLX arrays
        are immutable and safe to share.

        Args:
            tokens: Input token sequence.

        Returns:
            Tuple of (cache, remaining_tokens):
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        if not tokens:
            self._stats.misses += 1
            return None, tokens

        tokens_key = tuple(tokens)

        # Check for exact match
        if tokens_key in self._entries:
            entry = self._entries[tokens_key]
            # Move to end (most recently used)
            self._entries.move_to_end(tokens_key)
            self._stats.hits += 1
            self._stats.tokens_saved += len(tokens)
            # Return reference directly - MLX arrays are immutable
            return entry.cache, []

        # Check for prefix matches (shorter cached sequences)
        best_match: _CacheEntry | None = None
        best_length = 0

        # Also check supersequence matches (longer cached sequences where
        # our tokens are a prefix of the cached key).  This handles the
        # common case of identical repeated prompts: the cache stores
        # prompt+output tokens, but fetch uses only prompt tokens.
        best_super: _CacheEntry | None = None

        # Track longest-common-prefix (LCP) for divergent sequences:
        # cached and request share a prefix but then diverge.  This
        # handles the agentic pattern where the same system+context
        # prefix is reused with a different final user message.
        best_lcp_entry: _CacheEntry | None = None
        best_lcp_length = 0

        for cached_key, entry in self._entries.items():
            cached_len = len(cached_key)
            # Check if cached sequence is a prefix of requested tokens
            if (
                cached_len < len(tokens)
                and cached_len > best_length
                and tokens_key[:cached_len] == cached_key
            ):
                best_match = entry
                best_length = cached_len
            # Check if requested tokens are a prefix of cached sequence
            elif cached_len >= len(tokens) and cached_key[: len(tokens)] == tokens_key:
                # Prefer the longest match (most KV data already computed)
                if best_super is None or cached_len > len(best_super.tokens):
                    best_super = entry
            else:
                # Divergent case: find longest common prefix (LCP).
                # Quick guard: first token must match, and the entry must
                # be long enough to potentially beat the current best.
                min_len = min(cached_len, len(tokens_key))
                current_best = max(best_length, best_lcp_length)
                if min_len > current_best and cached_key[0] == tokens_key[0]:
                    # Compute exact LCP length
                    lcp = 0
                    for j in range(min_len):
                        if cached_key[j] != tokens_key[j]:
                            break
                        lcp = j + 1
                    logger.debug(
                        f"[cache_fetch] LCP scan: cached_len={cached_len} "
                        f"req_len={len(tokens_key)} lcp={lcp} "
                        f"current_best={current_best}"
                    )
                    if lcp > current_best:
                        best_lcp_entry = entry
                        best_lcp_length = lcp

        if best_super is not None:
            # Supersequence hit — cached entry covers all requested tokens.
            # However, for hybrid Mamba+Transformer models the MambaCache
            # state is cumulative and includes output tokens.  Trimming
            # only works for pure KVCache layers.  Check whether ALL
            # layers are trimmable before using the supersequence match.
            n_cached = len(best_super.tokens)
            n_requested = len(tokens)
            excess = n_cached - n_requested

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_super.cache
            )

            if excess > 0 and has_non_trimmable:
                # Cannot safely trim MambaCache/ArraysCache layers.
                # Fall through to prefix match or miss.  The prompt-only
                # entry (stored by _prompt_cache_save) should provide
                # the correct exact-match hit on repeated prompts.
                logger.debug(
                    "[cache_fetch] supersequence match skipped: "
                    "non-trimmable cache layers (hybrid model)"
                )
            elif excess > 0:
                # Pure KVCache model — safe to trim
                trimmed_cache = _trim_cache_offset(best_super.cache, excess)
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                return trimmed_cache, []
            else:
                # Exact length match via supersequence path
                self._entries.move_to_end(best_super.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += n_requested
                return best_super.cache, []

        if best_match is not None:
            # Move matched entry to end (most recently used)
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            return best_match.cache, remaining

        # Divergent prefix match: cached entry and request share a common
        # prefix but then diverge (e.g. same system prompt + context,
        # different final user message).  Trim cached KV state to the
        # shared prefix length and prefill only the divergent suffix.
        if best_lcp_entry is not None and best_lcp_length > best_length:
            excess = len(best_lcp_entry.tokens) - best_lcp_length

            has_non_trimmable = any(
                not (hasattr(lc, "offset") and hasattr(lc, "keys"))
                for lc in best_lcp_entry.cache
            )
            logger.debug(
                f"[cache_fetch] LCP candidate: lcp={best_lcp_length} "
                f"entry_len={len(best_lcp_entry.tokens)} excess={excess} "
                f"non_trimmable={has_non_trimmable} "
                f"cache_layers={len(best_lcp_entry.cache)} "
                f"layer_types={[type(lc).__name__ for lc in best_lcp_entry.cache[:3]]}"
            )

            if not has_non_trimmable:
                trimmed_cache = _trim_cache_offset(best_lcp_entry.cache, excess)
                self._entries.move_to_end(best_lcp_entry.tokens)
                self._stats.hits += 1
                self._stats.tokens_saved += best_lcp_length
                remaining = tokens[best_lcp_length:]
                logger.debug(
                    f"[cache_fetch] LCP hit: shared={best_lcp_length} "
                    f"trimmed={excess} remaining={len(remaining)}"
                )
                return trimmed_cache, remaining

        self._stats.misses += 1

        return None, tokens

    def store(
        self, tokens: list[int], cache: list[Any], evict_prefixes: bool = True
    ) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.
            evict_prefixes: If True, evict existing entries whose token
                sequence is a strict prefix of ``tokens``.  Set to False
                when storing prompt+output entries to preserve prompt-only
                entries created by prompt_cache_save (those are the entries
                that future requests will actually match).

        Returns:
            True if stored successfully, False if rejected.
        """
        if not tokens or not cache:
            return False

        tokens_key = tuple(tokens)

        # If already cached, just update LRU order
        if tokens_key in self._entries:
            self._entries.move_to_end(tokens_key)
            return True

        # Create entry and estimate memory
        entry = _CacheEntry.create(tokens, cache)

        # Check if single entry exceeds limit
        if entry.memory_bytes > self._max_memory:
            logger.warning(
                f"Cache entry too large: {entry.memory_bytes / _BYTES_PER_MB:.1f}MB "
                f"exceeds limit {self._max_memory / _BYTES_PER_MB:.1f}MB"
            )
            return False

        # Prefix-subset eviction: remove entries whose token sequence
        # is a strict prefix of the new entry.  In a multi-turn agentic
        # workload, request N+1 always extends the conversation from
        # request N — keeping the shorter entry wastes memory because
        # it is fully subsumed by the longer one.
        #
        # Skip when evict_prefixes=False (prompt+output stores).  The
        # prompt+output key includes generated tokens that won't appear
        # in the next request's prompt, so it must NOT evict the
        # prompt-only entry which IS the correct prefix for future hits.
        to_remove = (
            [
                key
                for key in self._entries
                if len(key) < len(tokens_key) and tokens_key[: len(key)] == key
            ]
            if evict_prefixes
            else []
        )
        for key in to_remove:
            old = self._entries.pop(key)
            self._current_memory -= old.memory_bytes
            self._stats.evictions += 1
            logger.debug(
                f"[prefix_evict] removed {len(key)} tokens, "
                f"freed {old.memory_bytes / _BYTES_PER_MB:.2f}MB, "
                f"new_entry={len(tokens_key)} tokens"
            )
        if to_remove:
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory

        # Evict until we have room
        while (
            self._current_memory + entry.memory_bytes > self._max_memory
            or len(self._entries) >= self._config.max_entries
        ) and self._entries:
            self._evict_lru()

        # Store entry
        self._entries[tokens_key] = entry
        self._current_memory += entry.memory_bytes
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"Stored cache: {len(tokens)} tokens, "
            f"{entry.memory_bytes / _BYTES_PER_MB:.2f}MB, "
            f"total={self._current_memory / _BYTES_PER_MB:.1f}MB"
        )

        return True

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return

        # popitem(last=False) removes oldest entry (FIFO order = LRU)
        tokens_key, entry = self._entries.popitem(last=False)
        self._current_memory -= entry.memory_bytes
        self._stats.evictions += 1
        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        logger.debug(
            f"[lru_evict] removed {len(tokens_key)} tokens, "
            f"freed {entry.memory_bytes / _BYTES_PER_MB:.2f}MB"
        )

    def remove(self, tokens: list[int]) -> bool:
        """
        Remove a specific cache entry.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if entry was found and removed.
        """
        tokens_key = tuple(tokens)
        entry = self._entries.pop(tokens_key, None)
        if entry is not None:
            self._current_memory -= entry.memory_bytes
            self._stats.entry_count = len(self._entries)
            self._stats.current_memory_bytes = self._current_memory
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._current_memory = 0
        self._stats = CacheStats(max_memory_bytes=self._max_memory)
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics while preserving cache contents."""
        self._stats = CacheStats(
            max_memory_bytes=self._max_memory,
            current_memory_bytes=self._current_memory,
            entry_count=len(self._entries),
        )

    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory / _BYTES_PER_MB

    @property
    def memory_limit_mb(self) -> float:
        """Memory limit in MB."""
        return self._max_memory / _BYTES_PER_MB

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def __contains__(self, tokens: list[int]) -> bool:
        """Check if tokens are cached."""
        return tuple(tokens) in self._entries

    # -----------------------------------------------------------------
    # Disk persistence — survives server restarts
    # -----------------------------------------------------------------

    def save_to_disk(self, cache_dir: str) -> bool:
        """Save all cache entries to disk using mlx_lm's safetensors format.

        Directory layout::

            cache_dir/
              index.json          # token keys + metadata per entry
              entry_0.safetensors # KV arrays for entry 0
              entry_1.safetensors
              ...

        Returns True if at least one entry was saved.
        """
        import json
        import os
        import time as _time

        if not self._entries:
            logger.info("[cache_persist] nothing to save (0 entries)")
            return False

        t0 = _time.monotonic()
        os.makedirs(cache_dir, exist_ok=True)

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot save")
            return False

        index = {
            "version": 2,
            "num_entries": len(self._entries),
            "total_memory_bytes": self._current_memory,
            "entries": [],
        }

        saved = 0
        for i, (tokens_key, entry) in enumerate(self._entries.items()):
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            try:
                save_prompt_cache(
                    entry_path,
                    entry.cache,
                    metadata={"num_tokens": str(len(tokens_key))},
                )
                # Save tokens separately (can be 100K+ ints → binary is smaller)
                tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")
                import array as _array

                arr = _array.array("i", tokens_key)  # 32-bit signed ints
                with open(tokens_path, "wb") as f:
                    arr.tofile(f)

                index["entries"].append(
                    {
                        "index": i,
                        "num_tokens": len(tokens_key),
                        "memory_bytes": entry.memory_bytes,
                    }
                )
                saved += 1
                logger.info(
                    f"[cache_persist] saved entry {i}: "
                    f"{len(tokens_key)} tokens, "
                    f"{entry.memory_bytes / _BYTES_PER_MB:.1f}MB KV, "
                    f"file={entry_path}"
                )
            except Exception as e:
                logger.warning(f"[cache_persist] failed to save entry {i}: {e}")

        index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] SAVED {saved}/{len(self._entries)} entries "
            f"to {cache_dir} in {dt:.1f}s "
            f"({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return saved > 0

    def load_from_disk(self, cache_dir: str) -> int:
        """Load cache entries from disk.

        Returns the number of entries successfully loaded.
        """
        import json
        import os
        import time as _time

        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            logger.info(f"[cache_persist] no index at {index_path}, nothing to load")
            return 0

        t0 = _time.monotonic()

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except ImportError:
            logger.warning("[cache_persist] mlx_lm not available, cannot load")
            return 0

        with open(index_path) as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version < 2:
            logger.warning(f"[cache_persist] unsupported version {version}, skipping")
            return 0

        loaded = 0
        for entry_meta in index.get("entries", []):
            i = entry_meta["index"]
            entry_path = os.path.join(cache_dir, f"entry_{i}.safetensors")
            tokens_path = os.path.join(cache_dir, f"entry_{i}_tokens.bin")

            if not os.path.exists(entry_path) or not os.path.exists(tokens_path):
                logger.warning(f"[cache_persist] missing files for entry {i}, skipping")
                continue

            try:
                # Load tokens from binary
                import array as _array

                arr = _array.array("i")
                with open(tokens_path, "rb") as f:
                    arr.fromfile(f, entry_meta["num_tokens"])
                tokens = list(arr)

                # Load KV cache
                cache = load_prompt_cache(entry_path)

                # Estimate memory
                memory = estimate_kv_cache_memory(cache)

                # Check if it fits
                if self._current_memory + memory > self._max_memory:
                    logger.info(
                        f"[cache_persist] entry {i} would exceed memory limit "
                        f"({(self._current_memory + memory) / _BYTES_PER_MB:.0f}MB > "
                        f"{self._max_memory / _BYTES_PER_MB:.0f}MB), stopping load"
                    )
                    break

                tokens_key = tuple(tokens)
                entry = _CacheEntry(
                    tokens=tokens_key,
                    cache=cache,
                    memory_bytes=memory,
                )
                self._entries[tokens_key] = entry
                self._current_memory += memory
                loaded += 1

                logger.info(
                    f"[cache_persist] loaded entry {i}: "
                    f"{len(tokens)} tokens, "
                    f"{memory / _BYTES_PER_MB:.1f}MB KV"
                )

            except Exception as e:
                logger.warning(f"[cache_persist] failed to load entry {i}: {e}")

        self._stats.entry_count = len(self._entries)
        self._stats.current_memory_bytes = self._current_memory

        dt = _time.monotonic() - t0
        logger.info(
            f"[cache_persist] LOADED {loaded} entries from {cache_dir} "
            f"in {dt:.1f}s ({self._current_memory / _BYTES_PER_MB:.0f}MB total)"
        )
        return loaded
