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


def estimate_kv_cache_memory(cache: list[Any]) -> int:
    """
    Estimate memory usage of a KV cache in bytes.

    This function inspects MLX arrays in the cache and calculates their
    total memory footprint.

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
            if hasattr(keys, "nbytes"):
                total_bytes += keys.nbytes
            if hasattr(values, "nbytes"):
                total_bytes += values.nbytes
        elif hasattr(layer_cache, "state") and not isinstance(layer_cache, dict):
            # Cache with state property returning (keys, values)
            try:
                keys, values = layer_cache.state
                if hasattr(keys, "nbytes"):
                    total_bytes += keys.nbytes
                if hasattr(values, "nbytes"):
                    total_bytes += values.nbytes
            except (TypeError, ValueError):
                pass
        elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
            # Standard KVCache with keys/values attributes (not dict)
            keys_attr = layer_cache.keys
            values_attr = layer_cache.values
            # Ensure these are arrays, not methods
            if not callable(keys_attr) and hasattr(keys_attr, "nbytes"):
                total_bytes += keys_attr.nbytes
            if not callable(values_attr) and hasattr(values_attr, "nbytes"):
                total_bytes += values_attr.nbytes

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

        if best_match is not None:
            # Move matched entry to end (most recently used)
            self._entries.move_to_end(best_match.tokens)
            self._stats.hits += 1
            self._stats.tokens_saved += best_length
            remaining = tokens[best_length:]
            return best_match.cache, remaining

        self._stats.misses += 1
        return None, tokens

    def store(self, tokens: list[int], cache: list[Any]) -> bool:
        """
        Store KV cache for future reuse.

        This method stores the cache reference directly (no copy) and
        tracks memory usage. If memory limit is exceeded, LRU entries
        are evicted until there's room.

        Args:
            tokens: Token sequence that was processed.
            cache: The computed KV cache to store.

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
            f"Evicted cache: {len(tokens_key)} tokens, "
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
