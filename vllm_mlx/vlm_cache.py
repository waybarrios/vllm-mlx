# SPDX-License-Identifier: Apache-2.0
"""
VLM (Vision Language Model) KV Cache Manager.

This module provides caching for VLM inference, allowing reuse of
computed KV states for repeated image+prompt combinations.

Features:
- Image content hashing for cache keys
- LRU eviction policy
- Stats tracking (hits, misses, tokens saved)
- Support for both single images and multi-image inputs
"""

import copy
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VLMCacheStats:
    """Statistics for VLM cache performance."""
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    image_cache_hits: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "image_cache_hits": self.image_cache_hits,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


@dataclass
class VLMCacheEntry:
    """Entry in the VLM cache."""
    prompt_cache: List[Any]  # The KV cache state
    image_hash: str  # Hash of image content
    prompt_tokens: int  # Number of prompt tokens
    count: int = 1  # Reference count


def compute_image_hash(image_path: str) -> str:
    """
    Compute hash of image content for cache key.

    Args:
        image_path: Path to image file

    Returns:
        SHA256 hash of image content (first 16 chars)
    """
    try:
        path = Path(image_path)
        if path.exists():
            # Hash file content
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        else:
            # Hash the string itself (for URLs or base64)
            return hashlib.sha256(image_path.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Failed to hash image: {e}")
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]


def compute_images_hash(images: List[str]) -> str:
    """
    Compute combined hash for multiple images.

    Args:
        images: List of image paths/URLs

    Returns:
        Combined hash string
    """
    if not images:
        return "no_images"

    hashes = [compute_image_hash(img) for img in images]
    combined = "_".join(sorted(hashes))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class VLMCacheManager:
    """
    LRU Cache manager for VLM KV states.

    Caches KV states based on image content + prompt combination.
    When the same image+prompt is seen again, the cached KV state
    is reused to skip redundant computation.

    Example:
        >>> cache = VLMCacheManager(max_entries=50)
        >>> # First request - cache miss
        >>> kv_cache, _ = cache.fetch_cache(["image.jpg"], "Describe this")
        >>> # ... run generation ...
        >>> cache.store_cache(["image.jpg"], "Describe this", result_cache, num_tokens=100)
        >>> # Second request with same image - cache hit!
        >>> kv_cache, remaining = cache.fetch_cache(["image.jpg"], "Describe this")
    """

    def __init__(self, max_entries: int = 50):
        """
        Initialize VLM cache manager.

        Args:
            max_entries: Maximum number of cache entries (default: 50)
        """
        self.max_size = max_entries
        self._cache: OrderedDict[str, VLMCacheEntry] = OrderedDict()
        self.stats = VLMCacheStats()

    def _make_cache_key(self, images: List[str], prompt: str) -> str:
        """
        Create cache key from images and prompt.

        Args:
            images: List of image paths
            prompt: Text prompt

        Returns:
            Cache key string
        """
        image_hash = compute_images_hash(images)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"{image_hash}_{prompt_hash}"

    def fetch_cache(
        self,
        images: List[str],
        prompt: str
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Fetch cached KV state for image+prompt combination.

        Args:
            images: List of image paths
            prompt: Text prompt

        Returns:
            Tuple of (cache, hit) where:
            - cache: The KV cache if found, None otherwise
            - hit: True if cache hit, False if miss
        """
        self.stats.total_queries += 1

        cache_key = self._make_cache_key(images, prompt)

        if cache_key in self._cache:
            # Cache hit - move to end (most recently used)
            entry = self._cache.pop(cache_key)
            self._cache[cache_key] = entry
            entry.count += 1

            self.stats.hits += 1
            self.stats.tokens_saved += entry.prompt_tokens
            if images:
                self.stats.image_cache_hits += 1

            logger.debug(f"VLM cache hit: {cache_key[:20]}...")

            # Return deep copy to prevent modification
            return copy.deepcopy(entry.prompt_cache), True

        self.stats.misses += 1
        logger.debug(f"VLM cache miss: {cache_key[:20]}...")
        return None, False

    def store_cache(
        self,
        images: List[str],
        prompt: str,
        cache: List[Any],
        num_tokens: int = 0,
    ) -> None:
        """
        Store KV cache for future reuse.

        Args:
            images: List of image paths
            prompt: Text prompt
            cache: The KV cache state to store
            num_tokens: Number of tokens in the cached sequence
        """
        if not cache:
            return

        cache_key = self._make_cache_key(images, prompt)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            # Remove oldest (least recently used)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats.evictions += 1
            logger.debug(f"VLM cache evicted: {oldest_key[:20]}...")

        # Store new entry with deep copy
        image_hash = compute_images_hash(images)
        entry = VLMCacheEntry(
            prompt_cache=copy.deepcopy(cache),
            image_hash=image_hash,
            prompt_tokens=num_tokens,
        )
        self._cache[cache_key] = entry
        logger.debug(f"VLM cache stored: {cache_key[:20]}... ({num_tokens} tokens)")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = VLMCacheStats()

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._cache.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __repr__(self) -> str:
        return f"<VLMCacheManager entries={len(self)} max={self.max_size}>"
