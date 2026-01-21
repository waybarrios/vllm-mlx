# SPDX-License-Identifier: Apache-2.0
"""
BatchMambaCache implementation for continuous batching with Mamba models.

mlx-lm's BatchGenerator requires cache objects to have an `extract` method,
but MambaCache (which extends ArraysCache) doesn't have one. This module
provides a BatchMambaCache wrapper that adds batching support.
"""

import logging
from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.cache import MambaCache

logger = logging.getLogger(__name__)


class BatchMambaCache(MambaCache):
    """
    Batch-aware MambaCache for continuous batching.

    This extends MambaCache to support batch operations required by
    mlx-lm's BatchGenerator, specifically the `extract` method.
    """

    def __init__(self, left_padding: Optional[List[int]] = None):
        """
        Initialize BatchMambaCache.

        Args:
            left_padding: Amount of left padding for each sequence in batch
        """
        super().__init__(left_padding=left_padding)
        self._batch_size = len(left_padding) if left_padding else 0

    def extract(self, idx: int) -> MambaCache:
        """
        Extract a single cache from the batch.

        Args:
            idx: Index of the sequence to extract

        Returns:
            A new MambaCache with the extracted state
        """
        cache = MambaCache()
        # Extract the state arrays for this index
        cache.cache = [
            mx.contiguous(c[idx : idx + 1]) if c is not None else None
            for c in self.cache
        ]
        cache.left_padding = None  # Single sequence, no batch padding
        return cache

    @classmethod
    def merge(cls, caches: List[MambaCache]) -> "BatchMambaCache":
        """
        Merge multiple MambaCache objects into a BatchMambaCache.

        Args:
            caches: List of MambaCache objects to merge

        Returns:
            A new BatchMambaCache containing all caches
        """
        if not caches:
            return cls([])

        # Get the structure from the first cache
        batch_size = len(caches)

        # MambaCache stores 2 arrays (size=2 in ArraysCache.__init__)
        merged_cache = cls([0] * batch_size)

        # Merge each array in the cache
        num_arrays = len(caches[0].cache)
        merged_cache.cache = []

        for i in range(num_arrays):
            arrays = [c.cache[i] for c in caches if c.cache[i] is not None]
            if arrays:
                merged_cache.cache.append(mx.concatenate(arrays, axis=0))
            else:
                merged_cache.cache.append(None)

        return merged_cache


def patch_mlx_lm_for_mamba():
    """
    Patch mlx-lm to support MambaCache in BatchGenerator.

    This modifies the _make_cache function to handle MambaCache by
    converting it to BatchMambaCache.
    """
    import importlib

    gen_module = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import (
        KVCache,
        ArraysCache,
        RotatingKVCache,
        CacheList,
        MambaCache as OrigMambaCache,
    )
    from mlx_lm.generate import BatchKVCache, BatchRotatingKVCache

    # Store original function
    _original_make_cache = gen_module._make_cache

    def _patched_make_cache(model, left_padding):
        """
        Convert a list of regular caches into their corresponding
        batch-aware caches, with support for MambaCache.
        """

        def to_batch_cache(c):
            if isinstance(c, KVCache):
                return BatchKVCache(left_padding)
            elif isinstance(c, OrigMambaCache):
                # Handle MambaCache -> BatchMambaCache
                return BatchMambaCache(left_padding)
            elif isinstance(c, ArraysCache):
                c.left_padding = mx.array(left_padding)
                return c
            elif isinstance(c, RotatingKVCache):
                if c.keep > 0:
                    raise ValueError(
                        "RotatingKVCache with keep tokens is not supported."
                    )
                return BatchRotatingKVCache(c.max_size, left_padding)
            elif isinstance(c, CacheList):
                return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
            else:
                raise ValueError(f"{type(c)} does not yet support batching")

        if hasattr(model, "make_cache"):
            cache = model.make_cache()
            return [to_batch_cache(c) for c in cache]
        else:
            return [BatchKVCache(left_padding) for _ in model.layers]

    # Patch the module
    gen_module._make_cache = _patched_make_cache

    # Also patch _merge_caches to handle BatchMambaCache
    _original_merge_caches = gen_module._merge_caches

    def _patched_merge_caches(caches):
        """Merge caches with MambaCache support."""
        batch_cache = []
        for i in range(len(caches[0])):
            cache = None
            if isinstance(caches[0][i], KVCache):
                cache = BatchKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], RotatingKVCache):
                cache = BatchRotatingKVCache.merge([c[i] for c in caches])
            elif isinstance(caches[0][i], (OrigMambaCache, BatchMambaCache)):
                cache = BatchMambaCache.merge([c[i] for c in caches])
            else:
                raise ValueError(
                    f"{type(caches[0][i])} does not yet support batching with history"
                )
            batch_cache.append(cache)
        return batch_cache

    gen_module._merge_caches = _patched_merge_caches

    logger.info("Patched mlx-lm for MambaCache batching support")


# Auto-patch when module is imported
_patched = False


def ensure_mamba_support():
    """Ensure MambaCache batching support is enabled."""
    global _patched
    if not _patched:
        patch_mlx_lm_for_mamba()
        _patched = True
