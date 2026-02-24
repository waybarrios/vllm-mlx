# SPDX-License-Identifier: Apache-2.0
"""Cache utilities for speculative decoding KV cache trimming.

Provides batched cache operations for per-sequence variable trimming.
If upstream mlx-lm lacks ``trim_per_sequence``, a polyfill is injected
automatically so speculative decoding works on any mlx-lm >= 0.30.
"""

from __future__ import annotations

import logging
from typing import Any, List

import mlx.core as mx

logger = logging.getLogger(__name__)

_polyfill_applied = False


def _ensure_trim_per_sequence() -> None:
    """Add trim_per_sequence to BatchKVCache if the method is missing.

    Upstream mlx-lm may not have this method; we polyfill it so that
    speculative decoding can do per-sequence variable trimming on any
    mlx-lm version that has the offset/left_padding attributes.
    """
    global _polyfill_applied
    if _polyfill_applied:
        return
    _polyfill_applied = True

    try:
        from mlx_lm.models.cache import BatchKVCache
    except ImportError:
        return

    if hasattr(BatchKVCache, "trim_per_sequence"):
        return

    def trim_per_sequence(self, n: mx.array) -> None:
        """Trim each sequence by a different amount.

        Args:
            n: An array of shape ``(B,)`` with the number of tokens to
                trim from each sequence.
        """
        n = mx.minimum(n, self.left_padding + self.offset)
        self.offset -= n
        self._idx = int(mx.max(self.left_padding + self.offset).item())

    BatchKVCache.trim_per_sequence = trim_per_sequence
    logger.info("Polyfilled BatchKVCache.trim_per_sequence for spec decode")


# Apply polyfill at import time
_ensure_trim_per_sequence()


def _trim_layer(layer: Any, trim_amounts: mx.array) -> None:
    """Trim a single cache layer, recursing into CacheList."""
    try:
        from mlx_lm.models.cache import CacheList

        if isinstance(layer, CacheList):
            for sub in layer.caches:
                _trim_layer(sub, trim_amounts)
            return
    except ImportError:
        caches = getattr(layer, "caches", None)
        if isinstance(caches, (tuple, list)):
            for sub in caches:
                _trim_layer(sub, trim_amounts)
            return
    layer.trim_per_sequence(trim_amounts)


def batch_variable_trim(
    cache_layers: List[Any],
    trim_amounts: mx.array,
) -> None:
    """Per-sequence variable trim using trim_per_sequence().

    Iterates over each cache layer (one per transformer layer) and
    calls trim_per_sequence on each.  Handles ``CacheList`` layers
    by recursing into their sub-caches.

    Args:
        cache_layers: List of BatchKVCache (or CacheList) instances (one per layer).
        trim_amounts: [B] int32 array of positions to trim per sequence.
    """
    if int(trim_amounts.max()) == 0:
        return
    for cache_layer in cache_layers:
        _trim_layer(cache_layer, trim_amounts)


def can_per_seq_trim(cache_layers: List[Any]) -> bool:
    """Check if cache supports per-sequence trim.

    Returns False if ANY leaf cache lacks trim_per_sequence.
    """
    if not cache_layers:
        return False
    for layer in cache_layers:
        if not _layer_supports_per_seq_trim(layer):
            return False
    return True


def _layer_supports_per_seq_trim(layer: Any) -> bool:
    """Check a single cache layer (recurse into CacheList)."""
    try:
        from mlx_lm.models.cache import CacheList

        if isinstance(layer, CacheList):
            return all(_layer_supports_per_seq_trim(c) for c in layer.caches)
    except ImportError:
        caches = getattr(layer, "caches", None)
        if isinstance(caches, (tuple, list)):
            return all(_layer_supports_per_seq_trim(c) for c in caches)
    return hasattr(layer, "trim_per_sequence")


def fixup_cache_after_filter(cache_layers: List[Any]) -> None:
    """Recompute _idx for all cache layers after batch.filter().

    BatchKVCache.filter() does not recompute _idx to reflect the
    new max(offset + left_padding).  If the removed entry held the
    maximum position, _idx stays stale, causing the model to read
    garbage KV positions on subsequent forward passes.

    Also evaluates cache metadata to prevent lazy graph accumulation.

    Args:
        cache_layers: List of BatchKVCache (or CacheList) instances.
    """
    if not cache_layers:
        return
    for layer in cache_layers:
        _fixup_layer_idx(layer)
    # Materialize all metadata so the next forward starts clean
    _eval_cache_metadata(cache_layers)


def _fixup_layer_idx(layer: Any) -> None:
    """Recompute _idx for a single cache layer (recurse into CacheList)."""
    try:
        from mlx_lm.models.cache import CacheList
        if isinstance(layer, CacheList):
            for sub in layer.caches:
                _fixup_layer_idx(sub)
            return
    except ImportError:
        caches = getattr(layer, "caches", None)
        if isinstance(caches, (tuple, list)):
            for sub in caches:
                _fixup_layer_idx(sub)
            return
    # Recompute _idx
    if hasattr(layer, "offset") and hasattr(layer, "left_padding") and hasattr(layer, "_idx"):
        new_idx = int(mx.max(layer.left_padding + layer.offset).item())
        if new_idx != layer._idx:
            logger.debug(f"fixup_cache_after_filter: _idx {layer._idx} -> {new_idx}")
            layer._idx = new_idx


def _eval_cache_metadata(cache_layers: List[Any]) -> None:
    """Force-evaluate cache offset and left_padding to prevent lazy accumulation."""
    tensors = []
    for layer in cache_layers:
        _collect_cache_metadata(layer, tensors)
    if tensors:
        mx.eval(*tensors)


def _collect_cache_metadata(layer: Any, tensors: list) -> None:
    """Collect offset/left_padding tensors from a cache layer."""
    try:
        from mlx_lm.models.cache import CacheList
        if isinstance(layer, CacheList):
            for sub in layer.caches:
                _collect_cache_metadata(sub, tensors)
            return
    except ImportError:
        caches = getattr(layer, "caches", None)
        if isinstance(caches, (tuple, list)):
            for sub in caches:
                _collect_cache_metadata(sub, tensors)
            return
    if hasattr(layer, "offset"):
        tensors.append(layer.offset)
    if hasattr(layer, "left_padding"):
        tensors.append(layer.left_padding)
