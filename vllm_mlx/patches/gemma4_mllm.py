# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-vlm's Gemma 4 attention to support BatchKVCache.

Gemma 4 Attention reads cache.offset into a local variable before calling
update_and_fetch, then uses the same variable later for RoPE on queries:

    offset = cache.offset          # reference to mx.array([22])
    keys = self.rope(keys, offset=offset)
    keys, values = cache.update_and_fetch(keys, values)
    #  ^^^ self.offset += 1  mutates the SAME mx.array in-place!
    queries = self.rope(queries, offset=offset)   # offset is now 23!

For KVCache, cache.offset is a Python int (immutable), so the local copy
is unaffected. For BatchKVCache, cache.offset is an mx.array and
mx.array.__iadd__ is *in-place*, so the local reference is silently
mutated by update_and_fetch, giving queries the wrong RoPE position.

This patch replaces Gemma4 Attention.__call__ with a version that
snapshots cache.offset as a defensive copy before any mutation can occur.
The mx.array copy preserves per-sequence offsets needed for correct RoPE
in continuous batching (unlike int conversion which would lose this info).
"""

import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _snapshot_cache_offset(cache):
    """Snapshot cache offset, making a defensive copy if it's an mx.array.

    BatchKVCache stores offset as mx.array (per-batch-item).
    mx.array.__iadd__ is in-place, so update_and_fetch mutates the original.
    We return a copy to preserve the pre-update value for RoPE on queries.
    """
    if cache is None:
        return 0
    off = cache.offset
    if isinstance(off, int):
        return off
    if isinstance(off, mx.array):
        return off + 0  # defensive copy — new array, same values
    return off


def patch_gemma4_attention_for_batching() -> bool:
    """Monkey-patch Gemma4 Attention.__call__ to snapshot offset before update
    AND to match the mlx-vlm 0.5.0+ signature (shared_kv / offset kwargs and a
    3-tuple return).

    Background — two distinct reasons this patch exists
    ---------------------------------------------------
    1. **BatchKVCache compatibility (the original reason for this patch).**
       BatchedEngine wraps the per-request KV cache in a ``BatchKVCache``
       whose ``offset`` is an ``mx.array`` (one slot per batched sequence)
       rather than a scalar int. The stock mlx-vlm Gemma 4 attention reads
       ``cache.offset`` after ``update_and_fetch`` has already advanced it
       in place, producing an off-by-one RoPE position for the just-
       appended tokens — see ``_snapshot_cache_offset`` for the fix.

    2. **mlx-vlm 0.5.0 signature change (added 2026-05-21).**
       Upstream Gemma 4 added ``shared_kv: Optional[tuple]`` and ``offset``
       kwargs, and ``DecoderLayer.__call__`` now unpacks a 3-tuple return
       ``(output, (keys, values), offset)`` from ``self.self_attn``. The
       previous version of this patch only accepted ``(x, mask, cache)``
       and returned a single ``mx.array``, which made every Gemma 4
       request under ``--continuous-batching`` crash with::

           TypeError: patch_gemma4_attention_for_batching.<locals>._patched_call()
           got an unexpected keyword argument 'shared_kv'

       This update mirrors the upstream signature/return shape so the
       DecoderLayer's tuple-unpack succeeds, while keeping the offset
       snapshot behavior in #1 above.

    Returns True if patch was applied, False if mlx-vlm is not installed
    or Gemma 4 module not available.
    """
    try:
        from mlx_vlm.models.gemma4.language import Attention as Gemma4Attention
        from mlx_vlm.models.base import scaled_dot_product_attention
    except ImportError:
        logger.debug("[Gemma4 patch] mlx-vlm Gemma4 module not available")
        return False

    if getattr(Gemma4Attention, "_batch_patched", False):
        logger.debug("[Gemma4 patch] Already patched")
        return True

    _orig_call = Gemma4Attention.__call__

    def _patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> Any:
        # Returns a 3-tuple to match mlx-vlm 0.5.0's
        #     h, shared_kv, offset = self.self_attn(...)
        # in DecoderLayer. Returning a single mx.array (as the old patch
        # did) breaks tuple-unpacking with TypeError.
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            # KV-shared layers (Gemma 4's first N decoder layers) reuse the
            # previous layer's k/v rather than computing their own. mlx-vlm
            # 0.5.0+ threads them through via the ``shared_kv`` kwarg from
            # the DecoderLayer, replacing the older
            # ``self.is_kv_shared_layer + cache.state`` lookup the previous
            # version of this patch used.
            keys, values = shared_kv
        else:
            # Snapshot offset BEFORE update_and_fetch can mutate it in-place
            # (the original reason for this patch — see module docstring).
            # If the caller supplied an offset, trust it; otherwise snapshot
            # from cache. This matches upstream's branch where the
            # DecoderLayer can override offset.
            if offset is None:
                offset = _snapshot_cache_offset(cache)

            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            # k_eq_v: values come from raw k_proj (pre-k_norm) — upstream
            # quirk preserved exactly.
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # 3-tuple return mirrors mlx-vlm 0.5.0 upstream. The middle slot is
        # the (keys, values) the next KV-shared layer will read back via
        # its own ``shared_kv`` kwarg.
        return self.o_proj(output), (keys, values), offset

    Gemma4Attention.__call__ = _patched_call
    Gemma4Attention._batch_patched = True
    logger.info("[Gemma4 patch] Attention patched for BatchKVCache support")
    return True
