# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-vlm Gemma 4 Attention to trim oversized masks.

mlx-vlm 0.5.0's stock Gemma 4 attention assumes the mask's last dim matches
keys.shape[-2] exactly. vllm-mlx's BatchedEngine MLLM path (continuous
batching) sometimes passes a mask sized for the max sequence in the batch
while a specific layer's keys end up shorter — sliding-window layers cap
keys at window=512, the mask is built once for the full prompt. Without a
trim, scaled_dot_product_attention sees a shape mismatch.

That mask trim is the only behavior this patch adds; everything else
mirrors mlx-vlm 0.5.0 verbatim (signature, return shape, offset handling).
The previous reason for this patch — BatchKVCache's in-place `+=` on
`cache.offset` corrupting RoPE — is now handled upstream: mlx-vlm 0.5.0
line 223 does `offset = mx.array(cache.offset) if cache is not None else 0`
which is a defensive copy. (Confirmed in review of PR #564.)
"""

import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def patch_gemma4_attention_for_batching() -> bool:
    """Patch Gemma 4 Attention.__call__ to trim oversized masks.

    Otherwise mirrors mlx-vlm 0.5.0 upstream verbatim. Returns True if
    applied, False if mlx-vlm is not installed or Gemma 4 unavailable.
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

    def _patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> Any:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            offset = mx.array(cache.offset) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        # Only addition vs upstream: trim mask if longer than keys.
        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2] :]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset

    Gemma4Attention.__call__ = _patched_call
    Gemma4Attention._batch_patched = True
    logger.info("[Gemma4 patch] Attention patched (mask trim for BatchedEngine)")
    return True
