# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-vlm's GLM-4.6V model to support BatchKVCache.

GLM-4.6V (glm4v_moe) computes position_ids from cache[0].offset once at
the start of GLM4VModel.__call__, then derives position_embeddings used
by ALL decoder layers:

    position_ids = mx.arange(cache[0].offset, cache[0].offset + seq_len)
    position_embeddings = self.rotary_emb(h, position_ids)
    for layer in self.layers:
        h = layer(h, mask, cache[i], position_embeddings)

For regular KVCache, cache.offset is a Python int, so mx.arange works fine.
For BatchKVCache, cache.offset is an mx.array (per-batch-item offsets), and
mx.arange does not support mx.array start/stop arguments, producing wrong
position_ids that corrupt RoPE embeddings for ALL layers.

This patch replaces GLM4VModel.__call__ with a version that converts
cache[0].offset to int before computing position_ids.
"""

import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def patch_glm4v_moe_for_batching() -> bool:
    """Monkey-patch GLM4VModel.__call__ to handle BatchKVCache offset.

    Returns True if patch was applied, False if mlx-vlm is not installed
    or GLM-4.6V module not available.
    """
    try:
        from mlx_vlm.models.glm4v_moe.language import (
            GLM4VModel,
            create_attention_mask,
        )
    except ImportError:
        logger.debug("[GLM4V patch] mlx-vlm glm4v_moe module not available")
        return False

    if getattr(GLM4VModel, "_batch_patched", False):
        logger.debug("[GLM4V patch] Already patched")
        return True

    def _patched_call(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds.astype(self.norm.weight.dtype)

        if position_ids is None:
            # BatchKVCache stores offset as mx.array — convert to int
            # so mx.arange gets scalar start/stop arguments.
            offset = cache[0].offset
            if isinstance(offset, mx.array):
                offset = int(offset.max().item())
            position_ids = mx.arange(offset, offset + h.shape[-2])
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        position_embeddings = self.rotary_emb(h, position_ids)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * self.num_layers

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], position_embeddings)

        return self.norm(h)

    GLM4VModel.__call__ = _patched_call
    GLM4VModel._batch_patched = True
    logger.info("[GLM4V patch] GLM4VModel patched for BatchKVCache support")
    return True
