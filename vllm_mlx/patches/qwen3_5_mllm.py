# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for mlx-vlm's Qwen3.5 attention to support BatchKVCache.

Qwen 3.6 artifacts use the mlx-vlm Qwen3.5 language module in this stack.
The attention patch therefore lives in the Qwen3.5 compatibility module while
serving Qwen 3.6 27B/35B/122B artifacts.

mlx-vlm's Qwen3_5Attention uses cache.offset directly for kv_seq_len
computation and mask slicing. BatchKVCache stores offset as mx.array
(per-batch-item), not int, causing:

    mask = mask[..., :kv_seq_len]
    ValueError: Slice indices must be integers or None.

This patch replaces Qwen3_5Attention.__call__ with a version that converts
cache.offset to int before using it for arithmetic/slicing, while leaving the
actual cache.offset untouched so update_and_fetch still works correctly with
per-batch offsets.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _cache_offset_to_int(cache) -> int:
    """Extract cache offset as int, handling BatchKVCache mx.array offset."""
    if cache is None:
        return 0
    off = cache.offset
    if isinstance(off, int):
        return off
    if isinstance(off, mx.array):
        return int(off.max().item()) if off.ndim > 0 else int(off.item())
    return int(off)


def _default_target_verify_linears(linears, x, target_verify: bool):
    return tuple(linear(x) for linear in linears)


def _default_target_verify_left_padded_attention(*args, **kwargs):
    return None


def _normalize_position_inputs(
    position_ids: Optional[mx.array],
    position_embeddings: Optional[tuple[mx.array, mx.array]],
    length: int,
) -> tuple[Optional[mx.array], Optional[tuple[mx.array, mx.array]]]:
    if position_ids is None or position_ids.shape[-1] == length:
        return position_ids, position_embeddings
    logger.debug(
        "[Qwen3.5 patch] Recomputing stale position_ids: got %s, expected %s",
        position_ids.shape[-1],
        length,
    )
    return None, None


def _position_ids_for_offset(offset: int, length: int) -> mx.array:
    position_ids = mx.arange(offset, offset + length)
    position_ids = mx.expand_dims(position_ids, axis=0)
    return mx.tile(position_ids, (3, 1, 1))


def _kv_seq_len(keys: mx.array, cache: Optional[Any], offset: int) -> int:
    length = keys.shape[-2]
    return length + offset + 1 if cache is not None else length


def _apply_rotary(
    attention,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    position_ids: mx.array,
    position_embeddings: Optional[tuple[mx.array, mx.array]],
    apply_multimodal_rotary_pos_emb,
) -> tuple[mx.array, mx.array]:
    if position_embeddings is not None:
        cos, sin = position_embeddings
        return apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

    if hasattr(attention.rotary_emb, "apply_rotary"):
        return attention.rotary_emb.apply_rotary(
            queries,
            keys,
            position_ids,
            unsqueeze_dim=1,
        )

    cos, sin = attention.rotary_emb(values, position_ids)
    return apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)


def _slice_attention_mask(
    mask: Optional[mx.array],
    cache: Optional[Any],
    kv_seq_len: int,
    length: int,
) -> Optional[mx.array]:
    if mask is None or not isinstance(mask, mx.array):
        return mask
    if cache is not None and hasattr(cache, "_idx") and hasattr(cache, "left_padding"):
        kv_seq_len = int(cache._idx) + length
    elif isinstance(kv_seq_len, mx.array):
        kv_seq_len = int(kv_seq_len.max().item())
    return mask[..., : int(kv_seq_len)]


def _maybe_target_verify_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    cache: Optional[Any],
    mask: Optional[mx.array],
    scale: float,
    target_verify: bool,
    length: int,
    left_padded_decode: bool,
    target_verify_left_padded_attention,
) -> Optional[mx.array]:
    if not ((target_verify and length > 1) or left_padded_decode):
        return None
    return target_verify_left_padded_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


def patch_qwen35_attention_for_batching() -> bool:
    """Monkey-patch Qwen3_5Attention.__call__ to handle BatchKVCache.

    Returns True if patch was applied, False if mlx-vlm is not installed
    or Qwen3.5 module not available.
    """
    try:
        qwen35_language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
        from mlx_lm.models.base import scaled_dot_product_attention
    except ImportError:
        logger.debug("[Qwen3.5 patch] mlx-vlm Qwen3.5 module not available")
        return False

    Qwen3_5Attention = qwen35_language.Qwen3_5Attention
    apply_multimodal_rotary_pos_emb = qwen35_language.apply_multimodal_rotary_pos_emb
    target_verify_linears = getattr(
        qwen35_language,
        "_target_verify_linears",
        _default_target_verify_linears,
    )
    target_verify_left_padded_attention = getattr(
        qwen35_language,
        "_target_verify_left_padded_attention",
        _default_target_verify_left_padded_attention,
    )

    if getattr(Qwen3_5Attention, "_batch_patched", False):
        logger.debug("[Qwen3.5 patch] Already patched")
        return True

    def _patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        target_verify: bool = False,
    ) -> mx.array:
        B, L, _D = x.shape

        q_proj_output, keys, values = target_verify_linears(
            (self.q_proj, self.k_proj, self.v_proj),
            x,
            target_verify,
        )
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        offset = _cache_offset_to_int(cache)
        position_ids, position_embeddings = _normalize_position_inputs(
            position_ids,
            position_embeddings,
            L,
        )

        if position_ids is None:
            position_ids = _position_ids_for_offset(offset, L)
        kv_seq_len = _kv_seq_len(keys, cache, offset)

        queries, keys = _apply_rotary(
            self,
            queries,
            keys,
            values,
            position_ids,
            position_embeddings,
            apply_multimodal_rotary_pos_emb,
        )

        mask = _slice_attention_mask(mask, cache, kv_seq_len, L)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        left_padded_decode = (
            mask == "left_padded_decode" if isinstance(mask, str) else False
        )
        if left_padded_decode:
            mask = None

        output = _maybe_target_verify_attention(
            queries,
            keys,
            values,
            cache=cache,
            mask=mask,
            scale=self.scale,
            target_verify=target_verify,
            length=L,
            left_padded_decode=left_padded_decode,
            target_verify_left_padded_attention=target_verify_left_padded_attention,
        )

        if output is None:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))

    Qwen3_5Attention.__call__ = _patched_call
    setattr(Qwen3_5Attention, "_batch_patched", True)
    logger.info("[Qwen3.5 patch] Attention patched for BatchKVCache support")
    return True
