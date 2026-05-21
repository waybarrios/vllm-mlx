# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch: store the GatedDeltaNet recurrent state in a 16-bit dtype.

Qwen3.5/3.6 hybrid models alternate full-attention layers with GatedDeltaNet
linear-attention layers. The GatedDeltaNet recurrent matrix state — kept in
``cache[1]`` of an ``ArraysCache`` — is hardcoded float32 in mlx-lm
(``gated_delta.py`` allocates ``mx.zeros(..., dtype=mx.float32)``). For a 64-layer
Qwen3.6-27B that is 48 linear layers x ~3.0 MB/seq = ~144 MB of float32 state
per sequence, the dominant per-sequence live-memory cost.

This patch stores that recurrent matrix in a 16-bit dtype *between* decode
steps, halving the persistent footprint, while upcasting to float32 right
before each forward so kernel/ops compute precision is unchanged:

    cache[1] (16-bit)  --upcast-->  float32  --forward-->  float32  --downcast-->  16-bit

The Metal kernel already templates on ``state.dtype`` and accumulates in a
float32 register array, so feeding it float32 keeps compute bit-identical to
the unpatched path. The pure-ops fallback also receives float32, so it stays
fp32-safe too. Only the *stored* tensor is 16-bit.

Two 16-bit dtypes are supported (both 2 bytes, same 50% memory saving):
  - ``bfloat16`` — 7-bit mantissa, full float32 exponent range
  - ``float16``  — 10-bit mantissa (8x finer), narrower exponent range

float16's finer mantissa typically tracks the float32 recurrent state much
more closely; bfloat16 is the conservative default for exponent range. Select
via the ``VLLM_MLX_LINEAR_STATE_DTYPE`` env var (``bf16`` | ``fp16``) or the
``dtype`` argument. float32 was originally chosen for SSM numerical stability;
run the A/B quality check before enabling in production. The patch is opt-in
and per-process: production processes that never call the patch function are
byte-for-byte unaffected.

The conv state (``cache[0]``) is left untouched — it already follows
``inputs.dtype`` (bfloat16) and is small (~120 KB/layer).
"""

import logging
import os
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_SENTINEL = "_linear_state_bf16_patched"

_DTYPE_MAP = {
    "bf16": mx.bfloat16,
    "bfloat16": mx.bfloat16,
    "fp16": mx.float16,
    "float16": mx.float16,
}


def _resolve_dtype(dtype: Optional[Any]) -> Any:
    """Resolve the storage dtype from an explicit arg or the env var."""
    if dtype is not None:
        return dtype
    env = os.environ.get("VLLM_MLX_LINEAR_STATE_DTYPE", "bf16").lower()
    return _DTYPE_MAP.get(env, mx.bfloat16)


def _wrap_gated_delta_call(cls: Any, dtype: Any = mx.bfloat16) -> bool:
    """Wrap a GatedDeltaNet ``__call__`` to keep cache[1] in ``dtype``.

    Idempotent: a class already patched is left untouched.
    """
    if getattr(cls, _SENTINEL, False):
        return True

    _orig_call = cls.__call__

    def _patched_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Any:
        # Upcast the stored recurrent state to float32 before the forward so
        # kernel/ops compute precision is unchanged.
        if cache is not None and cache[1] is not None and cache[1].dtype == dtype:
            cache[1] = cache[1].astype(mx.float32)

        out = _orig_call(self, inputs, mask, cache)

        # Downcast the updated recurrent state back to the 16-bit storage dtype.
        if cache is not None and cache[1] is not None and cache[1].dtype != dtype:
            cache[1] = cache[1].astype(dtype)

        return out

    cls.__call__ = _patched_call
    setattr(cls, _SENTINEL, True)
    return True


def patch_gated_delta_bf16_state(dtype: Optional[Any] = None) -> bool:
    """Patch mlx-lm and mlx-vlm GatedDeltaNet to store cache[1] in 16-bit.

    Args:
        dtype: storage dtype (``mx.bfloat16`` or ``mx.float16``). If None,
            resolved from the ``VLLM_MLX_LINEAR_STATE_DTYPE`` env var
            (default ``bf16``).

    Returns True if at least one GatedDeltaNet class was patched, False if
    neither mlx-lm nor mlx-vlm Qwen3.5 modules are importable.
    """
    dtype = _resolve_dtype(dtype)
    dtype_name = "float16" if dtype == mx.float16 else "bfloat16"
    patched_any = False

    try:
        from mlx_lm.models.qwen3_5 import GatedDeltaNet as _LmGatedDeltaNet

        if _wrap_gated_delta_call(_LmGatedDeltaNet, dtype):
            patched_any = True
            logger.info(
                "[linear-state-16bit] mlx_lm GatedDeltaNet patched (%s)", dtype_name
            )
    except ImportError:
        logger.debug("[linear-state-16bit] mlx_lm qwen3_5 not available")

    try:
        from mlx_vlm.models.qwen3_5.language import (
            Qwen3_5GatedDeltaNet as _VlmGatedDeltaNet,
        )

        if _wrap_gated_delta_call(_VlmGatedDeltaNet, dtype):
            patched_any = True
            logger.info(
                "[linear-state-16bit] mlx_vlm Qwen3_5GatedDeltaNet patched (%s)",
                dtype_name,
            )
    except ImportError:
        logger.debug("[linear-state-16bit] mlx_vlm qwen3_5 not available")

    if not patched_any:
        logger.warning(
            "[linear-state-16bit] no GatedDeltaNet class found; patch is a no-op"
        )
    return patched_any
