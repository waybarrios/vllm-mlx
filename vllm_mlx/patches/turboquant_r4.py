# SPDX-License-Identifier: Apache-2.0
"""
Runtime patch for TurboQuant R4 — online Hadamard rotation before down_proj.

TurboQuant fuses three rotations into model weights offline:
    R1 — residual stream Hadamard, into embed/lm_head + all in/out projections
    R2 — head-space rotation (skipped for gated attention models like Qwen3.5/3.6)
    R4 — intermediate-axis Hadamard, into down_proj input side ONLY (offline);
         the matching online Hadamard on the intermediate activation must be
         applied at runtime, which is what this patch does.

For a SwiGLU MLP:
    intermediate = silu(gate_proj(x)) * up_proj(x)
    intermediate_rotated = mx.hadamard_transform(intermediate)   # R4 online
    out = down_proj(intermediate_rotated)                         # weights pre-rotated

This patch wraps Qwen3NextMLP.__call__ (shared expert) and SwitchGLU.__call__
(routed experts) to inject mx.hadamard_transform on the intermediate axis.

Activation only — does NOT touch weights. The patch is idempotent (sentinel
attribute prevents double-wrap) and is a no-op if mlx_lm is not importable.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_turboquant_r4_patched"


def patch_qwen3_next_mlp_r4() -> bool:
    """Patch Qwen3NextMLP.__call__ to apply R4 Hadamard before down_proj.

    Idempotent. Returns True if patched (or already patched).
    """
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP
    except ImportError as e:
        logger.warning("[TurboQuant R4] Qwen3NextMLP not importable: %s", e)
        return False

    if getattr(Qwen3NextMLP, _PATCHED_ATTR, False):
        return True

    from mlx_lm.models.qwen3_next import swiglu as _swiglu

    def _patched_call(self, x: mx.array) -> mx.array:
        intermediate = _swiglu(self.gate_proj(x), self.up_proj(x))
        # Force contiguous + fp32 to avoid lazy-eval / strided issues with Hadamard.
        intermediate = mx.contiguous(intermediate.astype(mx.float32))
        intermediate = mx.hadamard_transform(intermediate)
        return self.down_proj(intermediate.astype(x.dtype))

    Qwen3NextMLP.__call__ = _patched_call
    setattr(Qwen3NextMLP, _PATCHED_ATTR, True)
    logger.info(
        "[TurboQuant R4] Qwen3NextMLP patched (online Hadamard before down_proj)"
    )
    return True


def patch_switch_glu_r4() -> bool:
    """Patch SwitchGLU.__call__ (routed MoE experts) to apply R4 Hadamard
    on the intermediate before down_proj.

    SwitchGLU's standard forward:
        x_up = self.up_proj(x, idx)
        x_gate = self.gate_proj(x, idx)
        x = self.down_proj(self.activation(x_up, x_gate), idx)

    Patched: insert mx.hadamard_transform between activation and down_proj.
    """
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
    except ImportError as e:
        logger.warning("[TurboQuant R4] SwitchGLU not importable: %s", e)
        return False

    if getattr(SwitchGLU, _PATCHED_ATTR, False):
        return True

    def _patched_call(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        intermediate = self.activation(x_up, x_gate)
        # Force contiguous + fp32 for Hadamard correctness on strided tensors.
        intermediate = mx.contiguous(intermediate.astype(mx.float32))
        intermediate = mx.hadamard_transform(intermediate)
        intermediate = intermediate.astype(x.dtype)
        x = self.down_proj(intermediate, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    SwitchGLU.__call__ = _patched_call
    setattr(SwitchGLU, _PATCHED_ATTR, True)
    logger.info("[TurboQuant R4] SwitchGLU patched (online Hadamard on intermediate)")
    return True


def install_turboquant_r4(scope: str = "all") -> None:
    """Install R4 patches. Safe to call multiple times.

    scope:
      "all"     — patch both Qwen3NextMLP (shared expert) and SwitchGLU (routed)
      "shared"  — patch only Qwen3NextMLP
      "experts" — patch only SwitchGLU
    """
    if scope in ("all", "shared"):
        patch_qwen3_next_mlp_r4()
    if scope in ("all", "experts"):
        patch_switch_glu_r4()


def is_turboquant_model(config: dict) -> bool:
    """Detect TurboQuant model from its quantization config."""
    q = config.get("quantization") or config.get("quantization_config") or {}
    return q.get("type") == "turboquant_v1"
