# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware admission controller for multi-user inference.

Core principle: Load affects latency, never quality.
Once a request starts generating, it runs to completion.
The admission controller only decides WHEN to start.
"""

import mlx.core as mx


def compute_kv_per_token(
    num_hidden_layers: int,
    full_attention_interval: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Compute KV cache bytes per token for this model.

    Only full attention layers contribute to KV cache.
    Linear attention (GatedDeltaNet) layers use fixed-size
    SSM state regardless of context length.

    Args:
        num_hidden_layers: Total layer count.
        full_attention_interval: Every Nth layer is full attention.
            1 = all attention (dense), 4 = 25% attention (Qwen3.5).
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per head.
        dtype_bytes: Bytes per element (2 for bfloat16, 1 for int8 quantized).

    Returns:
        Bytes of KV cache consumed per token.
    """
    if full_attention_interval <= 0:
        full_attention_interval = 1
    attention_layers = num_hidden_layers // full_attention_interval
    return attention_layers * num_kv_heads * head_dim * 2 * dtype_bytes  # 2 = K + V


class MemoryMonitor:
    """Reads actual Metal memory to make admission decisions.

    Uses mx.get_active_memory() — real GPU allocations, not estimates.
    """

    def __init__(self, headroom_bytes: int = 8 * 1024**3):
        self.headroom_bytes = headroom_bytes
        if mx.metal.is_available():
            info = mx.device_info()
            self._device_usable = info["max_recommended_working_set_size"]
        else:
            self._device_usable = 0

    def free_memory(self) -> int:
        """Return bytes of free GPU-usable memory."""
        if not mx.metal.is_available():
            return 0
        return self._device_usable - mx.get_active_memory()

    def can_admit(self, prefill_bytes: int) -> bool:
        """Can we admit a request that needs prefill_bytes of KV cache?"""
        return self.free_memory() >= prefill_bytes + self.headroom_bytes
