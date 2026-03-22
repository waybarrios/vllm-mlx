# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware admission controller for multi-user inference.

Core principle: Load affects latency, never quality.
Once a request starts generating, it runs to completion.
The admission controller only decides WHEN to start.
"""


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
