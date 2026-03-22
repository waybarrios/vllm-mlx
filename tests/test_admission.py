# SPDX-License-Identifier: Apache-2.0
from vllm_mlx.admission import compute_kv_per_token


def test_kv_per_token_qwen35_35b():
    """Qwen3.5-35B: 10 attn layers, 2 KV heads, 256 head_dim, bfloat16."""
    result = compute_kv_per_token(
        num_hidden_layers=40,
        full_attention_interval=4,
        num_kv_heads=2,
        head_dim=256,
        dtype_bytes=2,
    )
    assert result == 20_480  # 10 * 2 * 256 * 2 * 2


def test_kv_per_token_qwen35_122b():
    """Qwen3.5-122B: 12 attn layers, 2 KV heads, 256 head_dim, bfloat16."""
    result = compute_kv_per_token(
        num_hidden_layers=48,
        full_attention_interval=4,
        num_kv_heads=2,
        head_dim=256,
        dtype_bytes=2,
    )
    assert result == 24_576  # 12 * 2 * 256 * 2 * 2


def test_kv_per_token_dense_model():
    """Dense model (no interval): all layers are attention."""
    result = compute_kv_per_token(
        num_hidden_layers=32,
        full_attention_interval=1,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    assert result == 32 * 8 * 128 * 2 * 2  # 131_072
