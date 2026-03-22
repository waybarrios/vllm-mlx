# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from vllm_mlx.admission import MemoryMonitor, RequestQueue, compute_kv_per_token


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


def test_memory_monitor_free_memory():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 80 * 1024**3
        mock_mx.get_cache_memory.return_value = 10 * 1024**3
        monitor = MemoryMonitor()
        free = monitor.free_memory()
        assert free == 30 * 1024**3  # 120 - 80 - 10


def test_memory_monitor_can_admit():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 95 * 1024**3
        mock_mx.get_cache_memory.return_value = 5 * 1024**3
        monitor = MemoryMonitor(headroom_bytes=8 * 1024**3)
        # 20 GB free, 8 GB headroom, 5 GB prefill = 20 >= 13 → admit
        assert monitor.can_admit(prefill_bytes=5 * 1024**3) is True
        # 20 GB free, 8 GB headroom, 15 GB prefill = 20 < 23 → reject
        assert monitor.can_admit(prefill_bytes=15 * 1024**3) is False


def test_request_queue_fifo():
    q = RequestQueue(policy="fifo")
    q.enqueue("req-1", prompt_tokens=1000)
    q.enqueue("req-2", prompt_tokens=500)
    q.enqueue("req-3", prompt_tokens=2000)
    # FIFO: order of insertion
    assert q.peek().request_id == "req-1"
    assert q.dequeue().request_id == "req-1"
    assert q.dequeue().request_id == "req-2"
    assert q.dequeue().request_id == "req-3"
    assert q.is_empty()


def test_request_queue_length():
    q = RequestQueue(policy="fifo")
    assert len(q) == 0
    q.enqueue("req-1", prompt_tokens=100)
    assert len(q) == 1
    q.dequeue()
    assert len(q) == 0


def test_request_queue_cancel():
    q = RequestQueue(policy="fifo")
    q.enqueue("req-1", prompt_tokens=100)
    q.enqueue("req-2", prompt_tokens=200)
    q.cancel("req-1")
    assert q.dequeue().request_id == "req-2"
