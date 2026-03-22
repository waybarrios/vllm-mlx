# SPDX-License-Identifier: Apache-2.0
import asyncio

from unittest.mock import patch

from vllm_mlx.admission import (
    AdmissionController,
    MemoryMonitor,
    RequestQueue,
    compute_kv_per_token,
)


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


def test_admission_controller_admit_when_room():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 50 * 1024**3  # 70 GB free
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480,
            headroom_bytes=8 * 1024**3,
        )
        admitted, position = controller.try_admit("req-1", prompt_tokens=10_000)
        assert admitted is True
        assert position is None


def test_admission_controller_queue_when_full():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3  # 5 GB free
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480,
            headroom_bytes=8 * 1024**3,
        )
        # 10K tokens * 20KB = 200MB prefill. 5GB free < 200MB + 8GB → queue
        admitted, position = controller.try_admit("req-1", prompt_tokens=10_000)
        assert admitted is False
        assert position == 0


def test_admission_controller_drain_queue():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        # Start full
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480,
            headroom_bytes=8 * 1024**3,
        )
        controller.try_admit("req-1", prompt_tokens=1000)
        assert controller.queue_length == 1
        # Memory freed
        mock_mx.get_active_memory.return_value = 50 * 1024**3
        ready = controller.check_queue()
        assert len(ready) == 1
        assert ready[0].request_id == "req-1"
        assert controller.queue_length == 0


def test_admission_controller_fifo_no_bypass():
    """Small requests must queue behind large ones even if they'd fit."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 112 * 1024**3  # 8 GB free
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480,
            headroom_bytes=8 * 1024**3,
        )
        # Large request can't fit (~3.8GB prefill + 8GB headroom > 8GB free)
        admitted1, pos1 = controller.try_admit("req-large", prompt_tokens=200_000)
        assert admitted1 is False
        assert pos1 == 0
        # Small request WOULD fit (20KB + 8GB < 8GB is false, but even if
        # memory freed later, FIFO requires it to queue behind the large one)
        admitted2, pos2 = controller.try_admit("req-small", prompt_tokens=1)
        assert admitted2 is False
        assert pos2 == 1


def test_admission_controller_cancel():
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480,
            headroom_bytes=8 * 1024**3,
        )
        controller.try_admit("req-1", prompt_tokens=1000)
        assert controller.queue_length == 1
        assert controller.cancel("req-1") is True
        assert controller.queue_length == 0
        assert controller.cancel("nonexistent") is False


def test_admission_controller_wait_for_admission():
    """Test async wait — admitted immediately when room."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 50 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )
        # Should return immediately — plenty of memory
        asyncio.get_event_loop().run_until_complete(
            controller.wait_for_admission("req-1", prompt_tokens=100)
        )
        assert controller.queue_length == 0


def test_admission_controller_on_request_complete():
    """Test that completing a request releases queued ones."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )

        # Queue a request (not enough memory)
        admitted, pos = controller.try_admit("req-1", prompt_tokens=1000)
        assert admitted is False

        # Set up wait event manually
        event = asyncio.Event()
        controller._wait_events["req-1"] = event

        # Free memory and signal completion
        mock_mx.get_active_memory.return_value = 50 * 1024**3
        ready = controller.on_request_complete()
        assert len(ready) == 1
        assert ready[0].request_id == "req-1"
        assert event.is_set()  # Waiter was signaled


def test_admission_controller_cancel_wakes_waiter():
    """Test that cancelling a queued request wakes up its waiter."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )

        # Queue a request
        controller.try_admit("req-1", prompt_tokens=1000)

        # Set up wait event
        event = asyncio.Event()
        controller._wait_events["req-1"] = event

        # Cancel should wake the waiter
        assert controller.cancel("req-1") is True
        assert event.is_set()
        assert "req-1" not in controller._wait_events


def test_admission_wait_cleans_up_on_cancellation():
    """Verify CancelledError during wait cleans up queue and events."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )

        async def simulate_cancel():
            task = asyncio.create_task(
                controller.wait_for_admission("req-cancel", prompt_tokens=1000)
            )
            await asyncio.sleep(0)  # Let it reach the await event.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Queue and events should be cleaned up
            assert controller.queue_length == 0
            assert "req-cancel" not in controller._wait_events

        asyncio.get_event_loop().run_until_complete(simulate_cancel())


def test_admission_controller_eviction_callback():
    """When queue is non-empty and memory tight, evict prefix cache."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )

        eviction_count = []

        def mock_evict():
            eviction_count.append(1)
            # Simulate freeing memory by reducing active
            mock_mx.get_active_memory.return_value = 50 * 1024**3
            return True  # True = evicted something

        controller.set_eviction_callback(mock_evict)

        # Queue a request (not enough memory)
        controller.try_admit("req-1", prompt_tokens=1000)
        assert controller.queue_length == 1

        # check_queue should call eviction, then admit
        ready = controller.check_queue()
        assert len(eviction_count) > 0  # Eviction was called
        assert len(ready) == 1  # Request was admitted after eviction
        assert ready[0].request_id == "req-1"


def test_admission_controller_eviction_no_callback():
    """Without eviction callback, check_queue just waits."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )
        controller.try_admit("req-1", prompt_tokens=1000)
        ready = controller.check_queue()
        assert len(ready) == 0  # Can't admit, no eviction callback


def test_admission_controller_eviction_exhausted():
    """If eviction can't free enough memory, request stays queued."""
    with patch("vllm_mlx.admission.mx") as mock_mx:
        mock_mx.metal.is_available.return_value = True
        mock_mx.device_info.return_value = {
            "max_recommended_working_set_size": 120 * 1024**3
        }
        mock_mx.get_active_memory.return_value = 115 * 1024**3
        mock_mx.get_cache_memory.return_value = 0
        controller = AdmissionController(
            kv_per_token=20_480, headroom_bytes=8 * 1024**3
        )

        def mock_evict_nothing():
            return False  # Nothing to evict

        controller.set_eviction_callback(mock_evict_nothing)
        controller.try_admit("req-1", prompt_tokens=1000)
        ready = controller.check_queue()
        assert len(ready) == 0  # Still can't admit
        assert controller.queue_length == 1
