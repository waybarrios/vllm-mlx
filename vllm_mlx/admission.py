# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware admission controller for multi-user inference.

Core principle: Load affects latency, never quality.
Once a request starts generating, it runs to completion.
The admission controller only decides WHEN to start.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


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

    Uses mx.get_active_memory() + mx.get_cache_memory() for true GPU
    memory usage, not just live tensors.
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
        return self._device_usable - mx.get_active_memory() - mx.get_cache_memory()

    def can_admit(self, prefill_bytes: int) -> bool:
        """Can we admit a request that needs prefill_bytes of KV cache?"""
        return self.free_memory() >= prefill_bytes + self.headroom_bytes


@dataclass
class QueuedRequest:
    request_id: str
    prompt_tokens: int
    enqueued_at: float = field(default_factory=time.time)


class RequestQueue:
    """Request queue with configurable ordering policy."""

    def __init__(self, policy: str = "fifo"):
        if policy != "fifo":
            raise ValueError(
                f"Unsupported policy: {policy!r}. Only 'fifo' is implemented."
            )
        self._policy = policy
        self._queue: deque[QueuedRequest] = deque()

    def enqueue(self, request_id: str, prompt_tokens: int) -> int:
        """Add request to queue. Returns position (0-indexed)."""
        entry = QueuedRequest(request_id=request_id, prompt_tokens=prompt_tokens)
        self._queue.append(entry)
        position = len(self._queue) - 1
        logger.info(
            f"[queue] {request_id} queued at position {position} ({prompt_tokens} tokens)"
        )
        return position

    def peek(self) -> Optional[QueuedRequest]:
        """Return next request without removing it."""
        if not self._queue:
            return None
        return self._queue[0]

    def dequeue(self) -> Optional[QueuedRequest]:
        """Remove and return next request per policy."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def cancel(self, request_id: str) -> bool:
        """Remove a request from the queue. Returns True if found."""
        for i, entry in enumerate(self._queue):
            if entry.request_id == request_id:
                del self._queue[i]
                return True
        return False

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


class AdmissionController:
    """Flow control for multi-user inference.

    Core principle: Load affects latency, never quality.
    Decides WHEN to start requests. Once started, a request
    runs to completion at full quality.
    """

    def __init__(
        self,
        kv_per_token: int,
        headroom_bytes: int = 8 * 1024**3,
        policy: str = "fifo",
    ):
        self.kv_per_token = kv_per_token
        self._monitor = MemoryMonitor(headroom_bytes=headroom_bytes)
        self._queue = RequestQueue(policy=policy)
        self._wait_events: dict[str, asyncio.Event] = {}
        self._eviction_callback: Optional[callable] = None

    def try_admit(
        self, request_id: str, prompt_tokens: int
    ) -> Tuple[bool, Optional[int]]:
        """Try to admit a request for immediate processing.

        Returns:
            (True, None) if admitted.
            (False, queue_position) if queued.
        """
        # FIFO: if anything is already waiting, join the queue regardless of memory.
        if not self._queue.is_empty():
            position = self._queue.enqueue(request_id, prompt_tokens)
            return False, position
        prefill_bytes = prompt_tokens * self.kv_per_token
        if self._monitor.can_admit(prefill_bytes):
            logger.info(
                f"[admit] {request_id} ADMITTED ({prompt_tokens} tokens, "
                f"{prefill_bytes / 1e6:.0f} MB prefill, "
                f"{self._monitor.free_memory() / 1e9:.1f} GB free)"
            )
            return True, None
        position = self._queue.enqueue(request_id, prompt_tokens)
        return False, position

    def set_eviction_callback(self, callback) -> None:
        """Set a callback to evict prefix cache entries under memory pressure.

        The callback should return True if it evicted something, False if
        there's nothing left to evict.
        """
        self._eviction_callback = callback

    def check_queue(self) -> List[QueuedRequest]:
        """Check if queued requests can now be admitted.

        Called after a request completes or memory is freed.
        If an eviction callback is registered and the front-of-queue request
        can't fit, the callback is invoked once to free prefix cache memory,
        then admission is re-checked. If the callback returns False (nothing
        left to evict), the loop stops.
        Returns list of newly-admittable requests.
        """
        ready = []
        while not self._queue.is_empty():
            entry = self._queue.peek()
            prefill_bytes = entry.prompt_tokens * self.kv_per_token
            if self._monitor.can_admit(prefill_bytes):
                ready.append(self._queue.dequeue())
                logger.info(
                    f"[admit] {entry.request_id} DEQUEUED → admitted "
                    f"(waited {time.time() - entry.enqueued_at:.1f}s)"
                )
            else:
                # FIFO: do not skip the front to admit smaller requests behind it.
                # Prevents starvation of large requests.
                # Try evicting prefix cache to make room.
                if self._eviction_callback is not None and self._eviction_callback():
                    continue  # Re-check after eviction freed memory
                break  # No eviction possible or nothing left to evict
        return ready

    async def wait_for_admission(self, request_id: str, prompt_tokens: int) -> None:
        """Block until this request can be admitted. Returns immediately if room."""
        admitted, position = self.try_admit(request_id, prompt_tokens)
        if admitted:
            return
        event = asyncio.Event()
        self._wait_events[request_id] = event
        logger.info(f"[admit] {request_id} waiting for admission (position {position})")
        try:
            await event.wait()
        except asyncio.CancelledError:
            self._queue.cancel(request_id)
            self._wait_events.pop(request_id, None)
            raise
        self._wait_events.pop(request_id, None)

    def on_request_complete(self) -> List[QueuedRequest]:
        """Called when a request completes. Drains queue and signals waiters."""
        ready = self.check_queue()
        for entry in ready:
            event = self._wait_events.get(entry.request_id)
            if event:
                event.set()
        return ready

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued request and wake up its waiter."""
        result = self._queue.cancel(request_id)
        event = self._wait_events.pop(request_id, None)
        if event:
            event.set()  # Wake up waiter so it can clean up
        return result

    @property
    def queue_length(self) -> int:
        return len(self._queue)
