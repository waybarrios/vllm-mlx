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
    hybrid_override_pattern: Optional[str] = None,
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
        hybrid_override_pattern: Layer-type string from config.json (e.g.
            Nemotron-H "MEMEMEM*E..."). '*' = attention, 'M' = Mamba,
            'E' = MoE, '-' = MLP.  When provided, attention layers are
            counted directly from the pattern instead of using
            full_attention_interval.

    Returns:
        Bytes of KV cache consumed per token.
    """
    if hybrid_override_pattern:
        attention_layers = hybrid_override_pattern.count("*")
    else:
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
        return max(
            0, self._device_usable - mx.get_active_memory() - mx.get_cache_memory()
        )

    def can_admit(self, prefill_bytes: int) -> bool:
        """Can we admit a request that needs prefill_bytes of KV cache?"""
        return self.free_memory() >= prefill_bytes + self.headroom_bytes


_VALID_POLICIES = ("fifo", "shortest_first", "priority")


@dataclass
class QueuedRequest:
    request_id: str
    prompt_tokens: int
    priority: int = 0  # Higher = more important (only used by "priority" policy)
    enqueued_at: float = field(default_factory=time.time)


class RequestQueue:
    """Request queue with configurable ordering policy.

    Policies:
        fifo: First-in-first-out. Fair, prevents starvation. Head-of-line
            blocking: a large request at the front blocks smaller ones behind it.
        shortest_first: Dequeue the request with the fewest prompt tokens.
            Maximizes throughput by clearing short requests first. Starvation
            guard: requests waiting longer than ``starvation_timeout_s`` are
            promoted to highest priority.
        priority: Dequeue the request with the highest ``priority`` value.
            Same-priority requests are ordered FIFO (by enqueue time).
    """

    def __init__(self, policy: str = "fifo", starvation_timeout_s: float = 120.0):
        if policy not in _VALID_POLICIES:
            raise ValueError(
                f"Unsupported policy: {policy!r}. "
                f"Valid policies: {', '.join(_VALID_POLICIES)}"
            )
        self._policy = policy
        self._starvation_timeout_s = starvation_timeout_s
        self._queue: deque[QueuedRequest] = deque()

    @property
    def policy(self) -> str:
        return self._policy

    def enqueue(self, request_id: str, prompt_tokens: int, priority: int = 0) -> int:
        """Add request to queue. Returns position (0-indexed)."""
        entry = QueuedRequest(
            request_id=request_id, prompt_tokens=prompt_tokens, priority=priority
        )
        self._queue.append(entry)
        position = len(self._queue) - 1
        logger.info(
            f"[queue] {request_id} queued at position {position} "
            f"({prompt_tokens} tokens, priority={priority})"
        )
        return position

    def peek(self) -> Optional[QueuedRequest]:
        """Return the next request that would be dequeued, without removing it."""
        if not self._queue:
            return None
        if self._policy == "fifo":
            return self._queue[0]
        return self._select_next()

    def dequeue(self) -> Optional[QueuedRequest]:
        """Remove and return next request per policy."""
        if not self._queue:
            return None
        if self._policy == "fifo":
            return self._queue.popleft()
        entry = self._select_next()
        if entry is not None:
            self._queue.remove(entry)
        return entry

    def _select_next(self) -> Optional[QueuedRequest]:
        """Select the next request to dequeue based on policy.

        For shortest_first: pick the request with fewest prompt_tokens.
        Starvation guard: any request waiting longer than starvation_timeout_s
        is treated as having 0 tokens (highest priority to dequeue).

        For priority: pick the request with highest priority value.
        Ties broken by earliest enqueue time (FIFO within same priority).
        """
        if not self._queue:
            return None

        now = time.time()

        if self._policy == "shortest_first":
            best = None
            for entry in self._queue:
                waited = now - entry.enqueued_at
                starved = waited >= self._starvation_timeout_s
                # Starved requests get effective size 0 (dequeue first)
                effective_tokens = 0 if starved else entry.prompt_tokens
                if best is None:
                    best = (entry, effective_tokens, entry.enqueued_at)
                else:
                    _, best_tokens, best_time = best
                    # Prefer fewer tokens; break ties by earlier enqueue
                    if effective_tokens < best_tokens or (
                        effective_tokens == best_tokens
                        and entry.enqueued_at < best_time
                    ):
                        best = (entry, effective_tokens, entry.enqueued_at)
            return best[0] if best else None

        elif self._policy == "priority":
            best = None
            for entry in self._queue:
                if best is None:
                    best = entry
                elif entry.priority > best.priority:
                    best = entry
                elif (
                    entry.priority == best.priority
                    and entry.enqueued_at < best.enqueued_at
                ):
                    best = entry
            return best

        # Fallback (shouldn't reach here — FIFO handled in dequeue)
        return self._queue[0]

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
        self, request_id: str, prompt_tokens: int, priority: int = 0
    ) -> Tuple[bool, Optional[int]]:
        """Try to admit a request for immediate processing.

        Returns:
            (True, None) if admitted.
            (False, queue_position) if queued.
        """
        # FIFO: if anything is already waiting, join the queue regardless of memory.
        # (shortest_first and priority also respect queue — no cutting the line)
        if not self._queue.is_empty():
            position = self._queue.enqueue(request_id, prompt_tokens, priority)
            return False, position
        prefill_bytes = prompt_tokens * self.kv_per_token
        if self._monitor.can_admit(prefill_bytes):
            logger.info(
                f"[admit] {request_id} ADMITTED ({prompt_tokens} tokens, "
                f"{prefill_bytes / 1e6:.0f} MB prefill, "
                f"{self._monitor.free_memory() / 1e9:.1f} GB free)"
            )
            return True, None
        position = self._queue.enqueue(request_id, prompt_tokens, priority)
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

        Policy behavior:
            fifo: Only admit the front of the queue. Never skips.
            shortest_first: Admit the smallest request that fits, even if
                it's not at the front. Starved requests (past timeout) are
                treated as smallest.
            priority: Admit the highest-priority request that fits. Ties
                broken by FIFO order.

        If an eviction callback is registered and the selected request can't
        fit, the callback is invoked to free prefix cache memory.

        Returns list of newly-admittable requests.
        """
        ready = []

        if self._queue.policy == "fifo":
            # FIFO: strict head-of-line. Never skip the front.
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
                    if (
                        self._eviction_callback is not None
                        and self._eviction_callback()
                    ):
                        continue
                    break
        else:
            # shortest_first / priority: scan for best admittable request
            while not self._queue.is_empty():
                candidate = self._queue.peek()  # Policy-ordered best
                prefill_bytes = candidate.prompt_tokens * self.kv_per_token
                if self._monitor.can_admit(prefill_bytes):
                    self._queue.dequeue()
                    ready.append(candidate)
                    logger.info(
                        f"[admit] {candidate.request_id} DEQUEUED → admitted "
                        f"(waited {time.time() - candidate.enqueued_at:.1f}s, "
                        f"policy={self._queue.policy})"
                    )
                else:
                    # Best candidate doesn't fit. Try eviction once.
                    if (
                        self._eviction_callback is not None
                        and self._eviction_callback()
                    ):
                        continue
                    break

        return ready

    async def wait_for_admission(
        self,
        request_id: str,
        prompt_tokens: int,
        priority: int = 0,
        timeout: float = 600.0,
    ) -> None:
        """Block until this request can be admitted. Returns immediately if room.

        Raises asyncio.TimeoutError after *timeout* seconds (default 600).
        Raises asyncio.CancelledError if cancel() is called for this request.
        The caller MUST call on_request_complete() in a finally block after
        the request finishes generating.
        """
        admitted, position = self.try_admit(request_id, prompt_tokens, priority)
        if admitted:
            return
        event = asyncio.Event()
        self._wait_events[request_id] = event
        self._cancelled_ids: set = getattr(self, "_cancelled_ids", set())
        logger.info(f"[admit] {request_id} waiting for admission (position {position})")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._queue.cancel(request_id)
            self._wait_events.pop(request_id, None)
            raise
        except asyncio.CancelledError:
            self._queue.cancel(request_id)
            self._wait_events.pop(request_id, None)
            raise
        self._wait_events.pop(request_id, None)
        # Distinguish "admitted" from "cancelled" — cancel() marks the id
        if request_id in self._cancelled_ids:
            self._cancelled_ids.discard(request_id)
            raise asyncio.CancelledError(f"Request {request_id} was cancelled")

    def on_request_complete(self) -> List[QueuedRequest]:
        """Called when a request completes. Drains queue and signals waiters."""
        ready = self.check_queue()
        for entry in ready:
            event = self._wait_events.get(entry.request_id)
            if event:
                event.set()
        return ready

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued request. The waiter gets asyncio.CancelledError."""
        result = self._queue.cancel(request_id)
        self._cancelled_ids: set = getattr(self, "_cancelled_ids", set())
        self._cancelled_ids.add(request_id)
        event = self._wait_events.pop(request_id, None)
        if event:
            event.set()  # Wake up waiter — it checks _cancelled_ids and raises
        return result

    @property
    def queue_length(self) -> int:
        return len(self._queue)
