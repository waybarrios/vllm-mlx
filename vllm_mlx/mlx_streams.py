# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads.

MLX >= 0.31.2 (PR #3348) makes Metal CommandEncoders thread-local.  Arrays
created on one thread carry a stream index that does not exist in other threads'
TLS.  This means a model loaded on thread A cannot be used for inference on
thread B — the runtime raises:

    RuntimeError: There is no Stream(gpu, N) in current thread

The fundamental fix is to ensure model loading AND all inference happen on the
**same** persistent thread.  ``MLXWorkerThread`` provides this guarantee by
running a dedicated daemon thread with a simple task queue.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import queue
import threading
from collections.abc import Awaitable, Iterable
from typing import Any, Callable, TypeVar

import mlx.core as mx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Serialize stream rebinding so module-level generation_stream references are
# updated atomically across concurrent engine threads.
_STREAM_REBIND_LOCK = threading.Lock()


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Bind mlx-lm/mlx-vlm generation streams to the current thread.

    MLX streams are thread-local. If a model is loaded on one thread and
    generation runs on another, module-level generation streams created during
    import can point at a stream that does not exist in the worker thread.

    .. deprecated::
        Prefer ``MLXWorkerThread`` which keeps model load and inference on the
        same persistent thread, avoiding the need to rebind streams entirely.
    """
    with _STREAM_REBIND_LOCK:
        default_stream = mx.new_stream(mx.default_device())
        mx.set_default_stream(default_stream)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", default_stream)
        return default_stream


class MLXWorkerThread:
    """Persistent single-threaded executor for MLX/Metal operations.

    All submitted callables run on the **same** OS thread for the lifetime of
    the process.  This guarantees that:

    1. ``mx.new_stream()`` is called exactly once (during thread init).
    2. Model loading and inference share the same thread-local stream context.
    3. No ``RuntimeError: There is no Stream(gpu, N)`` can occur.

    Usage::

        worker = MLXWorkerThread()

        # From an async context:
        loop = asyncio.get_event_loop()
        result = await worker.submit(loop, heavy_mlx_function, arg1, arg2)

        # Shutdown (optional, thread is daemon):
        worker.shutdown()
    """

    def __init__(self, name: str = "mlx-worker") -> None:
        self._task_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True
        )
        self._thread.start()
        logger.debug("MLXWorkerThread '%s' started (tid=%d)", name, self._thread.ident)

    def _run(self) -> None:
        """Worker loop: pull (fn, args, kwargs, future) tuples and execute."""
        while True:
            item = self._task_queue.get()
            if item is None:
                # Shutdown sentinel
                break
            fn, args, kwargs, fut = item
            try:
                result = fn(*args, **kwargs)
                fut.get_loop().call_soon_threadsafe(fut.set_result, result)
            except BaseException as exc:
                fut.get_loop().call_soon_threadsafe(fut.set_exception, exc)

    def submit(
        self,
        loop: asyncio.AbstractEventLoop,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[T]:
        """Submit a callable to the worker thread, returning an awaitable Future.

        Args:
            loop: The asyncio event loop to create the Future on.
            fn: The callable to execute on the worker thread.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            An ``asyncio.Future`` that resolves with ``fn``'s return value or
            raises its exception.
        """
        fut = loop.create_future()
        self._task_queue.put((fn, args, kwargs, fut))
        return fut

    def shutdown(self) -> None:
        """Signal the worker thread to exit (best-effort, non-blocking)."""
        self._task_queue.put(None)

    @property
    def is_alive(self) -> bool:
        """Whether the worker thread is running."""
        return self._thread.is_alive()
