"""Process-wide single-threaded executor for all MLX work.

Metal's per-stream state (command buffers, encoder lifecycle, completion
handlers) is not thread-safe. MLX does not guard it — the caller must
ensure only one Python thread submits work against a given ``mx.Stream``
(and in practice, against the whole Metal device) at a time.

Under load, two Apple driver asserts surface when this invariant is
broken:

    "A command encoder is already encoding to this command buffer"
    "Completed handler provided after commit call"

Both are fatal (they abort the process).

This module exposes a single-worker ``ThreadPoolExecutor`` that every
MLX-submitting code path in vllm-mlx must use: the continuous-batching
scheduler, the embeddings handler, the rerank handler, and any future
add-on that calls into Metal. Routing all MLX work through one thread
keeps the per-stream state consistent and eliminates the class of Metal
races described above.

The event loop is not blocked while work runs here — callers use
``await run_mlx(fn, ...)``. Submit overhead is on the order of tens of
microseconds.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_lock = threading.Lock()


def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the process-wide single-threaded MLX executor.

    Lazily constructed on first call. Safe to call from any thread.
    """
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mlx"
                )
    return _executor


async def run_mlx(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ``fn(*args, **kwargs)`` on the shared MLX thread and await its result.

    The event loop is free to schedule other coroutines while the MLX
    thread runs ``fn``. Any synchronous MLX-touching code path (model
    forward, ``mx.eval``, ``mx.clear_cache``, etc.) should be wrapped
    with this helper to keep all Metal submissions serialized on one
    thread.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(get_executor(), lambda: fn(*args, **kwargs))
    return await loop.run_in_executor(get_executor(), fn, *args)


def shutdown() -> None:
    """Tear down the MLX executor. Called from graceful-shutdown paths."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
