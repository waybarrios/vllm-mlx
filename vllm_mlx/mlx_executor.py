# SPDX-License-Identifier: Apache-2.0
"""MLXExecutor: Centralized thread-routing abstraction for MLX operations.

MLX streams and tensor memory are thread-local. Buffers carry the stream
of the thread that created them, and evaluating or modifying them from another
thread raises "There is no Stream(gpu, N) in current thread".

MLXExecutor owns the single generation thread (or wraps an existing worker)
and guarantees all operations touching MLX (model inference, prompt cache
loading/saving, runtime cache clears, etc.) execute on the correct owner thread.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from .mlx_streams import bind_generation_streams

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MLXExecutor:
    """Unified dispatcher for MLX operations ensuring thread-affinity.

    All MLX invocations (model forward, scheduler stepping, prefix cache
    load/save, clear_cache) should route through this executor to prevent
    stream thread-affinity errors.

    If a call is already executing on the owner thread, ``run()`` and ``arun()``
    execute inline without thread-dispatch overhead.
    """

    def __init__(
        self,
        worker: ThreadPoolExecutor | None = None,
        owns_worker: bool | None = None,
        inline: bool = False,
        thread_name_prefix: str = "engine-core",
    ) -> None:
        """Initialize MLXExecutor.

        Args:
            worker: Existing ThreadPoolExecutor to use. If None and not inline,
                    a single-worker pool is created.
            owns_worker: Whether this executor owns the lifecycle of the worker pool.
                         Defaults to True if worker is newly created, False if passed in.
            inline: If True, execute all tasks inline on the caller's thread (useful
                    for MLLM models whose stepping occurs on the Event Loop).
            thread_name_prefix: Prefix for created worker threads.
        """
        self._inline = inline
        self._streams_bound = False
        self._owner_thread_id: int | None = None

        if inline:
            self._worker = None
            self._owns_worker = False
            self._owner_thread_id = threading.get_ident()
        elif worker is not None:
            self._worker = worker
            self._owns_worker = owns_worker if owns_worker is not None else False
        else:
            self._worker = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=thread_name_prefix
            )
            self._owns_worker = owns_worker if owns_worker is not None else True

    @property
    def worker(self) -> ThreadPoolExecutor | None:
        """Return the underlying ThreadPoolExecutor, if any."""
        return self._worker

    @property
    def is_inline(self) -> bool:
        """Return True if executor runs tasks inline on the caller's thread."""
        return self._inline

    @property
    def owner_thread_id(self) -> int | None:
        """Return the ident of the owner thread."""
        if self._owner_thread_id is None and self._worker is not None:
            self._owner_thread_id = self._worker.submit(threading.get_ident).result()
        return self._owner_thread_id

    def bind_streams(self, bind_fn: Callable[[], Any] | None = None) -> None:
        """Bind MLX streams on the owner thread.

        Must be called once on the owner thread before loading models or prompt
        caches, so that all created buffers carry the owner stream.
        """
        fn = bind_fn if bind_fn is not None else bind_generation_streams

        def _bind() -> int:
            fn()
            return threading.get_ident()

        if self._inline:
            self._owner_thread_id = _bind()
            self._streams_bound = True
        elif self._worker is not None:
            self._owner_thread_id = self._worker.submit(_bind).result()
            self._streams_bound = True


    def is_on_owner_thread(self) -> bool:
        """Check if the current thread is the designated MLX owner thread."""
        if self._inline:
            return True
        owner_id = self.owner_thread_id
        return owner_id is not None and threading.get_ident() == owner_id

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Synchronously execute a callable on the MLX owner thread.

        If already on the owner thread, runs inline with zero dispatch overhead.
        Otherwise blocks until the worker thread finishes execution.
        """
        if self._inline or self._worker is None or self.is_on_owner_thread():
            return fn(*args, **kwargs)
        return self._worker.submit(fn, *args, **kwargs).result()

    async def arun(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Asynchronously execute a callable on the MLX owner thread.

        If already on the owner thread, runs inline (and awaits if awaitable).
        Otherwise dispatches to the worker thread via loop.run_in_executor
        without blocking the asyncio event loop.
        """
        if self._inline or self._worker is None or self.is_on_owner_thread():
            res = fn(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res

        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(self._worker, call)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying worker pool if owned by this executor."""
        if self._owns_worker and self._worker is not None:
            self._worker.shutdown(wait=wait)
            self._worker = None
            self._owner_thread_id = None
            self._streams_bound = False
