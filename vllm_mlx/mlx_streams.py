# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
import threading
from collections.abc import Iterable

import mlx.core as mx

# Serialize stream rebinding so module-level generation_stream references are
# updated atomically across concurrent engine threads.
_STREAM_REBIND_LOCK = threading.Lock()


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
    stream: object | None = None,
) -> object:
    """Give the calling thread a stream and point mlx-lm/mlx-vlm at it.

    Pass ``stream`` to re-point at a stream this thread already owns instead of
    allocating another one; a thread that binds per request would otherwise leak
    a new stream each time.

    Call this **once**, on the single thread that owns MLX work, before any
    model is loaded on it.

    It does not make state portable between threads, and must not be used to
    try. MLX streams exist only in the thread that created them, and an array
    with pending primitives carries the stream those primitives were built on.
    Once a model or a prompt cache exists, rebinding this module-level handle
    changes a global that the existing buffers do not consult, so evaluating
    them from another thread still raises "There is no Stream(gpu, N) in
    current thread". Load and generation must simply share one thread.
    """
    with _STREAM_REBIND_LOCK:
        default_stream = (
            stream if stream is not None else mx.new_stream(mx.default_device())
        )
        mx.set_default_stream(default_stream)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", default_stream)
        return default_stream


def snapshot_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> dict[str, object]:
    """Record the module-level generation streams before a rebind.

    ``generation_stream`` is a module attribute, so it is process-global even
    though the stream it names is only usable on its creating thread. A worker
    that binds and then exits leaves that global naming a stream no live thread
    can enter, and the next caller gets "There is no Stream(gpu, N) in current
    thread". Pair this with :func:`restore_generation_streams` when the binding
    thread is retired.
    """
    snapshot: dict[str, object] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "generation_stream"):
            snapshot[module_name] = getattr(module, "generation_stream")
    return snapshot


def restore_generation_streams(snapshot: dict[str, object]) -> None:
    """Put back the handles captured by :func:`snapshot_generation_streams`."""
    if not snapshot:
        return
    with _STREAM_REBIND_LOCK:
        for module_name, stream in snapshot.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", stream)
