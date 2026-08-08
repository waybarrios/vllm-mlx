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
) -> object:
    """Give the calling thread a stream and point mlx-lm/mlx-vlm at it.

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
