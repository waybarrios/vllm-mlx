# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
import sys
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, cast

import mlx.core as mx

# Serialize stream rebinding so module-level generation_stream references are
# updated atomically across concurrent engine threads.
_STREAM_REBIND_LOCK = threading.RLock()


@dataclass(eq=False)
class GenerationStreamBinding:
    """Generation globals temporarily owned by one retiring worker."""

    _previous: dict[str, object]
    _bound: dict[str, object] = field(default_factory=dict)

    def retire(self) -> None:
        """Restore owned globals without reviving an already retired worker."""
        with _STREAM_REBIND_LOCK:
            if self not in _ACTIVE_GENERATION_BINDINGS:
                return
            for module_name, bound in self._bound.items():
                previous = self._previous[module_name]
                for binding in _ACTIVE_GENERATION_BINDINGS:
                    if binding._previous.get(module_name) is bound:
                        binding._previous[module_name] = previous
                module = sys.modules.get(module_name)
                if getattr(module, "generation_stream", None) is bound:
                    setattr(module, "generation_stream", previous)
            _ACTIVE_GENERATION_BINDINGS.remove(self)


_ACTIVE_GENERATION_BINDINGS: list[GenerationStreamBinding] = []


@contextmanager
def capture_generation_streams() -> Iterator[GenerationStreamBinding]:
    """Atomically capture a worker rebind and track its predecessor handles.

    Keep the token until all work on the owning worker finishes, then retire
    it. Bind failures retire immediately because no generation work can start.
    """
    with _STREAM_REBIND_LOCK:
        binding = GenerationStreamBinding(snapshot_generation_streams())
        completed = False
        try:
            yield binding
            completed = True
        finally:
            for module_name, previous in binding._previous.items():
                module = sys.modules.get(module_name)
                if hasattr(module, "generation_stream"):
                    bound = getattr(module, "generation_stream")
                    if bound is not previous:
                        binding._bound[module_name] = bound
            if binding._bound:
                _ACTIVE_GENERATION_BINDINGS.append(binding)
            if not completed:
                binding.retire()


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
        # Callers retain opaque handles returned by this function; MLX validates them.
        mx.set_default_stream(cast("mx.Stream", default_stream))
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
