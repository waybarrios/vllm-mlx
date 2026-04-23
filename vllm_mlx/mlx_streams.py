# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
from collections.abc import Iterable

import mlx.core as mx


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Bind mlx-lm/mlx-vlm generation streams to the current thread.

    MLX streams are thread-local. If a model is loaded on one thread and
    generation runs on another, module-level generation streams created during
    import can point at a stream that does not exist in the worker thread.
    """
    default_stream = mx.new_stream(mx.default_device())
    mx.set_default_stream(default_stream)
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "generation_stream"):
            module.generation_stream = default_stream
    return default_stream
