# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for vllm-mlx inference.

Provides two engine implementations:
- SimpleEngine: Direct model calls for maximum single-user throughput
- BatchedEngine: Continuous batching for multiple concurrent users

Also re-exports core engine components for backwards compatibility.
"""

from .base import BaseEngine, GenerationOutput
from .simple import SimpleEngine
from .batched import BatchedEngine

_ENGINE_CORE_NAMES = frozenset({"EngineCore", "AsyncEngineCore", "EngineConfig"})


def __getattr__(name: str):
    """Lazily re-export engine_core symbols to avoid importing mlx at package load time."""
    if name in _ENGINE_CORE_NAMES:
        from .. import engine_core

        return getattr(engine_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEngine",
    "GenerationOutput",
    "SimpleEngine",
    "BatchedEngine",
    # Core engine components (lazy)
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
