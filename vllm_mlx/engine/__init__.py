# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for vllm-mlx inference.

The package stays intentionally light at import time so server- and
contract-level tests can import API modules without eagerly importing MLX,
engine_core, or the batched engine stack.
"""

from __future__ import annotations

_ENGINE_CORE_NAMES = frozenset({"EngineCore", "AsyncEngineCore", "EngineConfig"})
from typing import TYPE_CHECKING

from .base import BaseEngine, GenerationOutput

if TYPE_CHECKING:
    from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
    from .batched import BatchedEngine
    from .simple import SimpleEngine

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


def __getattr__(name: str):
    if name == "SimpleEngine":
        from .simple import SimpleEngine

        return SimpleEngine

    if name == "BatchedEngine":
        from .batched import BatchedEngine

        return BatchedEngine

    if name in {"EngineCore", "AsyncEngineCore", "EngineConfig"}:
        from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore

        return {
            "EngineCore": EngineCore,
            "AsyncEngineCore": AsyncEngineCore,
            "EngineConfig": EngineConfig,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
