"""Fallback surface for the optional Metal Context Engine extension.

The native module has the same import name (``vllm_mlx._metal_context``) and
is selected by Python automatically when it is present.  Keeping this small
fallback importable is important: normal MLX serving and Linux CI must not
require an Apple toolchain, while an explicit ``metal-context`` request still
gets an actionable capability error from the higher-level backend.
"""

from __future__ import annotations

import platform
from typing import Any

ABI_VERSION = 1


def capabilities() -> dict[str, Any]:
    """Return a precise, non-raising unavailable capability record."""

    return {
        "available": False,
        "compiled": False,
        "metal_device": False,
        "apple_silicon": platform.machine().lower() in {"arm64", "aarch64"},
        "serving_ready": False,
        "abi_version": ABI_VERSION,
        "backend": "metal-context",
        "reason": (
            "the optional native Metal Context Engine was not built; install "
            "on Apple Silicon with VLLM_MLX_BUILD_METAL_CONTEXT=1 and the "
            "Xcode Metal toolchain"
        ),
        "kernel": "metal_context_paged_decode",
        "block_sizes": (16, 32),
        "head_dims": (128,),
        "gqa": True,
        "partial_blocks": True,
        "online_softmax": True,
        "kv_dtype": "bfloat16",
    }


def paged_decode(*args: Any, **kwargs: Any) -> bytes:
    """Reject native dispatch when the optional extension is unavailable."""

    del args, kwargs
    raise RuntimeError("metal-context backend unavailable: " + capabilities()["reason"])


def shutdown() -> None:
    """Keep lifecycle calls harmless when no native module was installed."""


__all__ = ["ABI_VERSION", "capabilities", "paged_decode", "shutdown"]
