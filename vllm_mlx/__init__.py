# SPDX-License-Identifier: Apache-2.0
"""
vllm-mlx: Apple Silicon MLX backend for vLLM

This package provides native Apple Silicon GPU acceleration for vLLM
using Apple's MLX framework, mlx-lm for LLMs, and mlx-vlm for
vision-language models.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Support for LLM and multimodal models
"""

__version__ = "0.2.0"

# Continuous batching engine (core functionality, no torch required)
from vllm_mlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from vllm_mlx.engine import EngineCore, AsyncEngineCore, EngineConfig
from vllm_mlx.prefix_cache import PrefixCacheManager, PrefixCacheStats
from vllm_mlx.vlm_cache import VLMCacheManager, VLMCacheStats

# vLLM integration components (require torch)
# These are loaded lazily to allow CLI usage without torch
def __getattr__(name):
    """Lazy load vLLM integration components."""
    if name == "MLXPlatform":
        from vllm_mlx.platform import MLXPlatform
        return MLXPlatform
    elif name == "MLXWorker":
        from vllm_mlx.worker import MLXWorker
        return MLXWorker
    elif name == "MLXModelRunner":
        from vllm_mlx.model_runner import MLXModelRunner
        return MLXModelRunner
    elif name == "MLXAttentionBackend":
        from vllm_mlx.attention import MLXAttentionBackend
        return MLXAttentionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core (lazy loaded, require torch)
    "MLXPlatform",
    "MLXWorker",
    "MLXModelRunner",
    "MLXAttentionBackend",
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # LLM Prefix cache
    "PrefixCacheManager",
    "PrefixCacheStats",
    # VLM Cache (images/video)
    "VLMCacheManager",
    "VLMCacheStats",
    # Version
    "__version__",
]
