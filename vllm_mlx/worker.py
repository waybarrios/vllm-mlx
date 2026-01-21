# SPDX-License-Identifier: Apache-2.0
"""
MLX Worker for vLLM.

This module implements a vLLM worker that uses Apple's MLX framework
for model execution on Apple Silicon.
"""

import gc
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = logging.getLogger(__name__)


class MLXWorker:
    """
    Worker implementation for MLX-based inference on Apple Silicon.

    This worker uses mlx-lm for model loading and inference, providing
    native Apple Silicon GPU acceleration through Metal.

    Unlike CUDA workers that use PyTorch with CUDA, this worker:
    - Uses MLX arrays instead of PyTorch tensors for model weights
    - Leverages unified memory (no CPU<->GPU transfers needed)
    - Uses Metal-optimized kernels for attention and other operations
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize MLX worker.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index (usually 0 for single GPU)
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.load_config = vllm_config.load_config

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # MLX model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_runner = None

        # Device info
        self.device = torch.device("cpu")  # MLX uses its own device management

        logger.info(f"Initializing MLX Worker (rank={rank}, local_rank={local_rank})")

    def init_device(self) -> None:
        """Initialize MLX device and verify it's working."""
        try:
            import mlx.core as mx

            # Verify MLX is using GPU
            default_device = mx.default_device()
            logger.info(f"MLX default device: {default_device}")

            # Get device info
            from vllm_mlx.plugin import get_mlx_device_info

            info = get_mlx_device_info()
            logger.info(
                f"MLX Device: {info['chip_name']} with {info['memory_gb']:.1f}GB"
            )

            # Initialize model runner
            from vllm_mlx.model_runner import MLXModelRunner

            self.model_runner = MLXModelRunner(self.vllm_config)

        except ImportError as e:
            raise ImportError(
                f"MLX is required for MLXWorker: {e}. "
                "Install with: pip install mlx mlx-lm"
            )

    def load_model(self) -> None:
        """Load model using mlx-lm."""
        if self.model_runner is None:
            raise RuntimeError("init_device() must be called before load_model()")

        self.model_runner.load_model()
        logger.info(f"Model loaded: {self.model_config.model}")

    def determine_available_memory(self) -> int:
        """
        Determine available memory for KV cache.

        On Apple Silicon with unified memory, we use a portion of system RAM.
        """
        import subprocess

        try:
            # Get total system memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_memory = int(result.stdout.strip())

            # Use configured GPU memory utilization
            utilization = self.cache_config.gpu_memory_utilization
            available = int(total_memory * utilization * 0.5)  # Be conservative

            logger.info(
                f"Available memory for KV cache: {available / (1024**3):.2f}GB "
                f"(utilization: {utilization})"
            )
            return available

        except Exception as e:
            logger.warning(f"Could not determine memory: {e}, using default 4GB")
            return 4 * 1024 * 1024 * 1024  # 4GB default

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize KV cache with the given size."""
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        if self.model_runner:
            self.model_runner.initialize_cache(num_gpu_blocks)

        logger.info(f"Initialized cache: {num_gpu_blocks} GPU blocks")

    def get_kv_cache_spec(self) -> dict:
        """Get KV cache specification."""
        if self.model_runner:
            return self.model_runner.get_kv_cache_spec()
        return {}

    def compile_or_warm_up_model(self) -> None:
        """Warm up model for inference."""
        if self.model_runner:
            self.model_runner.warm_up()
        logger.info("Model warm-up complete")

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "ModelRunnerOutput | None":
        """
        Execute model inference for the given scheduler output.

        Args:
            scheduler_output: Contains requests to process

        Returns:
            ModelRunnerOutput with generation results
        """
        if self.model_runner is None:
            raise RuntimeError("Model not loaded")

        return self.model_runner.execute_model(scheduler_output)

    def get_model(self):
        """Get the underlying model."""
        if self.model_runner:
            return self.model_runner.model
        return None

    def check_health(self) -> None:
        """Check worker health."""
        try:
            import mlx.core as mx

            # Simple check - create and evaluate a small array
            test = mx.array([1.0, 2.0, 3.0])
            _ = mx.sum(test).item()
        except Exception as e:
            raise RuntimeError(f"MLX health check failed: {e}")

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down MLX Worker")

        # Clear model
        self.model = None
        self.tokenizer = None
        self.model_runner = None

        # Clear MLX cache
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception:
            pass

        gc.collect()

    # LoRA methods (not yet supported on MLX)
    def add_lora(self, lora_request) -> bool:
        logger.warning("LoRA not yet supported on MLX backend")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    # Sleep mode (not applicable for MLX)
    def sleep(self, level: int = 1) -> None:
        logger.debug("Sleep mode not applicable for MLX (unified memory)")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.debug("Wake up not applicable for MLX (unified memory)")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model_config.get_vocab_size()

    def get_cache_block_size_bytes(self) -> int:
        """Get size of a cache block in bytes."""
        if self.model_runner:
            return self.model_runner.get_cache_block_size_bytes()

        # Default calculation
        head_size = self.model_config.get_head_size()
        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        # 2 for K and V, assuming float16
        block_size = self.cache_config.block_size
        return 2 * block_size * num_layers * num_heads * head_size * 2

    def profile(self, is_start: bool = True) -> None:
        """Profiling (not yet implemented for MLX)."""
        logger.debug("Profiling not yet implemented for MLX")

    def __repr__(self) -> str:
        return f"<MLXWorker rank={self.rank} local_rank={self.local_rank}>"
