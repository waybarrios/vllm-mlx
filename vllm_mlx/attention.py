# SPDX-License-Identifier: Apache-2.0
"""
MLX Attention Backend for vLLM.

This module provides an attention backend that uses MLX's native
attention implementation, optimized for Apple Silicon.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class MLXAttentionMetadata:
    """Metadata for MLX attention computation."""

    # Sequence lengths for each request
    seq_lens: list[int]

    # Maximum sequence length in batch
    max_seq_len: int

    # Number of prefill tokens
    num_prefill_tokens: int = 0

    # Number of decode tokens
    num_decode_tokens: int = 0

    # Block tables for paged attention (if using)
    block_tables: Any | None = None

    # Slot mapping for KV cache
    slot_mapping: Any | None = None


class MLXAttentionBackend:
    """
    Attention backend using MLX's native attention.

    MLX provides optimized attention implementations that run on
    Apple Silicon's GPU via Metal. This backend wraps those
    implementations for use with vLLM.

    Note: mlx-lm handles attention internally, so this backend
    primarily serves as a compatibility layer.
    """

    @staticmethod
    def get_name() -> str:
        """Return backend name."""
        return "MLX"

    @staticmethod
    def get_impl_cls() -> type:
        """Return the implementation class."""
        return MLXAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type:
        """Return the metadata class."""
        return MLXAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """
        Get the shape of KV cache.

        Args:
            num_blocks: Number of cache blocks
            block_size: Tokens per block
            num_kv_heads: Number of KV attention heads
            head_size: Size of each attention head

        Returns:
            Shape tuple for KV cache tensor
        """
        # Shape: (num_blocks, block_size, num_kv_heads, head_size)
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        """Return supported attention head sizes."""
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def validate_configuration(
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        dtype: "torch.dtype",
        block_size: int,
        **kwargs,
    ) -> list[str]:
        """
        Validate attention configuration.

        Returns list of error messages (empty if valid).
        """
        errors = []

        if head_size not in MLXAttentionBackend.get_supported_head_sizes():
            errors.append(
                f"Head size {head_size} not in supported sizes: "
                f"{MLXAttentionBackend.get_supported_head_sizes()}"
            )

        return errors

    @staticmethod
    def supports_dtype(dtype: "torch.dtype") -> bool:
        """Check if dtype is supported."""
        import torch

        return dtype in [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def supports_block_size(block_size: int) -> bool:
        """Check if block size is supported."""
        return block_size in [8, 16, 32]

    @staticmethod
    def supports_attn_type(attn_type: str) -> bool:
        """Check if attention type is supported."""
        return attn_type in ["decoder", "encoder", "encoder_decoder"]


class MLXAttentionImpl:
    """
    MLX attention implementation.

    This class provides the actual attention computation using MLX.
    Since mlx-lm handles attention internally during generation,
    this serves as a compatibility interface.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: dict | None = None,
        logits_soft_cap: float | None = None,
        **kwargs,
    ):
        """
        Initialize MLX attention.

        Args:
            num_heads: Number of attention heads
            head_size: Size of each head
            scale: Attention scale factor
            num_kv_heads: Number of KV heads (for GQA/MQA)
            alibi_slopes: ALiBi position encoding slopes
            sliding_window: Sliding window attention size
            kv_cache_dtype: KV cache data type
            blocksparse_params: Block-sparse attention params
            logits_soft_cap: Soft cap for logits
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap

        logger.debug(
            f"MLXAttentionImpl initialized: heads={num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_size={head_size}"
        )

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any | None = None,
        attn_metadata: MLXAttentionMetadata | None = None,
        output: Any | None = None,
        **kwargs,
    ) -> Any:
        """
        Compute attention.

        Note: In the MLX backend, attention is handled internally by mlx-lm
        during the generation process. This method is provided for
        compatibility but may not be called directly.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            kv_cache: Optional KV cache
            attn_metadata: Attention metadata
            output: Optional output buffer

        Returns:
            Attention output tensor
        """
        try:
            import mlx.core as mx

            # Convert inputs to MLX arrays if needed
            if not isinstance(query, mx.array):
                query = mx.array(query.numpy() if hasattr(query, "numpy") else query)
            if not isinstance(key, mx.array):
                key = mx.array(key.numpy() if hasattr(key, "numpy") else key)
            if not isinstance(value, mx.array):
                value = mx.array(value.numpy() if hasattr(value, "numpy") else value)

            # Use MLX's scaled dot product attention
            # Shape: (batch, seq_len, num_heads, head_size)
            attn_output = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scale,
            )

            return attn_output

        except Exception as e:
            logger.error(f"MLX attention forward failed: {e}")
            raise


def create_mlx_attention_backend() -> type:
    """Factory function to create MLX attention backend."""
    return MLXAttentionBackend
