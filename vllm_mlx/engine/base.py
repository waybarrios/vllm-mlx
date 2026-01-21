# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for vllm-mlx inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[List[dict]] = None,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[List[dict]] = None,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        return {}

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics. Override in subclasses."""
        return None
