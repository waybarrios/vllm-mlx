# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, is_mllm_model
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


_MEDIA_TYPES = frozenset(
    {
        "image_url",
        "video_url",
        "audio_url",
        "image",
        "video",
        "audio",
    }
)


def _has_media_content(messages: list) -> bool:
    """Check if any message contains media content (images, video, audio)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _MEDIA_TYPES:
                    return True
    return False


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        force_mllm: bool = False,
        mtp: bool = False,
        prefill_step_size: int = 2048,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._prefill_step_size = prefill_step_size

        self._model = None
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
            )
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                mtp=self._mtp,
                prefill_step_size=self._prefill_step_size,
            )

        self._model.load()
        self._loaded = True

        # Build parallel mlx_lm TextModel for text-only MTP routing
        if self._is_mllm and self._mtp:
            try:
                from ..text_model_from_vlm import build_text_model

                self._text_model = build_text_model(self._model.model, self._model_name)

                if (
                    self._text_model is not None
                    and hasattr(self._text_model, "mtp")
                    and self._text_model.mtp is not None
                ):
                    self._text_tokenizer = self._model.get_tokenizer()

                    # Apply Qwen3.5 eos_token fix (matches MLXLanguageModel.load)
                    if "qwen3" in self._model_name.lower():
                        self._text_tokenizer.eos_token = "<|im_end|>"
                        self._text_tokenizer.eos_token_id = (
                            self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                        )

                    logger.info(
                        "MLLM+MTP routing: text-only → mlx_lm TextModel (MTP=True), "
                        "media → mlx_vlm"
                    )
                else:
                    logger.warning(
                        "TextModel built but no MTP — text-only requests won't use MTP"
                    )
                    self._text_model = None

            except Exception as e:
                logger.error("MLLM+MTP routing setup failed: %s", e)
                self._text_model = None
                self._text_tokenizer = None

        mtp_info = f", MTP={self._mtp}" if self._mtp else ""
        routing = ", routing=per-request" if self._text_model is not None else ""
        logger.info(
            f"SimpleEngine loaded: {self._model_name} "
            f"(MLLM={self._is_mllm}{mtp_info}{routing})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._loaded = False
        logger.info("SimpleEngine stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
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
        if not self._loaded:
            await self.start()

        async with self._generation_lock:
            # Run in thread pool to allow asyncio timeout to work
            output = await asyncio.to_thread(
                self._model.generate,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

            # Clean output text
            text = clean_output_text(output.text)

            return GenerationOutput(
                text=text,
                tokens=getattr(output, "tokens", []),
                prompt_tokens=getattr(output, "prompt_tokens", 0),
                completion_tokens=getattr(
                    output, "completion_tokens", len(getattr(output, "tokens", []))
                ),
                finish_reason=output.finish_reason,
            )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
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
        if not self._loaded:
            await self.start()

        async with self._generation_lock:
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            finished = False

            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                prompt_tokens = (
                    chunk.prompt_tokens
                    if hasattr(chunk, "prompt_tokens")
                    else prompt_tokens
                )
                completion_tokens += 1
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                accumulated_text += new_text

                finished = (
                    getattr(chunk, "finished", False) or completion_tokens >= max_tokens
                )
                finish_reason = None
                if finished:
                    finish_reason = getattr(chunk, "finish_reason", "stop")

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break

            if not finished:
                if prompt_tokens == 0:
                    prompt_tokens = len(self._model.tokenizer.encode(prompt))
                yield GenerationOutput(
                    text=accumulated_text,
                    new_text="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=True,
                    finish_reason=None,
                )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
        if not self._loaded:
            await self.start()

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        async with self._generation_lock:
            if self._is_mllm:
                # For MLLM, use the chat method which handles images/videos
                # Run in thread pool to allow asyncio timeout to work
                output = await asyncio.to_thread(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=template_tools,
                    **kwargs,
                )
                text = clean_output_text(output.text)
                return GenerationOutput(
                    text=text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finish_reason=output.finish_reason,
                )
            else:
                # For LLM, use the chat method
                # Run in thread pool to allow asyncio timeout to work
                output = await asyncio.to_thread(
                    self._model.chat,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=template_tools,
                    **kwargs,
                )
                text = clean_output_text(output.text)
                return GenerationOutput(
                    text=text,
                    tokens=output.tokens,
                    completion_tokens=len(output.tokens),
                    finish_reason=output.finish_reason,
                )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
        if not self._loaded:
            await self.start()

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Per-request routing: text-only through mlx_lm with MTP
        if (
            self._is_mllm
            and self._text_model is not None
            and not _has_media_content(messages)
        ):
            logger.info("Text-only request → LLM path (MTP=True)")
            async for chunk in self._stream_generate_text(
                messages,
                max_tokens,
                temperature,
                top_p,
                tools=template_tools,
                **kwargs,
            ):
                yield chunk
            return

        # Build prompt using tokenizer
        if self._is_mllm:
            if self._text_model is not None:
                logger.info("Media request → MLLM path")
            # For MLLM, use stream_chat which yields tokens incrementally.
            # Must hold _generation_lock to prevent concurrent Metal access
            # (e.g. OpenCode sends title + main request simultaneously).
            async with self._generation_lock:
                accumulated_text = ""
                token_count = 0

                # Run stream_chat in thread pool since it's synchronous
                def run_stream():
                    return list(
                        self._model.stream_chat(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            tools=template_tools,
                            **kwargs,
                        )
                    )

                chunks = await asyncio.to_thread(run_stream)

                for chunk in chunks:
                    token_count += 1
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    finished = chunk.finish_reason is not None

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=chunk.finish_reason if finished else None,
                    )

                    if finished:
                        break
            return

        # For LLM, apply chat template and stream
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # Disable thinking mode for coder models since it interferes
            # with tool call parsing (tags leak as raw text).
            enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if template_tools:
                template_kwargs["tools"] = template_tools

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Some templates don't support all kwargs
                for key in ["tools", "enable_thinking"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # Stream generate
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        ):
            yield output

    async def _stream_generate_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Text-only generation via mlx_lm TextModel with MTP.

        Used when MLLM+MTP routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.
        """
        import os

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Read enable_thinking from env (set by runtime_patches, consistent with MLLM path)
        enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
        enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            template_kwargs["tools"] = tools

        try:
            prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            # Template doesn't accept tools= or enable_thinking=
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

        # Build sampler
        sampler = make_sampler(temp=temperature, top_p=top_p)
        max_tokens = max_tokens or 4096

        # Run under generation lock, all tokens in single thread (Metal safety)
        async with self._generation_lock:

            def _run_all():
                results = []
                for resp in mlx_stream_generate(
                    self._text_model,
                    self._text_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=True,
                    prefill_step_size=self._prefill_step_size,
                ):
                    results.append(resp)
                return results

            all_resps = await asyncio.to_thread(_run_all)

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        for i, resp in enumerate(all_resps):
            token_count += 1
            new_text = resp.text if hasattr(resp, "text") else str(resp)
            accumulated_text += new_text

            is_last = i == len(all_resps) - 1
            finished = is_last or token_count >= max_tokens

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=0,
                completion_tokens=token_count,
                finished=finished,
                finish_reason="stop" if finished else None,
            )

            if finished:
                break

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=0,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
        }

        # Include Metal memory stats
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics (for MLLM models)."""
        if self._is_mllm and self._model is not None:
            return self._model.get_cache_stats()
        return None
