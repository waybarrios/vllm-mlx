# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.

For MLLM models, this engine supports a hybrid approach:
- Text-only requests: Use BatchGenerator for continuous batching
- Multimodal requests (with images/videos): Fall back to MLLM.chat() for correct processing

This is necessary because BatchGenerator only supports token IDs, not pixel_values.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, extract_multimodal_content, is_mllm_model
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


def _extract_media_from_messages(messages: list[dict[str, Any]]) -> tuple:
    """
    Extract images and videos from OpenAI-format messages.

    Returns:
        Tuple of (has_media, images_list, videos_list)
    """
    images = []
    videos = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            # Handle Pydantic models
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()

            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "image_url":
                img_url = item.get("image_url", {})
                if isinstance(img_url, str):
                    images.append(img_url)
                elif isinstance(img_url, dict):
                    url = img_url.get("url", "")
                    if url:
                        images.append(url)

            elif item_type == "image":
                img = item.get("image") or item.get("url", "")
                if img:
                    images.append(img)

            elif item_type == "video_url":
                vid_url = item.get("video_url", {})
                if isinstance(vid_url, str):
                    videos.append(vid_url)
                elif isinstance(vid_url, dict):
                    url = vid_url.get("url", "")
                    if url:
                        videos.append(url)

            elif item_type == "video":
                vid = item.get("video") or item.get("url", "")
                if vid:
                    videos.append(vid)

    has_media = bool(images or videos)
    return has_media, images, videos


class MLLMModelWrapper:
    """
    Wrapper for MLLM models to make them compatible with BatchGenerator.

    BatchGenerator expects model output to be subscriptable (logits array),
    but MLLM models return LanguageModelOutput objects. This wrapper extracts
    the logits from the output.

    Also handles Gemma 3's required pixel_values argument by injecting None
    for text-only requests.
    """

    def __init__(self, model):
        self._model = model
        # Detect if this is a Gemma 3 model (requires pixel_values as positional arg)
        self._is_gemma3 = (
            hasattr(model, "model_type")
            and "gemma3" in str(getattr(model, "model_type", "")).lower()
        )

    def __call__(self, *args, **kwargs):
        """Call the model and extract logits from LanguageModelOutput."""
        # Gemma 3 requires pixel_values as a positional argument, unlike Qwen
        # which makes it optional. Inject pixel_values=None for text-only requests.
        if self._is_gemma3 and "pixel_values" not in kwargs:
            kwargs["pixel_values"] = None

        output = self._model(*args, **kwargs)
        # If output has logits attribute, return just the logits
        if hasattr(output, "logits"):
            return output.logits
        return output

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self._model, name)


class BatchedEngine(BaseEngine):
    """
    Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.

    For MLLM (multimodal) models, this engine uses MLLMScheduler
    which handles images and videos alongside text generation.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        force_mllm: bool = False,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (1=every token)
            force_mllm: Force loading as MLLM even if not auto-detected
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._is_mllm = force_mllm or is_mllm_model(model_name)

        self._model = None
        self._processor = None  # For MLLM
        self._tokenizer = None  # For LLM
        self._engine = None  # AsyncEngineCore for LLM
        self._mllm_scheduler = None  # MLLMScheduler for MLLM
        self._mllm_instance = None  # MLXMultimodalLM instance
        self._loaded = False

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
        if self._is_mllm and self._processor:
            return getattr(self._processor, "tokenizer", self._processor)
        return self._tokenizer

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        if self._is_mllm:
            await self._start_mllm()
        else:
            await self._start_llm()

        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name} (mllm={self._is_mllm})")

    async def _start_mllm(self) -> None:
        """Start the MLLM engine with MLLMScheduler (continuous batching)."""
        from ..mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from ..models.mllm import MLXMultimodalLM

        # Load the MLLM model
        self._mllm_instance = MLXMultimodalLM(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        self._mllm_instance.load()

        self._model = self._mllm_instance.model
        self._processor = self._mllm_instance.processor

        # Create MLLM scheduler config with batch generator support
        if self._scheduler_config and hasattr(self._scheduler_config, "max_num_seqs"):
            max_num_seqs = self._scheduler_config.max_num_seqs
        else:
            max_num_seqs = 16  # Default for continuous batching

        # Get batch sizes from config if available
        prefill_batch_size = getattr(self._scheduler_config, "prefill_batch_size", 4)
        completion_batch_size = getattr(
            self._scheduler_config, "completion_batch_size", 16
        )

        mllm_config = MLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            enable_vision_cache=True,
            vision_cache_size=100,
        )

        # Create and start MLLM scheduler
        self._mllm_scheduler = MLLMScheduler(
            model=self._model,
            processor=self._processor,
            config=mllm_config,
        )
        await self._mllm_scheduler.start()

        logger.info(
            f"MLLM Scheduler started with continuous batching: "
            f"max_num_seqs={max_num_seqs}, prefill_batch={prefill_batch_size}, "
            f"completion_batch={completion_batch_size}"
        )

    async def _start_llm(self) -> None:
        """Start the LLM engine with AsyncEngineCore."""
        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig
        from ..utils.tokenizer import load_model_with_fallback

        # Build tokenizer config
        tokenizer_config = {"trust_remote_code": self._trust_remote_code}

        # Qwen3 fix
        if "qwen3" in self._model_name.lower() or "Qwen3" in self._model_name:
            tokenizer_config["eos_token"] = "<|im_end|>"

        self._model, self._tokenizer = load_model_with_fallback(
            self._model_name,
            tokenizer_config=tokenizer_config,
        )

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._mllm_scheduler:
            await self._mllm_scheduler.stop()
            self._mllm_scheduler = None

        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
            self._engine = None

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._mllm_instance = None
        self._loaded = False
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        num_images: int = 0,
    ) -> str:
        """Apply chat template to messages."""
        tokenizer = self.tokenizer

        if self._is_mllm and self._processor:
            # Use mlx_vlm's chat template for MLLM
            try:
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.utils import load_config

                config = getattr(self._model, "config", None)
                if config is None:
                    config = load_config(self._model_name)

                # Extract text from last user message
                text_prompt = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            text_prompt = content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str):
                                    text_prompt = item
                                    break
                                elif (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    text_prompt = item.get("text", "")
                                    break
                        break

                return apply_chat_template(
                    self._processor,
                    config,
                    text_prompt,
                    num_images=num_images,
                )
            except Exception as e:
                logger.warning(f"Failed to apply MLLM chat template: {e}")
                # Fall through to standard template

        if hasattr(tokenizer, "apply_chat_template"):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            }
            if tools:
                template_kwargs["tools"] = tools

            try:
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                for key in ["tools", "enable_thinking"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                return tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler and (images or videos):
            # Use MLLM scheduler for multimodal
            output = await self._mllm_scheduler.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return GenerationOutput(
                text=clean_output_text(output.output_text),
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
            )

        # Use LLM engine for text-only
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_output_text(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
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
            images: Optional image URLs/paths (for MLLM)
            videos: Optional video URLs/paths (for MLLM)
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        if self._is_mllm and self._mllm_scheduler and (images or videos):
            # Use MLLM scheduler for multimodal streaming
            request_id = await self._mllm_scheduler.add_request_async(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            async for output in self._mllm_scheduler.stream_outputs(request_id):
                yield GenerationOutput(
                    text=clean_output_text(output.output_text),
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                )
            return

        # Use LLM engine for text-only
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        async for output in self._engine.stream_outputs(request_id):
            text = clean_output_text(output.output_text)

            yield GenerationOutput(
                text=text,
                new_text=output.new_text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finished=output.finished,
                finish_reason=output.finish_reason,
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

        For MLLM models with images/videos, uses the native MLLM.chat() method
        which properly processes multimodal content through the vision encoder.
        For text-only requests, uses BatchGenerator for continuous batching.

        Args:
            messages: List of chat messages (OpenAI format)
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

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
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

        For MLLM models with images/videos, uses the native MLLM.stream_chat() method
        which properly processes multimodal content through the vision encoder.
        For text-only requests, uses BatchGenerator for continuous batching.

        Args:
            messages: List of chat messages (OpenAI format)
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

        # Extract images/videos from messages (OpenAI multimodal format)
        # Note: We only use extracted media here, messages are already processed by server
        _, extracted_images, extracted_videos = extract_multimodal_content(messages)
        all_images = (images or []) + extracted_images
        all_videos = (videos or []) + extracted_videos

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            num_images=len(all_images),
        )

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }

        if self._mllm_scheduler:
            stats["mllm_scheduler"] = self._mllm_scheduler.get_stats()
        elif self._engine:
            stats.update(self._engine.get_stats())

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._mllm_scheduler and self._mllm_scheduler.vision_cache:
            return self._mllm_scheduler.vision_cache.get_stats()
        elif self._engine:
            return self._engine.get_cache_stats()
        return None
