# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.

For MLLM models, all requests (text-only and multimodal) are routed through
the MLLMScheduler, which handles vision encoding and batched generation via
MLLMBatchGenerator. MLLM models only initialise the MLLM scheduler (not the
LLM engine), so text-only requests must also be routed through it.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, extract_multimodal_content, is_mllm_model
from ..message_utils import _normalize_messages
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
        mtp: bool = False,
        prefill_step_size: int | None = None,
        specprefill_enabled: bool = False,
        specprefill_draft_model_path: str | None = None,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_chunk_size: int = 4096,
        scheduler_policy: str = "fifo",
        scheduler_headroom_gb: float = 8.0,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (1=every token)
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable MTP per-request routing (text-only → TextModel, media → MLLM)
            prefill_step_size: Chunk size for prompt prefill (default 2048)
            specprefill_enabled: Enable SpecPrefill sparse prefill
            specprefill_draft_model_path: Draft model directory name under ~/ai-models/mlx_models/
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default 8192)
            specprefill_keep_pct: Fraction of tokens to keep (default 0.3)
            specprefill_chunk_size: Draft scoring chunk size for cooperative scheduling
                (default 4096). Set to 0 to disable cooperative mode (monolithic scoring).
            scheduler_policy: Request queue policy for admission control (default: fifo)
            scheduler_headroom_gb: Memory headroom in GB for admission control (default: 8.0)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._prefill_step_size = prefill_step_size or 2048

        # SpecPrefill configuration
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_draft_model_path = specprefill_draft_model_path
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_chunk_size = specprefill_chunk_size

        # Admission controller (memory-aware scheduling)
        self._scheduler_policy = scheduler_policy
        self._scheduler_headroom_gb = scheduler_headroom_gb
        self._admission: Optional["AdmissionController"] = None  # noqa: F821
        self._specprefill_lock = asyncio.Lock()
        self._draft_model = None

        self._model = None
        self._processor = None  # For MLLM
        self._tokenizer = None  # For LLM
        self._engine = None  # AsyncEngineCore for LLM
        self._mllm_scheduler = None  # MLLMScheduler for MLLM
        self._mllm_instance = None  # MLXMultimodalLM instance
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None
        self._text_generation_lock = asyncio.Lock()
        self._metal_lock = __import__("threading").Lock()  # Serializes all Metal GPU access

        # System prompt KV cache (reduces repeated prefill across requests)
        self._system_kv_snapshot = None  # List of (keys, values) per backbone layer
        self._system_kv_hash = None  # Hash of system prefix text
        self._system_kv_token_count = 0  # Tokens in cached prefix

    def _init_admission_controller(self, model_config) -> None:
        """Create the admission controller from model config.

        Extracts KV cache parameters from HuggingFace PretrainedConfig
        (attribute access) or dict-style configs. VLM models nest the
        language model config under ``text_config``.
        """
        from ..admission import AdmissionController, compute_kv_per_token

        def _get(cfg, key, default):
            """Read a config value from attr-based or dict-based config."""
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        # For VLM models, language model config is nested under text_config
        text_cfg = _get(model_config, "text_config", None)
        if text_cfg is not None:
            cfg = text_cfg
        else:
            cfg = model_config

        kv_per_token = compute_kv_per_token(
            num_hidden_layers=_get(cfg, "num_hidden_layers", 32),
            full_attention_interval=_get(cfg, "full_attention_interval", 1),
            num_kv_heads=_get(cfg, "num_key_value_heads", 8),
            head_dim=_get(cfg, "head_dim", 128),
        )

        self._admission = AdmissionController(
            kv_per_token=kv_per_token,
            headroom_bytes=int(self._scheduler_headroom_gb * 1024**3),
            policy=self._scheduler_policy,
        )
        logger.info(
            "[admission] KV per token: %s bytes, headroom: %.1fGB",
            f"{kv_per_token:,}",
            self._scheduler_headroom_gb,
        )

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

        # Initialize admission controller from config.json on disk
        try:
            import json
            from pathlib import Path

            config_path = Path(self._model_name) / "config.json"
            if config_path.exists():
                model_config = json.loads(config_path.read_text())
                self._init_admission_controller(model_config)
            else:
                logger.info("[admission] No config.json found — admission disabled")
        except Exception as e:
            logger.warning(f"[admission] Failed to init: {e} — admission disabled")

        # Build TextModel for MTP per-request routing (text-only → MTP, media → MLLM)
        if self._mtp:
            try:
                from ..text_model_from_vlm import build_text_model

                self._text_model = build_text_model(
                    self._mllm_instance.model, self._model_name
                )
                if self._text_model is not None:
                    # Get tokenizer from the MLLM instance (same model, shared tokenizer)
                    self._text_tokenizer = self._mllm_instance.get_tokenizer()

                    # Apply Qwen3.5 eos_token fix (matches SimpleEngine pattern)
                    if "qwen3" in self._model_name.lower():
                        self._text_tokenizer.eos_token = "<|im_end|>"
                        self._text_tokenizer.eos_token_id = (
                            self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                        )

                    # Check if TextModel actually has MTP
                    has_mtp = (
                        hasattr(self._text_model, "mtp")
                        and self._text_model.mtp is not None
                    )
                    if has_mtp:
                        logger.info(
                            "BatchedEngine MLLM+MTP routing: "
                            "text-only → TextModel (MTP), media → MLLM"
                        )
                    else:
                        logger.warning(
                            "TextModel built but no MTP head — "
                            "text-only won't use MTP"
                        )
                        self._text_model = None
                        self._text_tokenizer = None
            except Exception as e:
                logger.error(f"MTP TextModel build failed: {e}")
                self._text_model = None
                self._text_tokenizer = None

        # Load SpecPrefill draft model (for TextModel path — sparse cache
        # is incompatible with MTP, so specprefill generates autoregressively)
        if self._specprefill_enabled and self._specprefill_draft_model_path:
            try:
                from pathlib import Path

                from mlx_lm import load as mlx_lm_load

                draft_path = str(
                    Path.home()
                    / "ai-models"
                    / "mlx_models"
                    / self._specprefill_draft_model_path
                )
                self._draft_model, _ = mlx_lm_load(draft_path)
                logger.info(
                    "SpecPrefill draft model loaded: %s (threshold=%d, keep=%.0f%%)",
                    self._specprefill_draft_model_path,
                    self._specprefill_threshold,
                    self._specprefill_keep_pct * 100,
                )
            except Exception as e:
                logger.warning("Failed to load SpecPrefill draft model: %s", e)
                self._specprefill_enabled = False
                self._draft_model = None

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

        # Validate MTP support if enabled
        if self._scheduler_config and self._scheduler_config.enable_mtp:
            from ..patches.qwen3_next_mtp import validate_mtp_support

            if validate_mtp_support(self._model):
                logger.info("[MTP] Model validated for MTP speculative decoding")
            else:
                logger.warning(
                    "[MTP] MTP validation failed — --enable-mtp will be ignored. "
                    "See warnings above for details."
                )

        # Set Metal memory limits to make allocation failures graceful
        # instead of fatal Metal command buffer errors (SIGABRT)
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                device_info = mx.device_info()
                max_recommended = device_info.get(
                    "max_recommended_working_set_size",
                    device_info.get("memory_size", 0),
                )
                if max_recommended > 0:
                    soft_limit = int(max_recommended * 0.90)
                    mx.set_memory_limit(soft_limit)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)  # 32GB
                    logger.info(
                        f"Metal memory limits set: "
                        f"allocation_limit={soft_limit / 1e9:.1f}GB "
                        f"(90% of {max_recommended / 1e9:.1f}GB), "
                        f"cache_limit=32GB"
                    )
        except Exception as e:
            logger.warning(f"Failed to set Metal memory limits: {e}")

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

        # Initialize admission controller from config.json on disk
        # (mlx_lm models may not expose .config attribute)
        try:
            import json
            from pathlib import Path

            config_path = Path(self._model_name) / "config.json"
            if config_path.exists():
                model_config = json.loads(config_path.read_text())
                self._init_admission_controller(model_config)
            else:
                logger.info("[admission] No config.json found — admission disabled")
        except Exception as e:
            logger.warning(f"[admission] Failed to init: {e} — admission disabled")

        # Load SpecPrefill draft model (for LLM path)
        if self._specprefill_enabled and self._specprefill_draft_model_path:
            try:
                from pathlib import Path

                from mlx_lm import load as mlx_lm_load

                draft_path = str(
                    Path.home()
                    / "ai-models"
                    / "mlx_models"
                    / self._specprefill_draft_model_path
                )
                self._draft_model, _ = mlx_lm_load(draft_path)
                logger.info(
                    "SpecPrefill draft model loaded (LLM path): %s",
                    self._specprefill_draft_model_path,
                )
            except Exception as e:
                logger.warning("Failed to load SpecPrefill draft model: %s", e)
                self._specprefill_enabled = False
                self._draft_model = None

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
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._system_kv_snapshot = None
        self._system_kv_hash = None
        self._system_kv_token_count = 0
        self._admission = None
        self._loaded = False
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        num_images: int = 0,
    ) -> str:
        """Apply chat template to messages.

        Uses the processor's (or tokenizer's) apply_chat_template with the
        full message list so that system prompts and conversation history
        are preserved. The previous implementation extracted only the last
        user message text via mlx_vlm.prompt_utils.apply_chat_template,
        which dropped system prompts and all prior turns.
        """
        # Choose the best template applicator.
        # For MLLM models, the processor handles special vision tokens.
        # For text-only models, the tokenizer is sufficient.
        template_applicator = None
        if (
            self._is_mllm
            and self._processor
            and hasattr(self._processor, "apply_chat_template")
        ):
            template_applicator = self._processor
        elif hasattr(self.tokenizer, "apply_chat_template"):
            template_applicator = self.tokenizer

        if template_applicator is not None:
            # Convert OpenAI image_url content parts to HuggingFace format
            # so the processor can insert the correct vision placeholder tokens.
            if self._is_mllm and num_images > 0:
                messages = self._prepare_mllm_messages(messages)

            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools

            try:
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError as e:
                # Some templates don't accept 'tools'; retry without them.
                logger.debug(f"Chat template TypeError, retrying without extras: {e}")
                for key in ["tools"]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                return template_applicator.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            # Fallback for models without apply_chat_template
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    @staticmethod
    def _prepare_mllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style image_url content to HuggingFace format.

        The OpenAI API uses ``{"type": "image_url", "image_url": {"url": ...}}``
        while HuggingFace processors expect ``{"type": "image"}``.

        Args:
            messages: List of chat messages in OpenAI format. Each message is a
                dict with at least ``role`` and ``content`` keys.

        Returns:
            A new list of messages with ``image_url`` parts replaced by
            ``{"type": "image"}`` entries for the HuggingFace processor.
        """
        prepared = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        new_content.append({"type": "image"})
                    elif isinstance(part, (dict, str)):
                        new_content.append(part)
                    # skip non-dict/non-str parts to avoid passing unexpected types
                prepared.append({**msg, "content": new_content})
            else:
                prepared.append(msg)
        return prepared

    def _estimate_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count from chat messages for admission control.

        Uses the tokenizer when available, falls back to character-based
        estimate (1 token per 4 characters).
        """
        # Concatenate message text for a rough estimate
        text_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
        text = " ".join(text_parts)

        tokenizer = self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: ~4 chars per token
        return len(text) // 4

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

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all requests when model is multimodal.
            # MLLM models only initialise the _mllm_scheduler (not _engine),
            # so text-only requests must also be routed here.
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

        # Use LLM engine for text-only (non-MLLM models)
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

        if self._is_mllm and self._mllm_scheduler:
            # Use MLLM scheduler for all streaming when model is multimodal
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

        prefix_boundary = kwargs.pop("prefix_boundary", 0)
        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            prefix_boundary=prefix_boundary,
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

        For MLLM models, all requests (including text-only) are routed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

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

        # Normalize messages before any path (developer->system, merge consecutive)
        messages = _normalize_messages(messages)

        # Admission control — wait if memory is tight
        admission_id = None
        if self._admission is not None:
            prompt_tokens = self._estimate_prompt_tokens(messages)
            admission_id = str(uuid.uuid4())
            await self._admission.wait_for_admission(admission_id, prompt_tokens)

        try:
            # Per-request MTP routing: text-only → TextModel, media → MLLM
            if self._text_model is not None and not _has_media_content(messages):
                return await self._chat_text_model(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    **kwargs,
                )

            # Extract images/videos from messages (OpenAI multimodal format)
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
        finally:
            if self._admission is not None:
                self._admission.on_request_complete()

    def _compute_prefix_boundary(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None
    ) -> int:
        """Compute token count for the shared prefix across message variations.

        Uses a two-tokenization approach: tokenize the full prompt twice
        (once as-is, once with the last user message replaced by a dummy)
        and find the longest common prefix (LCP).  This gives the exact
        boundary where different user suffixes diverge, avoiding template
        discrepancies (e.g. Qwen3 <think> markers on last assistant).
        """
        # Find index of last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None or last_user_idx == 0:
            return 0
        try:
            template_tools = convert_tools_for_template(tools) if tools else None

            # Tokenize the real prompt
            real_prompt = self._apply_chat_template(messages, template_tools)

            # Build a dummy variant with different last user content
            dummy_messages = list(messages)
            dummy_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": "XXXXXXXXXX",
            }
            dummy_prompt = self._apply_chat_template(dummy_messages, template_tools)

            tokenizer = self.tokenizer
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer

            real_tokens = tokenizer.encode(real_prompt)
            dummy_tokens = tokenizer.encode(dummy_prompt)

            # Find LCP — the point where the two diverge is the boundary
            lcp = 0
            for j in range(min(len(real_tokens), len(dummy_tokens))):
                if real_tokens[j] != dummy_tokens[j]:
                    break
                lcp = j + 1

            return lcp
        except Exception:
            return 0

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

        For MLLM models, all requests (including text-only) are streamed through
        the MLLMScheduler for vision-aware batched generation.
        For non-MLLM models, uses the LLM engine with BatchGenerator.

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

        # Normalize messages before any path (developer->system, merge consecutive)
        messages = _normalize_messages(messages)

        # Admission control — wait if memory is tight
        admission_id = None
        if self._admission is not None:
            prompt_tokens = self._estimate_prompt_tokens(messages)
            admission_id = str(uuid.uuid4())
            await self._admission.wait_for_admission(admission_id, prompt_tokens)

        try:
            # Per-request MTP routing: text-only → TextModel, media → MLLM
            if self._text_model is not None and not _has_media_content(messages):
                async for output in self._stream_chat_text_model(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    **kwargs,
                ):
                    yield output
                return

            # Extract images/videos from messages (OpenAI multimodal format)
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

            # Compute prefix boundary for cache
            prefix_boundary = self._compute_prefix_boundary(messages, tools)
            if prefix_boundary > 0:
                kwargs["prefix_boundary"] = prefix_boundary

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
        finally:
            if self._admission is not None:
                self._admission.on_request_complete()

    async def _chat_text_model(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Non-streaming text-only generation via mlx_lm TextModel with MTP.

        Collects all streaming output into a single GenerationOutput.
        Used when MLLM+MTP routing is active and the request has no media.
        """
        logger.info("Text-only request → TextModel (MTP) [non-streaming]")
        accumulated_text = ""
        last_chunk = None
        async for chunk in self._stream_chat_text_model(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            **kwargs,
        ):
            accumulated_text = chunk.text
            last_chunk = chunk
        if last_chunk is not None:
            return GenerationOutput(
                text=accumulated_text,
                prompt_tokens=last_chunk.prompt_tokens,
                completion_tokens=last_chunk.completion_tokens,
                finish_reason=last_chunk.finish_reason,
            )
        return GenerationOutput(text="", finish_reason="stop")

    async def _stream_chat_text_model(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text-only generation via mlx_lm TextModel with MTP.

        Used when MLLM+MTP routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.

        SpecPrefill: when a draft model is loaded and the prompt exceeds the
        threshold, uses attention-based sparse prefill for faster TTFT.
        Composes with system KV cache (sparse-prefill only the suffix when
        cache hits). Falls back to normal path on any error.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct_override = kwargs.pop("specprefill_keep_pct", None)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Read enable_thinking from env (set by runtime_patches, consistent with MLLM path)
        enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
        enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if template_tools:
            template_kwargs["tools"] = template_tools

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

        # --- System KV cache: find system prefix boundary ---
        # ChatML (Qwen 3.5): everything before first <|im_start|>user is the system prefix
        USER_MARKER = "<|im_start|>user"
        marker_pos = prompt.find(USER_MARKER)
        if marker_pos > 0:
            system_prefix = prompt[:marker_pos]
            suffix = prompt[marker_pos:]
            prefix_hash = hashlib.sha256(system_prefix.encode()).hexdigest()[:16]
        else:
            system_prefix = None
            suffix = prompt
            prefix_hash = None

        # Check for cache hit
        cache_hit = (
            prefix_hash is not None
            and prefix_hash == self._system_kv_hash
            and self._system_kv_snapshot is not None
        )

        if cache_hit:
            logger.info(
                "Text-only request → TextModel (MTP) [streaming, system KV cache HIT: "
                "reusing %d cached tokens, hash=%s]",
                self._system_kv_token_count,
                prefix_hash,
            )
        else:
            logger.info("Text-only request → TextModel (MTP) [streaming]")

        prefill_step_size = self._prefill_step_size

        # --- SpecPrefill decision ---
        # Determine whether to use specprefill for this request.
        # Must be decided before entering the generation lock so we can
        # tokenize and check the threshold outside the critical section.
        _SPECPREFILL_MAX_TOKENS = 196608
        use_specprefill = False
        if self._draft_model is not None:
            if specprefill_override is True:
                use_specprefill = True
            elif specprefill_override is None and self._specprefill_enabled:
                use_specprefill = True
            # specprefill_override=False explicitly disables

        # Tokenize to determine token count for specprefill threshold check.
        # We need this for both specprefill and normal paths anyway.
        sp_tokens = None  # tokens to score (suffix or full prompt)
        sp_offset = 0  # position offset for sparse_prefill
        sp_n_total = 0  # total prompt tokens (for logging / threshold)

        if use_specprefill:
            if cache_hit:
                # Score only the suffix — system prefix is already cached
                sp_tokens = self._text_tokenizer.encode(suffix)
                sp_offset = self._system_kv_token_count
                sp_n_total = sp_offset + len(sp_tokens)
            else:
                # Score the full prompt
                sp_tokens = self._text_tokenizer.encode(prompt)
                sp_offset = 0
                sp_n_total = len(sp_tokens)

            n_sp_tokens = len(sp_tokens)

            # Threshold check (skip when force-enabled via per-request override)
            if (
                specprefill_override is not True
                and n_sp_tokens <= self._specprefill_threshold
            ):
                use_specprefill = False

            # Upper bound: cap to avoid draft model OOM
            if use_specprefill and n_sp_tokens > _SPECPREFILL_MAX_TOKENS:
                logger.warning(
                    "SpecPrefill: prompt %d tokens exceeds max %d, "
                    "falling back to normal path",
                    n_sp_tokens,
                    _SPECPREFILL_MAX_TOKENS,
                )
                use_specprefill = False

        # --- Cooperative SpecPrefill: draft scoring OUTSIDE the lock ---
        # When chunk_size > 0, draft scoring runs outside _text_generation_lock
        # via ChunkedDraftScorer, yielding between chunks so active requests
        # can generate tokens. The draft model is separate from the target
        # model — no shared mutable state.
        # When chunk_size == 0, the old monolithic path runs entirely under lock.
        use_cooperative = use_specprefill and self._specprefill_chunk_size > 0
        importance = None  # Pre-computed importance for cooperative path

        if use_cooperative:
            from ..cooperative_specprefill import ChunkedDraftScorer

            scorer = ChunkedDraftScorer(
                draft_model=self._draft_model,
                tokens=sp_tokens,
                chunk_size=self._specprefill_chunk_size,
                prefill_step_size=prefill_step_size,
            )
            try:
                while scorer.is_scoring:
                    def _locked_step():
                        with self._metal_lock:
                            return scorer.step()
                    await asyncio.to_thread(_locked_step)
                    await asyncio.sleep(0)  # Yield to event loop
                def _locked_finalize():
                    with self._metal_lock:
                        return scorer.finalize()
                importance = await asyncio.to_thread(_locked_finalize)
            except Exception as e:
                scorer.cleanup()
                logger.error(
                    "SpecPrefill cooperative scoring failed, "
                    "falling back to normal path: %s",
                    e,
                )
                use_specprefill = False
                use_cooperative = False
                importance = None

        # --- Selection + prefill + generation UNDER LOCK ---
        async with self._text_generation_lock:

            def _run_with_cache():
                if use_specprefill and use_cooperative and importance is not None:
                    try:
                        return _run_specprefill_with_importance(importance)
                    except Exception as e:
                        logger.error(
                            "SpecPrefill failed after scoring, "
                            "falling back to normal path: %s",
                            e,
                        )
                        # Fall through to normal path
                elif use_specprefill and not use_cooperative:
                    # Monolithic path (chunk_size=0): score + prefill under lock
                    try:
                        return _run_specprefill()
                    except Exception as e:
                        logger.error(
                            "SpecPrefill failed, falling back to normal path: %s", e
                        )
                        # Fall through to normal path
                if cache_hit:
                    return _run_cache_hit()
                else:
                    return _run_cache_miss()

            def _run_specprefill_with_importance(precomputed_importance):
                """Select chunks, sparse prefill, generate with pre-scored importance.

                Draft scoring has already been done outside the lock via
                ChunkedDraftScorer. This function handles phases 2-4:
                selection, sparse prefill, and autoregressive generation.

                Composes with system KV cache: when cache_hit, restores the
                system KV snapshot first, then sparse-prefills only the suffix
                tokens with position_offset = system_kv_token_count.

                Does NOT use MTP (sparse cache is incompatible with MTP
                speculative decoding).
                """
                import time
                from types import SimpleNamespace

                from ..specprefill import (
                    cleanup_rope,
                    select_chunks,
                    sparse_prefill,
                )

                # Build target cache (optionally restore system KV snapshot)
                target_cache = make_prompt_cache(self._text_model)
                if cache_hit:
                    for layer_idx, snapshot_state in enumerate(
                        self._system_kv_snapshot
                    ):
                        if layer_idx < len(target_cache):
                            target_cache[layer_idx].state = snapshot_state
                    mx.eval([c.state for c in target_cache if hasattr(c, "state")])

                try:
                    # Phase 2: Select important chunks
                    effective_keep = (
                        specprefill_keep_pct_override or self._specprefill_keep_pct
                    )
                    selected = select_chunks(
                        precomputed_importance, keep_pct=effective_keep
                    )
                    n_selected = selected.shape[0]
                    n_scored = len(sp_tokens)

                    # Phase 3: Sparse prefill on target model
                    t0 = time.monotonic()
                    logits = sparse_prefill(
                        self._text_model,
                        sp_tokens,
                        selected,
                        target_cache,
                        step_size=prefill_step_size,
                        position_offset=sp_offset,
                    )
                    t_prefill = time.monotonic() - t0

                    logger.info(
                        "SpecPrefill (cooperative): "
                        "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                        "(offset=%d, effective_keep=%.2f)",
                        n_selected,
                        n_scored,
                        n_selected / n_scored * 100,
                        t_prefill,
                        sp_offset,
                        effective_keep,
                    )

                    # Phase 4: Generate (simple autoregressive, no MTP)
                    eos_id = self._text_tokenizer.eos_token_id
                    y = sampler(logits[:, -1, :])
                    mx.eval(y)

                    results = []
                    generated_ids = []
                    prev_decoded = ""

                    for _ in range(max_tokens):
                        tok_id = y.item()
                        generated_ids.append(tok_id)

                        # Incremental text decode
                        decoded = self._text_tokenizer.decode(generated_ids)
                        new_text = decoded[len(prev_decoded) :]
                        prev_decoded = decoded

                        is_eos = tok_id == eos_id
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason="stop" if is_eos else None,
                            )
                        )

                        if is_eos:
                            break

                        # Next token
                        logits = self._text_model(y.reshape(1, -1), cache=target_cache)
                        y = sampler(logits[:, -1, :])
                        mx.eval(y)

                    return results, sp_n_total

                finally:
                    cleanup_rope(self._text_model)

            def _run_specprefill():
                """Monolithic: score tokens, sparse prefill, generate.

                Used when cooperative mode is disabled (chunk_size=0).
                Runs entirely under _text_generation_lock.

                Composes with system KV cache: when cache_hit, restores the
                system KV snapshot first, then sparse-prefills only the suffix
                tokens with position_offset = system_kv_token_count.

                Does NOT use MTP (sparse cache is incompatible with MTP
                speculative decoding).
                """
                import time
                from types import SimpleNamespace

                from ..specprefill import (
                    cleanup_rope,
                    score_tokens,
                    select_chunks,
                    sparse_prefill,
                )

                # Build target cache (optionally restore system KV snapshot)
                target_cache = make_prompt_cache(self._text_model)
                if cache_hit:
                    for layer_idx, snapshot_state in enumerate(
                        self._system_kv_snapshot
                    ):
                        if layer_idx < len(target_cache):
                            target_cache[layer_idx].state = snapshot_state
                    mx.eval([c.state for c in target_cache if hasattr(c, "state")])

                try:
                    # Phase 1: Score with draft model
                    t0 = time.monotonic()
                    importance = score_tokens(
                        self._draft_model,
                        sp_tokens,
                        prefill_step_size=prefill_step_size,
                    )
                    t_score = time.monotonic() - t0

                    # Phase 2: Select important chunks
                    effective_keep = (
                        specprefill_keep_pct_override or self._specprefill_keep_pct
                    )
                    selected = select_chunks(importance, keep_pct=effective_keep)
                    n_selected = selected.shape[0]
                    n_scored = len(sp_tokens)

                    # Phase 3: Sparse prefill on target model
                    t0 = time.monotonic()
                    logits = sparse_prefill(
                        self._text_model,
                        sp_tokens,
                        selected,
                        target_cache,
                        step_size=prefill_step_size,
                        position_offset=sp_offset,
                    )
                    t_prefill = time.monotonic() - t0

                    logger.info(
                        "SpecPrefill: scored %d tokens in %.1fs, "
                        "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                        "(offset=%d, effective_keep=%.2f)",
                        n_scored,
                        t_score,
                        n_selected,
                        n_scored,
                        n_selected / n_scored * 100,
                        t_prefill,
                        sp_offset,
                        effective_keep,
                    )

                    # Phase 4: Generate (simple autoregressive, no MTP)
                    eos_id = self._text_tokenizer.eos_token_id
                    y = sampler(logits[:, -1, :])
                    mx.eval(y)

                    results = []
                    generated_ids = []
                    prev_decoded = ""

                    for _ in range(max_tokens):
                        tok_id = y.item()
                        generated_ids.append(tok_id)

                        # Incremental text decode
                        decoded = self._text_tokenizer.decode(generated_ids)
                        new_text = decoded[len(prev_decoded) :]
                        prev_decoded = decoded

                        is_eos = tok_id == eos_id
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason="stop" if is_eos else None,
                            )
                        )

                        if is_eos:
                            break

                        # Next token
                        logits = self._text_model(y.reshape(1, -1), cache=target_cache)
                        y = sampler(logits[:, -1, :])
                        mx.eval(y)

                    return results, sp_n_total

                finally:
                    cleanup_rope(self._text_model)

            def _run_cache_hit():
                """Restore system KV snapshot, prefill only suffix, generate."""
                # Restore cached KV state into a fresh cache
                restored_cache = make_prompt_cache(self._text_model)
                for layer_idx, snapshot_state in enumerate(self._system_kv_snapshot):
                    if layer_idx < len(restored_cache):
                        restored_cache[layer_idx].state = snapshot_state
                mx.eval([c.state for c in restored_cache if hasattr(c, "state")])

                # Tokenize just the suffix and generate with the primed cache.
                # stream_generate accepts mx.array prompt (skips tokenization)
                # and prompt_cache is forwarded to mtp_generate_step.
                suffix_tokens = self._text_tokenizer.encode(suffix)
                suffix_array = mx.array(suffix_tokens)
                n_suffix = len(suffix_tokens)

                logger.info(
                    "System KV cache HIT: prefilling %d suffix tokens "
                    "(skipped %d cached tokens)",
                    n_suffix,
                    self._system_kv_token_count,
                )

                results = []
                for resp in mlx_stream_generate(
                    self._text_model,
                    self._text_tokenizer,
                    prompt=suffix_array,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=True,
                    prompt_cache=restored_cache,
                    prefill_step_size=prefill_step_size,
                ):
                    results.append(resp)
                return results, self._system_kv_token_count + len(suffix_tokens)

            def _run_cache_miss():
                """Full prefill + generation, then snapshot system KV for next time."""
                results = []
                for resp in mlx_stream_generate(
                    self._text_model,
                    self._text_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    mtp=True,
                    prefill_step_size=prefill_step_size,
                ):
                    results.append(resp)

                # Snapshot system KV for next request (if we found a system prefix)
                if prefix_hash is not None and system_prefix is not None:
                    try:
                        _snapshot_system_kv()
                    except Exception as e:
                        logger.warning("Failed to snapshot system KV cache: %s", e)

                # Get total prompt token count from generation response
                prompt_tokens = 0
                if results and hasattr(results[0], "prompt_tokens"):
                    prompt_tokens = results[0].prompt_tokens
                return results, prompt_tokens

            def _snapshot_system_kv():
                """Prefill just the system prefix on a fresh cache and save snapshot."""
                snapshot_cache = make_prompt_cache(self._text_model)
                prefix_tokens = self._text_tokenizer.encode(system_prefix)
                prefix_ids = mx.array(prefix_tokens)

                # Chunked prefill of system prefix
                for i in range(0, prefix_ids.size, prefill_step_size):
                    chunk = prefix_ids[i : i + prefill_step_size]
                    self._text_model(chunk[None], cache=snapshot_cache)
                    mx.eval([c.state for c in snapshot_cache if hasattr(c, "state")])

                # Save snapshot: deep copy of each cache layer's state
                self._system_kv_snapshot = []
                for c in snapshot_cache:
                    state = c.state
                    if isinstance(state, tuple) and len(state) == 2:
                        # KVCache: (keys, values) — copy to detach from cache
                        keys, values = state
                        self._system_kv_snapshot.append(
                            (mx.array(keys), mx.array(values))
                        )
                    elif isinstance(state, list):
                        # ArraysCache: list of arrays (Mamba/hybrid)
                        self._system_kv_snapshot.append(
                            [mx.array(a) if a is not None else None for a in state]
                        )
                    else:
                        # Unknown cache type — store as-is
                        self._system_kv_snapshot.append(state)

                self._system_kv_token_count = len(prefix_tokens)
                self._system_kv_hash = prefix_hash

                cache_bytes = 0
                for entry in self._system_kv_snapshot:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        cache_bytes += entry[0].nbytes + entry[1].nbytes
                    elif isinstance(entry, list):
                        cache_bytes += sum(a.nbytes for a in entry if a is not None)
                logger.info(
                    "System KV cache: stored %d-token snapshot " "(%.1f MB), hash=%s",
                    len(prefix_tokens),
                    cache_bytes / 1e6,
                    prefix_hash,
                )

            def _locked_run_with_cache():
                with self._metal_lock:
                    return _run_with_cache()
            result = await asyncio.to_thread(_locked_run_with_cache)
            all_resps, prompt_token_count = result

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
                prompt_tokens=prompt_token_count,
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
                prompt_tokens=prompt_token_count,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

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
            mllm_stats = self._mllm_scheduler.get_stats()
            stats["mllm_scheduler"] = mllm_stats
            # Promote Metal memory stats to top-level for /v1/status
            for key in (
                "metal_active_memory_gb",
                "metal_peak_memory_gb",
                "metal_cache_memory_gb",
            ):
                if key in mllm_stats:
                    stats[key] = mllm_stats[key]
        elif self._engine:
            stats.update(self._engine.get_stats())

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": self._specprefill_enabled,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
                "chunk_size": self._specprefill_chunk_size,
                "cooperative": self._specprefill_chunk_size > 0,
            }

        # System KV cache stats
        if self._system_kv_snapshot is not None:
            cache_bytes = 0
            for entry in self._system_kv_snapshot:
                if isinstance(entry, tuple) and len(entry) == 2:
                    cache_bytes += entry[0].nbytes + entry[1].nbytes
                elif isinstance(entry, list):
                    cache_bytes += sum(a.nbytes for a in entry if a is not None)
            stats["system_kv_cache"] = {
                "tokens": self._system_kv_token_count,
                "hash": self._system_kv_hash,
                "memory_mb": round(cache_bytes / 1e6, 1),
            }

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._mllm_scheduler and self._mllm_scheduler.vision_cache:
            return self._mllm_scheduler.vision_cache.get_stats()
        elif self._engine:
            return self._engine.get_cache_stats()
        return None

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self._engine:
            return self._engine.save_cache_to_disk(cache_dir)
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self._engine:
            return self._engine.load_cache_from_disk(cache_dir)
        return 0
