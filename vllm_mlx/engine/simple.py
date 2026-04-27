# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, has_media_content, is_mllm_model
from .base import (
    BaseEngine,
    GenerationOutput,
    cleanup_startup_cancellation,
    run_blocking_startup_work,
)
from ..mlx_streams import bind_generation_streams

logger = logging.getLogger(__name__)


def _bind_worker_generation_streams() -> None:
    """Rebind mlx generation streams inside the current worker thread."""
    bind_generation_streams()


def _seed_logits_processors(
    seed_tokens: mx.array | None,
    processors: list[Any] | None,
) -> list[Any] | None:
    """Wrap logits processors so continuation decode sees the full prompt."""
    if not processors:
        return None
    if seed_tokens is None or seed_tokens.size == 0:
        return list(processors)

    def _wrap(processor):
        def _seeded(tokens, logits):
            merged = seed_tokens
            if tokens is not None:
                if not isinstance(tokens, mx.array):
                    tokens_arr = mx.array(tokens, dtype=mx.uint32)
                else:
                    tokens_arr = tokens
                if tokens_arr.size > 0:
                    merged = mx.concatenate([seed_tokens, tokens_arr])
            return processor(merged, logits)

        return _seeded

    return [_wrap(processor) for processor in processors]


def _sample_with_processors(
    tokens: mx.array | None,
    logits: mx.array,
    sampler: Any,
    logits_processors: list[Any] | None,
) -> tuple[mx.array, mx.array]:
    """Sample a token while honoring any active logits processors."""
    if logits_processors:
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[None]
        for processor in logits_processors:
            logits = processor(tokens, logits)
        if is_1d:
            logits = logits.squeeze(0)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok = sampler(logprobs)
    return tok, logprobs


def _processors_can_retire(processors: list[Any] | None) -> bool:
    """True when any processor advertises a retire-to-content transition."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        isinstance(getattr(p, "is_retired", None), bool) for p in processors
    )


def _processors_retired(processors: list[Any] | None) -> bool:
    """True when any retire-capable processor has entered its retired state."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        getattr(p, "is_retired", False) is True for p in processors
    )


class _SpecPrefillCancelled(Exception):
    """Cooperative cancellation sentinel for blocking SpecPrefill workers."""


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        enable_cache: bool = True,
        force_mllm: bool = False,
        mtp: bool = False,
        mtp_num_draft_tokens: int = 1,
        prefill_step_size: int = 2048,
        specprefill_enabled: bool = False,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_draft_model: str | None = None,
        max_kv_size: int = 0,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            mtp_num_draft_tokens: Draft tokens per speculative MTP step
            prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
            specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
            specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
            specprefill_draft_model: Path to small draft model for importance scoring
            max_kv_size: Maximum KV cache size per sequence (0 = unbounded)
        """
        self._model_name = model_name
        self._created_at = time.time()
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._mtp_num_draft_tokens = mtp_num_draft_tokens
        self._prefill_step_size = prefill_step_size

        # SpecPrefill config
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_draft_model_path = specprefill_draft_model

        # KV cache size limit
        self._max_kv_size = max_kv_size

        self._model = None
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None

        # SpecPrefill draft model (loaded at start if enabled)
        self._draft_model = None

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()

        # System prompt KV cache (reduces repeated prefill across requests)
        self._system_kv_snapshot = None  # List of (keys, values) per backbone layer
        self._system_kv_hash = None  # Hash of system prefix text
        self._system_kv_token_count = 0  # Tokens in cached prefix

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

    def prepare_for_start(self) -> None:
        """Load the backing model off the serving event loop."""
        if self._model is not None:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
                max_kv_size=self._max_kv_size,
            )
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                mtp=self._mtp,
                mtp_num_draft_tokens=self._mtp_num_draft_tokens,
            )

        self._model.load()

    def _uses_default_prepare_for_start(self) -> bool:
        """Return True when prepare_for_start is the class implementation."""
        method = getattr(self.prepare_for_start, "__func__", None)
        return method is SimpleEngine.prepare_for_start

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return
        try:
            if self._model is None:
                if self._uses_default_prepare_for_start():
                    # MLX generation streams are thread-local. Keep model load on
                    # the event-loop thread so default LLM stream_generate() runs
                    # on the same thread that owns model-associated streams.
                    self.prepare_for_start()
                else:
                    # Test doubles and custom overrides may block; preserve the
                    # cancellation-safe threaded startup helper for those cases.
                    await run_blocking_startup_work(self.prepare_for_start)
            self._loaded = True

            if self._mtp and self._mtp_num_draft_tokens != 1:
                logger.warning(
                    "Native mlx_lm MTP currently ignores num_draft_tokens=%d; "
                    "effective speculative draft depth remains 1",
                    self._mtp_num_draft_tokens,
                )

            # Build parallel mlx_lm TextModel for text-only routing.
            # Even when MTP is disabled, text-only requests should not be trapped
            # on the slower mlx_vlm multimodal path.
            if self._is_mllm:
                try:
                    from ..text_model_from_vlm import build_text_model

                    self._text_model = build_text_model(
                        self._model.model, self._model_name
                    )

                    if self._text_model is not None:
                        self._text_tokenizer = self._model.get_tokenizer()

                        # Apply Qwen3.5 eos_token fix (matches MLXLanguageModel.load)
                        if "qwen3" in self._model_name.lower():
                            self._text_tokenizer.eos_token = "<|im_end|>"
                            self._text_tokenizer.eos_token_id = (
                                self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                            )

                        has_mtp = (
                            hasattr(self._text_model, "mtp")
                            and self._text_model.mtp is not None
                        )
                        logger.info(
                            "MLLM text routing: text-only -> mlx_lm TextModel "
                            "(MTP=%s), media -> mlx_vlm",
                            has_mtp and self._mtp,
                        )
                    else:
                        self._text_model = None
                        self._text_tokenizer = None

                except Exception as e:
                    logger.error("MLLM text routing setup failed: %s", e)
                    self._text_model = None
                    self._text_tokenizer = None

            # Load SpecPrefill draft model (small model for importance scoring)
            if self._specprefill_enabled and self._specprefill_draft_model_path:
                try:
                    from mlx_lm import load as mlx_lm_load

                    self._draft_model, _ = mlx_lm_load(
                        self._specprefill_draft_model_path
                    )
                    logger.info(
                        "SpecPrefill: draft model loaded (%s), threshold=%d, keep=%.0f%%",
                        self._specprefill_draft_model_path,
                        self._specprefill_threshold,
                        self._specprefill_keep_pct * 100,
                    )
                except Exception as e:
                    logger.error("SpecPrefill: draft model load failed: %s", e)
                    self._draft_model = None

            mtp_info = ""
            if self._mtp:
                mtp_info = (
                    f", MTP={self._mtp}(configured={self._mtp_num_draft_tokens}, "
                    "effective=1)"
                )
            routing = ", routing=per-request" if self._text_model is not None else ""
            specprefill_info = (
                ", SpecPrefill=active" if self._draft_model is not None else ""
            )
            logger.info(
                f"SimpleEngine loaded: {self._model_name} "
                f"(MLLM={self._is_mllm}{mtp_info}{routing}{specprefill_info})"
            )
        except asyncio.CancelledError:
            await cleanup_startup_cancellation(self.stop)
            raise

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._loaded = False
        self._system_kv_snapshot = None
        self._system_kv_hash = None
        self._system_kv_token_count = 0
        logger.info("SimpleEngine stopped")

    async def _run_blocking_serialized(self, func, /, *args, on_cancel=None, **kwargs):
        """Run a blocking MLX operation under the generation lock.

        Cancellation must not release the async lock before the worker thread
        finishes, or a follow-up request can enter MLX/Metal concurrently and
        corrupt the command-buffer state.
        """
        async with self._generation_lock:

            def run_bound():
                _bind_worker_generation_streams()
                return func(*args, **kwargs)

            task = asyncio.create_task(asyncio.to_thread(run_bound))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:
                        logger.debug(
                            "Blocking worker cancellation callback failed",
                            exc_info=True,
                        )
                try:
                    await task
                except BaseException:
                    pass
                raise

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

        Thin accumulator over stream_generate(). stream_generate() is the
        only code path that consumes per-request SpecPrefill overrides
        (`specprefill`, `specprefill_keep_pct`) and routes through
        _stream_generate_specprefill() when engaged. The prior direct
        self._model.generate() path silently dropped those overrides for
        non-streaming /v1/completions callers, so extra_body.specprefill
        was advertised by the server but had no effect on this route.

        By iterating stream_generate() and returning the last
        GenerationOutput, non-streaming clients get the same SpecPrefill
        engagement, accurate prompt_tokens reporting, and per-request
        override support as streaming clients.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional parameters forwarded to stream_generate,
                including per-request `specprefill` / `specprefill_keep_pct`

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        last_output: GenerationOutput | None = None
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            last_output = output

        if last_output is None:
            return GenerationOutput(text="", finish_reason="stop")

        text = clean_output_text(last_output.text)
        return GenerationOutput(
            text=text,
            tokens=list(last_output.tokens),
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            finish_reason=last_output.finish_reason,
            finished=True,
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

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct_override = kwargs.pop("specprefill_keep_pct", None)

        # SpecPrefill for non-MLLM models (MLLM+MTP handles it in _stream_generate_text)
        if not self._is_mllm and self._draft_model is not None:
            use_specprefill = True
            if specprefill_override is False:
                use_specprefill = False

            if use_specprefill:
                tokenizer = self._model.tokenizer
                add_special = tokenizer.bos_token is None or not prompt.startswith(
                    tokenizer.bos_token
                )
                tokens_list = tokenizer.encode(prompt, add_special_tokens=add_special)
                n_tokens = len(tokens_list)

                # Threshold check (skip when force-enabled via per-request override)
                if (
                    specprefill_override is not True
                    and n_tokens <= self._specprefill_threshold
                ):
                    use_specprefill = False

                # Upper bound: cap to avoid draft model OOM
                _SPECPREFILL_MAX_TOKENS = 65536
                if use_specprefill and n_tokens > _SPECPREFILL_MAX_TOKENS:
                    logger.warning(
                        "SpecPrefill: prompt %d tokens exceeds max %d, "
                        "falling back to normal path",
                        n_tokens,
                        _SPECPREFILL_MAX_TOKENS,
                    )
                    use_specprefill = False

                if use_specprefill:
                    async for output in self._stream_generate_specprefill(
                        prompt,
                        tokens_list,
                        max_tokens,
                        temperature,
                        top_p,
                        stop=stop,
                        specprefill_keep_pct=specprefill_keep_pct_override,
                        **kwargs,
                    ):
                        yield output
                    return

        async with self._generation_lock:
            # Non-stream chat runs in a worker thread and rebinds generation
            # streams there. Rebind again on the current thread before
            # stream_generate so nonstream->stream mode switches remain valid.
            _bind_worker_generation_streams()

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
                    if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens
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

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        async def aggregate_stream_chat() -> GenerationOutput:
            final_output = GenerationOutput(text="")
            async for output in self.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            ):
                final_output = output
            text = clean_output_text(final_output.text)
            return GenerationOutput(
                text=text,
                tokens=list(final_output.tokens),
                prompt_tokens=final_output.prompt_tokens,
                completion_tokens=final_output.completion_tokens,
                finish_reason=final_output.finish_reason,
            )

        # mlx-lm non-streaming chat with tools can stall indefinitely on some
        # local models, while the streaming path completes normally. Reuse the
        # streaming implementation and aggregate its final state so both chat
        # APIs share the same tool-capable execution path.
        if tools and not self._is_mllm:
            return await aggregate_stream_chat()

        # Text-only requests on MLLM models should always aggregate the
        # streaming path for non-streaming chat. This keeps one execution seam
        # and avoids mlx_vlm non-stream thread/stream ownership mismatches.
        if self._is_mllm and not has_media_content(messages):
            return await aggregate_stream_chat()

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        if self._is_mllm:
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
            output = await self._run_blocking_serialized(
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
            output = await self._run_blocking_serialized(
                self._model.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=template_tools,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            )
            text = clean_output_text(output.text)
            # Preserve upstream prompt accounting while routing the blocking
            # chat call through the cancellation-safe serialized runner.
            tokenizer = self._model.tokenizer
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if template_tools:
                template_kwargs["tools"] = template_tools
            prompt_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
            prompt_token_count = len(prompt_ids)
            return GenerationOutput(
                text=text,
                tokens=output.tokens,
                prompt_tokens=prompt_token_count,
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

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Per-request routing: text-only through mlx_lm TextModel
        if (
            self._is_mllm
            and self._text_model is not None
            and not has_media_content(messages)
        ):
            has_mtp = (
                hasattr(self._text_model, "mtp") and self._text_model.mtp is not None
            )
            logger.info("Text-only request → LLM path (MTP=%s)", has_mtp and self._mtp)
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
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
            accumulated_text = ""
            token_count = 0

            # Text-only fallback when no TextModel exists: keep execution on the
            # current thread. Routing through to_thread can break mlx_vlm stream
            # ownership on some models (Stream(gpu, N) mismatch).
            if self._text_model is None and not has_media_content(messages):
                local_kwargs = dict(kwargs)
                if chat_template_kwargs:
                    local_kwargs["chat_template_kwargs"] = chat_template_kwargs

                async with self._generation_lock:
                    _bind_worker_generation_streams()
                    for chunk in self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    ):
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

            # Run stream_chat in thread pool since it's synchronous
            def run_stream():
                local_kwargs = dict(kwargs)
                if chat_template_kwargs:
                    local_kwargs["chat_template_kwargs"] = chat_template_kwargs
                return list(
                    self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    )
                )

            chunks = await self._run_blocking_serialized(run_stream)

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
            # Per-request enable_thinking override; default: True unless coder model.
            enable_thinking = kwargs.pop("enable_thinking", None)
            if enable_thinking is None:
                enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            if template_tools:
                template_kwargs["tools"] = template_tools

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Some templates don't support all kwargs
                for key in ["tools", "enable_thinking", *chat_template_kwargs.keys()]:
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

    async def _stream_generate_specprefill(
        self,
        prompt: str,
        tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        specprefill_keep_pct: float | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """SpecPrefill path for non-MTP models (Nemotron, GPT-OSS, etc).

        Scores token importance with the draft model, sparse-prefills the target
        model, then generates autoregressively. Falls back to normal generation
        on any error.
        """
        from threading import Event

        model = self._model.model
        tokenizer = self._model.tokenizer
        n_tokens = len(tokens)
        cancel_requested = Event()

        def _request_cancel() -> None:
            cancel_requested.set()

        def _cancel_check() -> None:
            if cancel_requested.is_set():
                raise _SpecPrefillCancelled()

        def _run_all():
            try:
                return _run_specprefill()
            except _SpecPrefillCancelled:
                raise
            except Exception as e:
                logger.error("SpecPrefill failed, falling back to normal path: %s", e)
                return _run_normal()

        def _run_specprefill():
            """Score tokens, sparse prefill, generate autoregressively."""
            import time
            from types import SimpleNamespace

            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.sample_utils import make_sampler

            from ..specprefill import (
                cleanup_rope,
                score_tokens,
                select_chunks,
                sparse_prefill,
            )

            cache = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)

            try:
                # Phase 1: Score with draft model
                t0 = time.monotonic()
                importance = score_tokens(
                    self._draft_model,
                    tokens,
                    prefill_step_size=self._prefill_step_size,
                    cancel_check=_cancel_check,
                )
                t_score = time.monotonic() - t0

                # Phase 2: Select important chunks
                _cancel_check()
                effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                selected = select_chunks(importance, keep_pct=effective_keep)
                n_selected = selected.shape[0]

                # Phase 3: Sparse prefill on target model
                t0 = time.monotonic()
                logits = sparse_prefill(
                    model,
                    tokens,
                    selected,
                    cache,
                    step_size=self._prefill_step_size,
                    cancel_check=_cancel_check,
                )
                t_prefill = time.monotonic() - t0

                logger.info(
                    "SpecPrefill: scored %d tokens in %.1fs, "
                    "sparse prefill %d/%d (keep=%.0f%%) in %.1fs",
                    n_tokens,
                    t_score,
                    n_selected,
                    n_tokens,
                    n_selected / n_tokens * 100,
                    t_prefill,
                )

                # Phase 4: Generate via engine's standard pipelined path
                sampler = make_sampler(temp=temperature, top_p=top_p)
                _cancel_check()
                first_token_id = sampler(logits[:, -1, :]).item()
                first_text = tokenizer.decode([first_token_id])
                eos_id = tokenizer.eos_token_id

                results = [
                    SimpleNamespace(
                        text=first_text,
                        finish_reason="stop" if first_token_id == eos_id else None,
                    )
                ]

                if first_token_id != eos_id:
                    for chunk in self._model.stream_generate(
                        prompt=mx.array([first_token_id]),
                        max_tokens=max_tokens - 1,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        prompt_cache=cache,
                    ):
                        _cancel_check()
                        new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason=getattr(chunk, "finish_reason", None),
                            )
                        )

                return results

            finally:
                cleanup_rope(model)

        def _run_normal():
            """Fallback: normal generation without specprefill."""
            from types import SimpleNamespace

            results = []
            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                _cancel_check()
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                results.append(
                    SimpleNamespace(
                        text=new_text,
                        finish_reason=getattr(chunk, "finish_reason", None),
                    )
                )
            return results

        all_resps = await self._run_blocking_serialized(
            _run_all, on_cancel=_request_cancel
        )

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        for i, resp in enumerate(all_resps):
            token_count += 1
            new_text = resp.text
            accumulated_text += new_text

            is_last = i == len(all_resps) - 1
            finished = is_last or token_count >= max_tokens

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=finished,
                finish_reason=resp.finish_reason or ("stop" if finished else None),
            )

            if finished:
                break

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    async def _stream_generate_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Text-only generation via mlx_lm TextModel.

        Used when text-only MLLM routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models import cache as cache_module
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct = kwargs.pop("specprefill_keep_pct", None)
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        stop = kwargs.pop("stop", None)
        external_logits_processors = kwargs.pop("logits_processors", None)
        abort_event = threading.Event()

        # Per-request enable_thinking override; fall back to env var / default True.
        enable_thinking = kwargs.pop("enable_thinking", None)
        if enable_thinking is None:
            enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
            enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template for full prompt
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        template_kwargs.update(chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        try:
            full_prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            # Template doesn't accept tools= or enable_thinking=
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            full_prompt = self._text_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        penalty_processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        )
        all_processors = (external_logits_processors or []) + (penalty_processors or [])
        custom_logits_active = bool(all_processors)
        max_tokens = max_tokens or 4096

        # --- System prompt KV caching ---
        backbone_cache = None  # Backbone-only cache (no MTP), used by both paths
        prompt_to_send = full_prompt  # Default: send full prompt text
        cache_hit = False
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        system_tokens = None
        suffix_tokens = None
        full_tokens_list = None

        # Extract system messages for caching
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system and self._text_model is not None:
            # Find system prefix boundary in full prompt text.
            # ChatML format: system section ends where first non-system message begins.
            # Works with tools (rendered inside system section by Qwen templates).
            system_prefix_end = -1
            for marker in ("<|im_start|>user\n", "<|im_start|>assistant\n"):
                idx = full_prompt.find(marker)
                if idx > 0:
                    system_prefix_end = idx
                    break

            if system_prefix_end > 0:
                system_prefix_text = full_prompt[:system_prefix_end]
                system_hash = hashlib.sha256(system_prefix_text.encode()).hexdigest()[
                    :16
                ]

                # Tokenize both (matching stream_generate's tokenization logic)
                tokenizer = self._text_tokenizer
                add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                    tokenizer.bos_token
                )
                full_tokens_list = tokenizer.encode(
                    full_prompt, add_special_tokens=add_special
                )
                full_token_count = len(full_tokens_list)

                system_tokens_list = tokenizer.encode(
                    system_prefix_text, add_special_tokens=add_special
                )
                system_token_count = len(system_tokens_list)

                # Verify system tokens are a proper prefix of full tokens
                prefix_valid = (
                    len(full_tokens_list) > system_token_count
                    and full_tokens_list[:system_token_count] == system_tokens_list
                )

                if prefix_valid:
                    system_tokens = system_tokens_list
                    suffix_tokens = full_tokens_list[system_token_count:]

                    if (
                        system_hash == self._system_kv_hash
                        and self._system_kv_snapshot is not None
                        and system_token_count == self._system_kv_token_count
                    ):
                        # Cache HIT — restore KV state into fresh backbone cache
                        def make_cache_with_snapshot(
                            text_model,
                            system_kv_snapshot,
                            _max_kv_size=self._max_kv_size,
                        ):
                            import mlx.core as mx
                            from mlx_lm.models.cache import make_prompt_cache

                            backbone_cache = make_prompt_cache(
                                text_model, max_kv_size=_max_kv_size or None
                            )
                            for i, saved_state in enumerate(system_kv_snapshot):
                                backbone_cache[i].state = saved_state

                            prompt_to_send = mx.array(suffix_tokens)
                            return backbone_cache, prompt_to_send

                        backbone_cache, prompt_to_send = (
                            await self._run_blocking_serialized(
                                make_cache_with_snapshot,
                                self._text_model,
                                self._system_kv_snapshot,
                            )
                        )
                        cache_hit = True

                        logger.info(
                            "System KV cache HIT: reusing %d cached tokens, "
                            "prefilling %d new tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                    else:
                        # Cache MISS — will prefill system tokens and snapshot
                        logger.info(
                            "System KV cache MISS: will prefill %d system tokens, "
                            "%d suffix tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                else:
                    logger.debug(
                        "System KV cache: prefix token validation failed, "
                        "using full prompt (%d tokens)",
                        len(full_tokens_list),
                    )
                    system_token_count = 0

        # Determine if SpecPrefill should be used
        # Per-request boolean override: True = force enable, False = force disable
        if specprefill_override is False:
            use_specprefill = False
        elif specprefill_override is True and self._draft_model is not None:
            use_specprefill = True  # Force enable, skip threshold check
        else:
            use_specprefill = self._draft_model is not None

        # For specprefill, ensure we have token IDs (not just prompt text)
        if use_specprefill and suffix_tokens is None and full_tokens_list is None:
            tokenizer = self._text_tokenizer
            add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                tokenizer.bos_token
            )
            full_tokens_list = tokenizer.encode(
                full_prompt, add_special_tokens=add_special
            )
            full_token_count = len(full_tokens_list)

        # Tokens for specprefill: suffix (if system KV) or full prompt
        specprefill_tokens = (
            suffix_tokens if suffix_tokens is not None else full_tokens_list
        )
        specprefill_offset = system_token_count if suffix_tokens is not None else 0

        # Threshold check: only use specprefill on long prompts
        # (skipped when per-request boolean forces enable)
        if (
            use_specprefill
            and specprefill_override is not True
            and (
                specprefill_tokens is None
                or len(specprefill_tokens) <= self._specprefill_threshold
            )
        ):
            use_specprefill = False

        # Upper bound: cap specprefill to avoid draft model OOM on very long prompts
        # 65536 tokens ~ 2GB draft KV cache on Qwen3.5-4B (32KB/token x 8 attn layers)
        _SPECPREFILL_MAX_TOKENS = 65536
        if (
            use_specprefill
            and specprefill_tokens is not None
            and len(specprefill_tokens) > _SPECPREFILL_MAX_TOKENS
        ):
            logger.warning(
                "SpecPrefill: prompt %d tokens exceeds max %d, "
                "falling back to normal path",
                len(specprefill_tokens),
                _SPECPREFILL_MAX_TOKENS,
            )
            use_specprefill = False

        loop = asyncio.get_running_loop()
        response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _emit_response(resp: Any) -> None:
            if abort_event.is_set():
                return
            loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

        def _emit_done() -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

        def _emit_error(exc: BaseException) -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

        def _seed_from_last_response(prompt_cache, last_resp):
            last_tok = getattr(last_resp, "token", None)
            if last_tok is not None:
                cache_module.trim_prompt_cache(prompt_cache, 1)
                return mx.array([last_tok], dtype=mx.uint32)
            return mx.array(
                self._text_tokenizer.encode(getattr(last_resp, "text", "")),
                dtype=mx.uint32,
            )

        def _resume_after_processor_retirement(
            model,
            prompt_cache,
            prompt,
            remaining_tokens: int,
        ) -> None:
            resume_kwargs = dict(
                max_tokens=remaining_tokens,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
                prompt_cache=prompt_cache,
            )
            if hasattr(model, "make_mtp_cache") and model.mtp is not None:
                # Resume speculative decode from the retained backbone cache with
                # a fresh MTP cache so stale speculative state cannot survive the
                # processor-to-content handoff.
                resume_kwargs["prompt_cache"] = prompt_cache + model.make_mtp_cache()
                resume_kwargs["mtp"] = True
                resume_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
            for resp in mlx_stream_generate(
                model,
                self._text_tokenizer,
                prompt=prompt,
                **resume_kwargs,
            ):
                if abort_event.is_set():
                    logger.info("Text route: abort requested; stopping resume decode")
                    break
                _emit_response(resp)

        # Run all Metal ops in a single serialized thread.
        def _run_all():
            nonlocal backbone_cache, prompt_to_send

            model = self._text_model
            can_retire_processors = _processors_can_retire(all_processors)
            use_mtp = (
                self._mtp
                and not custom_logits_active
                and hasattr(model, "mtp")
                and model.mtp is not None
            )
            if self._mtp and custom_logits_active:
                logger.info(
                    "Text route: disabling MTP for request-local logits processors"
                )

            # Cache MISS with valid prefix: prefill system tokens and snapshot
            if (
                not cache_hit
                and system_token_count > 0
                and system_tokens is not None
                and suffix_tokens is not None
            ):
                mc = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)
                sys_arr = mx.array(system_tokens)

                # Prefill system tokens in chunks (matching generate_step)
                step = self._prefill_step_size
                while sys_arr.size > step:
                    model(sys_arr[:step][None], cache=mc)
                    mx.eval([c.state for c in mc])
                    sys_arr = sys_arr[step:]
                    mx.clear_cache()
                if sys_arr.size > 0:
                    model(sys_arr[None], cache=mc)
                    mx.eval([c.state for c in mc])

                # Snapshot backbone cache (immutable mx.arrays, safe to reuse)
                snapshot = [c.state for c in mc]
                mx.eval([s for pair in snapshot for s in pair])

                self._system_kv_snapshot = snapshot
                self._system_kv_hash = system_hash
                self._system_kv_token_count = system_token_count

                backbone_cache = mc
                prompt_to_send = mx.array(suffix_tokens)
                logger.info(
                    "System KV cache: stored %d-token snapshot (%.1f MB), "
                    "prefilling %d remaining",
                    system_token_count,
                    sum(c.nbytes for c in mc) / 1e6,
                    len(suffix_tokens),
                )

            # --- SpecPrefill path (with fallback to normal on failure) ---
            if use_specprefill:
                try:
                    _run_specprefill(model, backbone_cache, use_mtp)
                    return
                except Exception as e:
                    logger.error(
                        "SpecPrefill failed, falling back to normal MTP path: %s",
                        e,
                    )
                    # Discard potentially corrupted cache
                    backbone_cache = None
                    prompt_to_send = full_prompt

            # --- Normal path (mlx_lm stream_generate) ---
            prompt_cache = None
            if backbone_cache is not None:
                # Add MTP cache on top of backbone
                if use_mtp and hasattr(model, "make_mtp_cache"):
                    mtp_cache = model.make_mtp_cache()
                    prompt_cache = backbone_cache + mtp_cache
                else:
                    prompt_cache = backbone_cache

            gen_kwargs = dict(
                max_tokens=max_tokens,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
            )
            if all_processors:
                gen_kwargs["logits_processors"] = all_processors
            if use_mtp:
                gen_kwargs["mtp"] = True
                gen_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
            if prompt_cache is not None:
                gen_kwargs["prompt_cache"] = prompt_cache
            if can_retire_processors and not use_mtp:
                shared_cache = prompt_cache
                if shared_cache is None:
                    shared_cache = make_prompt_cache(
                        model, max_kv_size=self._max_kv_size or None
                    )
                gen_kwargs["prompt_cache"] = shared_cache

                token_count = 0
                last_resp = None
                retired = False
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "Text route: abort requested; stopping decode after %d tokens",
                            token_count,
                        )
                        break
                    _emit_response(resp)
                    token_count += 1
                    last_resp = resp
                    retired = _processors_retired(all_processors)
                    if retired:
                        logger.info(
                            "Text route: request-local processor retired after %d tokens; "
                            "resuming content phase with MTP=%s",
                            token_count,
                            hasattr(model, "make_mtp_cache") and model.mtp is not None,
                        )
                        break

                if retired and token_count < max_tokens and last_resp is not None:
                    seed = _seed_from_last_response(shared_cache, last_resp)
                    _resume_after_processor_retirement(
                        model,
                        shared_cache,
                        seed,
                        max_tokens - token_count,
                    )
            else:
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info("Text route: abort requested; stopping decode")
                        break
                    _emit_response(resp)

        def _run_specprefill(model, bc, use_mtp):
            """Score tokens, sparse prefill, then continue on the standard decode path."""
            from types import SimpleNamespace

            from mlx_lm import stream_generate as mlx_stream_generate
            from mlx_lm.models.cache import make_prompt_cache

            from ..specprefill import (
                cleanup_rope,
                score_tokens,
                select_chunks,
                sparse_prefill,
            )

            # Create backbone cache if not already from system KV
            if bc is None:
                bc = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)

            try:
                # Phase 1: Score with draft model
                import time

                t0 = time.monotonic()
                importance = score_tokens(
                    self._draft_model,
                    specprefill_tokens,
                    prefill_step_size=self._prefill_step_size,
                )
                t_score = time.monotonic() - t0

                # Phase 2: Select important chunks
                effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                selected = select_chunks(importance, keep_pct=effective_keep)
                n_selected = selected.shape[0]
                n_total = len(specprefill_tokens)

                # Phase 3: Sparse prefill on target model
                t0 = time.monotonic()
                logits = sparse_prefill(
                    model,
                    specprefill_tokens,
                    selected,
                    bc,
                    step_size=self._prefill_step_size,
                    position_offset=specprefill_offset,
                )
                t_prefill = time.monotonic() - t0

                logger.info(
                    "SpecPrefill: scored %d tokens in %.1fs, "
                    "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                    "(offset=%d, effective_keep=%.2f)",
                    n_total,
                    t_score,
                    n_selected,
                    n_total,
                    n_selected / n_total * 100,
                    t_prefill,
                    specprefill_offset,
                    effective_keep,
                )

                # Phase 4: Sample the first token from the prefilled logits, then
                # continue through mlx_lm's normal decode path so MTP and request-
                # local logits processors remain active after sparse prefill.
                eos_id = self._text_tokenizer.eos_token_id
                seed_tokens = (
                    mx.array(full_tokens_list, dtype=mx.uint32)
                    if full_tokens_list is not None
                    else None
                )
                seeded_processors = _seed_logits_processors(seed_tokens, all_processors)
                y, _ = _sample_with_processors(
                    None,
                    logits[:, -1, :].squeeze(0),
                    sampler,
                    seeded_processors,
                )
                mx.eval(y)

                generated_ids = []
                prev_decoded = ""

                tok_id = y.item()
                generated_ids.append(tok_id)

                decoded = self._text_tokenizer.decode(generated_ids)
                new_text = decoded[len(prev_decoded) :]
                prev_decoded = decoded

                is_eos = tok_id == eos_id
                _emit_response(
                    SimpleNamespace(
                        text=new_text,
                        finish_reason="stop" if is_eos else None,
                    )
                )

                if abort_event.is_set():
                    logger.info(
                        "SpecPrefill text route: abort requested after seed token"
                    )
                    return

                if is_eos or max_tokens <= 1:
                    return

                prompt_cache = bc
                if use_mtp and hasattr(model, "make_mtp_cache"):
                    prompt_cache = bc + model.make_mtp_cache()

                continuation_prompt = mx.array([tok_id], dtype=mx.uint32)
                token_count = 1
                if _processors_retired(all_processors) and token_count < max_tokens:
                    logger.info(
                        "SpecPrefill text route: request-local processor retired after seed token; "
                        "resuming content phase with MTP=%s",
                        hasattr(model, "make_mtp_cache") and model.mtp is not None,
                    )
                    _resume_after_processor_retirement(
                        model,
                        bc,
                        continuation_prompt,
                        max_tokens - token_count,
                    )
                    return

                last_resp = None
                retired = False
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=continuation_prompt,
                    max_tokens=max_tokens - token_count,
                    sampler=sampler,
                    prefill_step_size=self._prefill_step_size,
                    logits_processors=seeded_processors,
                    prompt_cache=prompt_cache,
                    mtp=use_mtp,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "SpecPrefill text route: abort requested; stopping decode"
                        )
                        break
                    _emit_response(resp)
                    token_count += 1
                    last_resp = resp
                    retired = _processors_retired(all_processors)
                    if retired:
                        logger.info(
                            "SpecPrefill text route: request-local processor retired after %d tokens; "
                            "resuming content phase with MTP=%s",
                            token_count,
                            hasattr(model, "make_mtp_cache") and model.mtp is not None,
                        )
                        break

                if retired and token_count < max_tokens and last_resp is not None:
                    seed = _seed_from_last_response(bc, last_resp)
                    _resume_after_processor_retirement(
                        model,
                        bc,
                        seed,
                        max_tokens - token_count,
                    )

            finally:
                cleanup_rope(model)

        async def _produce_responses() -> None:
            try:
                await self._run_blocking_serialized(
                    _run_all,
                    on_cancel=abort_event.set,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                _emit_error(exc)
            else:
                _emit_done()

        producer_task = asyncio.create_task(_produce_responses())

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        try:
            while True:
                kind, payload = await response_queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise payload
                resp = payload

                token_count += 1
                new_text = resp.text if hasattr(resp, "text") else str(resp)
                accumulated_text += new_text

                stop_hit = False
                if stop:
                    stop_hit = any(stop_seq in accumulated_text for stop_seq in stop)
                finished = stop_hit or token_count >= max_tokens
                finish_reason = getattr(resp, "finish_reason", None)
                if stop_hit:
                    finish_reason = "stop"
                elif finish_reason is None and finished:
                    finish_reason = "stop"
                elif finish_reason is not None:
                    finished = True

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=full_token_count or 0,
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break
        finally:
            if not producer_task.done():
                abort_event.set()
            await producer_task

        if not finished:
            yield GenerationOutput(
                text=accumulated_text,
                new_text="",
                prompt_tokens=full_token_count or 0,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "uptime_seconds": time.time() - self._created_at,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
        }

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": True,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
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

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches."""
        if self._is_mllm and self._model is not None:
            self._model.clear_cache()
            return {"model_cache": True}
        return None
