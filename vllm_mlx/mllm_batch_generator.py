# SPDX-License-Identifier: Apache-2.0
"""
MLLM Batch Generator for multimodal continuous batching.

This module implements continuous batching for Multimodal Language Models (MLLMs)
like Qwen3-VL, following the same architecture as LLM continuous batching but
adapted for vision models.

Key insight: VLM models have a `model.language_model` which is a standard LLM.
After the initial forward pass with vision encoding, text generation uses only
the language model - which CAN be batched using the same BatchKVCache pattern.

Architecture:
1. Vision inputs are processed per-request (not batched)
2. Initial VLM forward pass extracts cross-attention states / encoder outputs
3. Language model generation is batched using BatchKVCache (like LLM batching)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .multimodal_processor import MultimodalProcessor
from .vision_embedding_cache import VisionEmbeddingCache

logger = logging.getLogger(__name__)


@dataclass
class MLLMBatchRequest:
    """
    Request data for MLLM batch processing.

    Contains all information needed to process a multimodal request
    within the batch generator.
    """

    uid: int  # Unique identifier within the batch generator
    request_id: str  # External request ID
    prompt: str  # Text prompt
    images: Optional[List[str]] = None  # Image paths/URLs/base64
    videos: Optional[List[str]] = None  # Video inputs
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Processed inputs (set after vision preprocessing)
    input_ids: Optional[mx.array] = None
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Generation state
    num_tokens: int = 0  # Tokens generated so far
    output_tokens: List[int] = field(default_factory=list)

    # Vision state (populated after initial VLM forward pass)
    vision_encoded: bool = False
    cross_attention_states: Optional[Any] = None  # For models that use cross-attention
    encoder_outputs: Optional[Any] = None  # For encoder-decoder models


@dataclass
class MLLMBatchResponse:
    """
    Response from a batch generation step.

    Contains the generated token and metadata for a single request.
    """

    uid: int  # Batch generator UID
    request_id: str  # External request ID
    token: int  # Generated token
    logprobs: mx.array  # Log probabilities
    finish_reason: Optional[str] = None  # "stop", "length", or None
    prompt_cache: Optional[Callable[[], List[Any]]] = None  # Cache extraction function


@dataclass
class MLLMBatch:
    """
    Represents an active batch of MLLM requests.

    Manages the batch state including tokens, caches, and metadata
    for all requests being processed together.
    """

    uids: List[int]
    request_ids: List[str]
    y: mx.array  # Current token(s) for each request [batch_size]
    logprobs: List[mx.array]  # Log probs for each request
    max_tokens: List[int]  # Max tokens per request
    num_tokens: List[int]  # Tokens generated per request
    cache: List[Any]  # BatchKVCache for language model
    requests: List[MLLMBatchRequest]  # Full request data

    def __len__(self) -> int:
        return len(self.uids)

    def filter(self, keep_idx: List[int]) -> None:
        """
        Filter batch to keep only requests at specified indices.

        Args:
            keep_idx: Indices of requests to keep
        """
        self.uids = [self.uids[k] for k in keep_idx]
        self.request_ids = [self.request_ids[k] for k in keep_idx]
        self.logprobs = [self.logprobs[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        self.requests = [self.requests[k] for k in keep_idx]

        keep_idx_array = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx_array]

        # Filter cache entries
        for c in self.cache:
            if hasattr(c, "filter"):
                c.filter(keep_idx_array)

    def extend(self, other: "MLLMBatch") -> None:
        """
        Extend this batch with another batch.

        Args:
            other: Batch to merge into this one
        """
        self.uids.extend(other.uids)
        self.request_ids.extend(other.request_ids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs.extend(other.logprobs)
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        self.requests.extend(other.requests)

        # Extend cache - handle None and incompatible caches
        for c, o in zip(self.cache, other.cache):
            if c is not None and o is not None and hasattr(c, "extend"):
                try:
                    # Only extend if both caches have valid keys
                    if (
                        hasattr(c, "keys")
                        and c.keys is not None
                        and hasattr(o, "keys")
                        and o.keys is not None
                    ):
                        c.extend(o)
                except Exception as e:
                    logger.warning(f"Failed to extend cache: {e}")

    def extract_cache(self, idx: int) -> List[Any]:
        """
        Extract cache for a single request (for caching).

        Args:
            idx: Index of request in batch

        Returns:
            Cache state for that request
        """
        return [c.extract(idx) if hasattr(c, "extract") else None for c in self.cache]


class MLLMBatchStats:
    """Statistics for MLLM batch generation."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.prompt_time: float = 0
        self.generation_tokens: int = 0
        self.generation_time: float = 0
        self.vision_encoding_time: float = 0
        self.num_images_processed: int = 0
        self.peak_memory: float = 0

    @property
    def prompt_tps(self) -> float:
        if self.prompt_time == 0:
            return 0
        return self.prompt_tokens / self.prompt_time

    @property
    def generation_tps(self) -> float:
        if self.generation_time == 0:
            return 0
        return self.generation_tokens / self.generation_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_time": self.prompt_time,
            "prompt_tps": self.prompt_tps,
            "generation_tokens": self.generation_tokens,
            "generation_time": self.generation_time,
            "generation_tps": self.generation_tps,
            "vision_encoding_time": self.vision_encoding_time,
            "num_images_processed": self.num_images_processed,
            "peak_memory": self.peak_memory,
        }


def _make_batch_cache(model: nn.Module, left_padding: List[int]) -> List[Any]:
    """
    Create batch-aware KV cache for the language model.

    Args:
        model: The language model (model.language_model from VLM)
        left_padding: Padding amounts for left-padded prompts

    Returns:
        List of BatchKVCache objects for each layer
    """
    from mlx_lm.models.cache import BatchKVCache, KVCache

    def to_batch_cache(c):
        if isinstance(c, KVCache):
            return BatchKVCache(left_padding)
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        return [BatchKVCache(left_padding) for _ in model.layers]


def _left_pad_prompts(
    prompts: List[List[int]], max_length: Optional[int] = None
) -> mx.array:
    """
    Left-pad prompts to uniform length.

    Args:
        prompts: List of token lists
        max_length: Target length (computed if not provided)

    Returns:
        Padded prompts as mx.array [batch_size, seq_len]
    """
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + list(p) for p in prompts])


class MLLMBatchGenerator:
    """
    Batch generator for Vision Language Models.

    This class manages continuous batching for MLLM requests:

    1. Vision Encoding Phase:
       - Process images/videos through vision encoder (per-request)
       - Extract vision features and merge with text embeddings
       - Store cross-attention states for language model

    2. Language Generation Phase:
       - Use language model with BatchKVCache for batched generation
       - Generate tokens for all requests simultaneously
       - Same pattern as LLM BatchGenerator

    Example:
        >>> generator = MLLMBatchGenerator(model, processor)
        >>> uids = generator.insert([request1, request2])
        >>> while responses := generator.next():
        ...     for resp in responses:
        ...         print(f"Request {resp.request_id}: token={resp.token}")
    """

    # Generation stream for async eval
    _stream = None

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        mm_processor: Optional[MultimodalProcessor] = None,
        max_tokens: int = 256,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        prefill_batch_size: int = 4,  # Smaller for MLLM due to vision overhead
        completion_batch_size: int = 16,  # Can be larger for text generation
        prefill_step_size: int = 1024,
        enable_vision_cache: bool = True,
        vision_cache_size: int = 100,
    ):
        """
        Initialize MLLM batch generator.

        Args:
            model: The VLM model (must have model.language_model)
            processor: The VLM processor for tokenization and image processing
            mm_processor: Optional MultimodalProcessor for input preparation
            max_tokens: Default max tokens per request
            stop_tokens: Set of stop token IDs
            sampler: Sampling function (default: argmax)
            prefill_batch_size: Max requests to prefill together
            completion_batch_size: Max requests for completion batching
            prefill_step_size: Tokens to process per prefill step
            enable_vision_cache: Enable vision embedding caching
            vision_cache_size: Max entries in vision cache
        """
        self.model = model
        self.processor = processor
        self.mm_processor = mm_processor

        # Get language model for text generation
        self.language_model = getattr(model, "language_model", model)

        # Check if this is actually a VLM with separate language model
        self.is_vlm = hasattr(model, "language_model")
        if self.is_vlm:
            logger.info(
                "MLLMBatchGenerator: Using VLM's language_model for batched generation"
            )
        else:
            logger.warning(
                "MLLMBatchGenerator: Model does not have language_model, using model directly"
            )

        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.prefill_step_size = prefill_step_size

        # Request management
        self.unprocessed_requests: List[MLLMBatchRequest] = []
        self.active_batch: Optional[MLLMBatch] = None
        self.uid_counter = 0

        # Statistics
        self._stats = MLLMBatchStats()

        # Vision embedding cache for repeated images
        self.vision_cache = VisionEmbeddingCache(
            max_pixel_entries=vision_cache_size,
            max_encoding_entries=vision_cache_size // 2,
            enabled=enable_vision_cache,
        )
        if enable_vision_cache:
            logger.info(
                f"MLLMBatchGenerator: Vision cache enabled (size={vision_cache_size})"
            )

        # Generation stream
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        # Memory management
        self._old_wired_limit = None
        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.metal.device_info()["max_recommended_working_set_size"]
            )

    def close(self) -> None:
        """Release resources and reset wired limit."""
        if self._old_wired_limit is not None:
            mx.synchronize(MLLMBatchGenerator._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def insert(
        self,
        requests: List[MLLMBatchRequest],
    ) -> List[int]:
        """
        Insert requests for batch processing.

        Args:
            requests: List of MLLMBatchRequest to process

        Returns:
            List of UIDs assigned to requests
        """
        uids = []
        for req in requests:
            req.uid = self.uid_counter
            self.uid_counter += 1
            self.unprocessed_requests.append(req)
            uids.append(req.uid)

        # Sort by estimated complexity (no images = simpler)
        self.unprocessed_requests = sorted(
            self.unprocessed_requests,
            key=lambda x: (
                0 if not x.images and not x.videos else 1,
                len(x.images or []) + len(x.videos or []),
            ),
        )

        logger.debug(f"Inserted {len(requests)} requests, UIDs: {uids}")
        return uids

    def remove(self, uids: List[int]) -> None:
        """
        Remove requests from processing.

        Args:
            uids: List of UIDs to remove
        """
        uid_set = set(uids)

        # Remove from active batch
        if self.active_batch is not None:
            keep_idx = [
                i for i, uid in enumerate(self.active_batch.uids) if uid not in uid_set
            ]
            if keep_idx:
                self.active_batch.filter(keep_idx)
            else:
                self.active_batch = None

        # Remove from unprocessed
        self.unprocessed_requests = [
            r for r in self.unprocessed_requests if r.uid not in uid_set
        ]

    def _preprocess_request(self, request: MLLMBatchRequest) -> None:
        """
        Preprocess a single MLLM request (vision encoding).

        This prepares the inputs by:
        1. Processing images/videos through the processor
        2. Tokenizing the prompt with image tokens
        3. Running vision encoder to get features

        Uses vision cache to skip processing for repeated images.

        Args:
            request: Request to preprocess
        """
        from mlx_vlm.utils import prepare_inputs

        tic = time.perf_counter()

        # Collect all images (including video frames)
        all_images = []

        if request.images:
            from .models.mllm import process_image_input

            for img in request.images:
                try:
                    path = process_image_input(img)
                    all_images.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")

        if request.videos:
            from .models.mllm import (
                process_video_input,
                extract_video_frames_smart,
                save_frames_to_temp,
                DEFAULT_FPS,
                MAX_FRAMES,
            )

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=DEFAULT_FPS,
                        max_frames=MAX_FRAMES,
                    )
                    frame_paths = save_frames_to_temp(frames)
                    all_images.extend(frame_paths)
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")

        # Check pixel cache first
        cached_pixels = self.vision_cache.get_pixel_cache(all_images, request.prompt)
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
            request.extra_kwargs = dict(cached_pixels.extra_kwargs)

            logger.debug(
                f"Pixel cache HIT for request {request.request_id}: "
                f"saved {cached_pixels.processing_time:.2f}s"
            )
            return

        # Cache miss - process images
        # Get model config
        model_config = getattr(self.model, "config", None)
        image_token_index = (
            getattr(model_config, "image_token_index", None) if model_config else None
        )

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            prompts=request.prompt,
            image_token_index=image_token_index,
        )

        request.input_ids = inputs.get("input_ids")
        request.pixel_values = inputs.get("pixel_values")
        request.attention_mask = inputs.get("attention_mask")

        # Extract extra kwargs
        request.extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        request.image_grid_thw = request.extra_kwargs.pop("image_grid_thw", None)

        processing_time = time.perf_counter() - tic

        # Store in pixel cache for future reuse
        if all_images and request.pixel_values is not None:
            self.vision_cache.set_pixel_cache(
                images=all_images,
                prompt=request.prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
                extra_kwargs=request.extra_kwargs,
                processing_time=processing_time,
            )

        self._stats.num_images_processed += len(all_images)
        self._stats.vision_encoding_time += processing_time

        logger.debug(
            f"Preprocessed request {request.request_id}: "
            f"{len(all_images)} images, {request.input_ids.size if request.input_ids is not None else 0} tokens "
            f"({processing_time:.2f}s)"
        )

    def _run_vision_encoding(self, request: MLLMBatchRequest) -> mx.array:
        """
        Run the initial VLM forward pass to encode vision and get first logits.

        This runs the full VLM model (vision + language) on the prompt,
        which encodes the images and prepares the language model cache.

        Args:
            request: Preprocessed request with input_ids and pixel_values

        Returns:
            Logits from the forward pass
        """
        # Build model call kwargs
        kwargs = dict(request.extra_kwargs)

        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.attention_mask is not None:
            kwargs["attention_mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw

        # Run full VLM forward pass
        # This processes vision inputs and fills the language model cache
        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        output = self.model(input_ids, **kwargs)
        request.vision_encoded = True

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            return output.logits
        return output

    def _process_prompts(self, requests: List[MLLMBatchRequest]) -> MLLMBatch:
        """
        Process a batch of requests through vision encoding and initial prefill.

        For MLLM, this is more complex than LLM:
        1. Preprocess each request (tokenize, process images)
        2. Run vision encoding for each request (cannot batch vision yet)
        3. Set up BatchKVCache for language model generation

        Args:
            requests: Requests to process

        Returns:
            MLLMBatch ready for generation
        """
        tic = time.perf_counter()

        # Preprocess all requests
        for req in requests:
            self._preprocess_request(req)

        # Get token sequences and lengths
        input_ids_list = [
            req.input_ids.tolist() if req.input_ids is not None else [0]
            for req in requests
        ]
        lengths = [len(ids) for ids in input_ids_list]
        max_length = max(lengths)
        padding = [max_length - seq_len for seq_len in lengths]

        self._stats.prompt_tokens += sum(lengths)

        # Create batch cache for language model
        batch_cache = _make_batch_cache(self.language_model, padding)

        # Run vision encoding for each request and fill cache
        # This must be done per-request because vision inputs differ
        first_tokens = []
        all_logprobs = []

        for i, req in enumerate(requests):
            # Run full VLM forward pass for this request
            # This fills the cache for layer i with this request's KV states
            with mx.stream(MLLMBatchGenerator._stream):
                logits = self._run_vision_encoding(req)

                # Extract last token logits
                last_logits = logits[:, -1, :]
                logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                sampled = self.sampler(logprobs)

                mx.eval(sampled, logprobs)

                first_tokens.append(sampled.item())
                all_logprobs.append(logprobs.squeeze(0))

            # Now we need to copy the VLM's cache state to the batch cache
            # This is model-specific, but typically the language model cache
            # is accessible via model.language_model or similar
            if hasattr(self.model, "language_model") and hasattr(
                self.model.language_model, "layers"
            ):
                for layer_idx, layer in enumerate(self.model.language_model.layers):
                    if hasattr(layer, "self_attn") and hasattr(
                        layer.self_attn, "cache"
                    ):
                        layer_cache = layer.self_attn.cache
                        if layer_cache is not None and layer_idx < len(batch_cache):
                            # Insert this request's cache into the batch cache at index i
                            if hasattr(batch_cache[layer_idx], "insert_single"):
                                batch_cache[layer_idx].insert_single(i, layer_cache)

        # Create initial y (first generated tokens)
        y = mx.array(first_tokens)

        self._stats.prompt_time += time.perf_counter() - tic

        return MLLMBatch(
            uids=[req.uid for req in requests],
            request_ids=[req.request_id for req in requests],
            y=y,
            logprobs=all_logprobs,
            max_tokens=[req.max_tokens for req in requests],
            num_tokens=[0] * len(requests),
            cache=batch_cache,
            requests=requests,
        )

    def _step(
        self, input_tokens: mx.array, cache: List[Any]
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Run one generation step through the language model.

        Args:
            input_tokens: Input tokens [batch_size, 1] or [batch_size]
            cache: BatchKVCache for the language model

        Returns:
            Tuple of (sampled tokens, logprobs list)
        """
        # Ensure correct shape
        if input_tokens.ndim == 1:
            input_tokens = input_tokens[:, None]

        # Run language model only (not full VLM)
        output = self.language_model(input_tokens, cache=cache)

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        logits = logits[:, -1, :]

        # Sample
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)

        return sampled, list(logprobs)

    def _next(self) -> List[MLLMBatchResponse]:
        """
        Internal next() implementation.

        Returns:
            List of MLLMBatchResponse for this step
        """
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active

        # Try to add more requests from unprocessed queue
        while num_to_add >= self.prefill_batch_size:
            requests = self.unprocessed_requests[: self.prefill_batch_size]

            # Nothing to add and nothing running
            if len(requests) == 0 and num_active == 0:
                self.active_batch = None
                return []

            # Nothing to add but have running requests
            if len(requests) == 0:
                break

            # Finish active generation tokens before prefill
            if batch is not None and not prompt_processing:
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            # Process new batch
            new_batch = self._process_prompts(requests)
            self.unprocessed_requests = self.unprocessed_requests[
                self.prefill_batch_size :
            ]
            prompt_processing = True

            if self.active_batch is None:
                self.active_batch = new_batch
            else:
                self.active_batch.extend(new_batch)

            num_active = len(self.active_batch)
            num_to_add -= len(requests)

        # Generate next token for active batch
        batch = self.active_batch
        if batch is None:
            return []

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()

        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic

        # Build responses and track finished
        keep_idx = []
        end_idx = []
        responses = []

        for i, (token, uid, request_id, num_tok, max_tok, req) in enumerate(
            zip(
                y,
                batch.uids,
                batch.request_ids,
                batch.num_tokens,
                batch.max_tokens,
                batch.requests,
            )
        ):
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            finish_reason = None
            cache_fn = None

            if token in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            if finish_reason is not None:
                # Extract cache for this request
                cache_fn = lambda idx=i: batch.extract_cache(idx)

            responses.append(
                MLLMBatchResponse(
                    uid=uid,
                    request_id=request_id,
                    token=token,
                    logprobs=logprobs[i],
                    finish_reason=finish_reason,
                    prompt_cache=cache_fn,
                )
            )

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def next(self) -> List[MLLMBatchResponse]:
        """
        Generate next token for all requests in the batch.

        Returns:
            List of MLLMBatchResponse, one per active request
        """
        with mx.stream(MLLMBatchGenerator._stream):
            return self._next()

    def stats(self) -> MLLMBatchStats:
        """
        Get generation statistics.

        Returns:
            MLLMBatchStats with timing and token counts
        """
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def get_vision_cache_stats(self) -> Dict[str, Any]:
        """Get vision cache statistics."""
        return self.vision_cache.get_stats()

    def has_pending(self) -> bool:
        """Check if there are pending or active requests."""
        return bool(self.unprocessed_requests or self.active_batch)
