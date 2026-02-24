# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for vllm-mlx continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from .paged_cache import PagedCacheManager
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .spec_decode import (
    AcceptResult,
    RequestState,
    SpecDecodeConfig,
    SpecDecodeRuntime,
    SpecDecodeStats,
    VerifyResult,
)
from .spec_decode.cache_utils import batch_variable_trim, can_per_seq_trim
from .spec_decode.ngram_proposer import NgramProposer
from .spec_decode.rejection_sampler import RejectionSampler
from .utils.mamba_cache import ensure_mamba_support

logger = logging.getLogger(__name__)

# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "cache",
    "BatchKVCache",
]


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting

    # KV cache quantization (reduces prefix cache memory)
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # Paged cache settings (experimental - for memory efficiency)
    use_paged_cache: bool = (
        False  # Use BlockAwarePrefixCache instead of PrefixCacheManager
    )
    paged_cache_block_size: int = 64  # Tokens per block
    max_cache_blocks: int = 1000  # Maximum number of cache blocks

    # Chunked prefill: max tokens to prefill per scheduler step (0 = disabled)
    # When enabled, large prompts are split into chunks so that active
    # generation requests are not starved during long prefills.
    chunked_prefill_tokens: int = 0

    # Mid-prefill cache saving: save intermediate KV cache every N tokens
    # during chunked prefill. If the client disconnects mid-prefill, the
    # saved cache is reused for the next request with the same prefix.
    # 0 = disabled. Only effective when chunked_prefill_tokens > 0.
    mid_prefill_save_interval: int = 8192

    # Speculative decoding settings
    speculative_method: Optional[str] = (
        None  # None = disabled, "ngram" = n-gram proposer
    )
    num_speculative_tokens: int = 3  # Number of draft tokens per step (k)
    spec_decode_disable_batch_size: Optional[int] = (
        None  # Disable spec decode above this batch size
    )
    draft_model_name: Optional[str] = (
        None  # Draft model path (for future model-based proposers)
    )
    spec_decode_auto_disable_threshold: float = (
        0.4  # Auto-disable below this acceptance rate
    )
    spec_decode_auto_disable_window: int = (
        50  # Rolling window size for auto-disable evaluation
    )
    model_name: Optional[str] = None  # Model name/path for MTP weight loading
    mtp_model_name: Optional[str] = (
        None  # Separate model for MTP weights (if different from main)
    )


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: List[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: Set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: List[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False


def _inner_cache(layer_cache):
    """Get the first inner cache from a CacheList, or return as-is.

    Used for introspecting batch state (offset, left_padding, _idx) which
    is shared across sub-caches in a CacheList.
    """
    if hasattr(layer_cache, "caches"):
        return layer_cache.caches[0]
    return layer_cache


def _install_chunked_prefill(
    batch_gen: "BatchGenerator",
    budget: int,
    mid_prefill_save=None,
    prompt_cache_save=None,
    pending_abort_ids: Optional[Set[str]] = None,
    uid_to_request_id: Optional[Dict[int, str]] = None,
    requests: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Monkey-patch a BatchGenerator instance so that large prefills are
    broken into chunks of at most *budget* tokens each.

    Between chunks the generation loop gets a chance to produce one token
    for every active request, preventing starvation during long prefills.

    Args:
        batch_gen: The BatchGenerator to patch.
        budget: Max tokens per prefill chunk.
        mid_prefill_save: Optional callback(uid, processed, prompt_cache)
            called after each chunk to save intermediate KV cache state.
    """
    import time as _time

    from mlx_lm.generate import (
        Batch,
        _left_pad_prompts,
        _make_cache,
        _merge_caches,
        _right_pad_prompts,
    )

    # Keep references to originals
    _orig_next = batch_gen._next
    _orig_remove = batch_gen.remove
    _orig_process_prompts = batch_gen._process_prompts

    # Partial prefill state (None when no prefill in progress)
    batch_gen._partial = None

    # Monkey-patch _process_prompts to capture prompt-only cache state.
    # At the point where _process_prompts returns, the Batch cache contains
    # the exact prompt-only state: all prompt tokens have been processed
    # through the model, but no output token has been fed back yet.
    # This is the only safe capture point for hybrid Mamba+Transformer
    # models whose MambaCache state is cumulative.
    if prompt_cache_save is not None:

        def _patched_process_prompts(prompts, _self=batch_gen):
            batch = _orig_process_prompts(prompts)
            for e, uid in enumerate(batch.uids):
                if batch.num_tokens[e] == 0:
                    try:
                        prompt_cache_save(uid, batch.extract_cache(e))
                    except Exception:
                        pass
            return batch

        batch_gen._process_prompts = _patched_process_prompts

    def _generation_step(self=batch_gen):
        """Run one generation step on the active batch. Returns responses."""
        batch = self.active_batch
        if batch is None or len(batch) == 0:
            return []

        tic_gen = _time.perf_counter()
        y, logprobs = batch.y, batch.logprobs
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i : i + 1]))
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        self._stats.generation_time += _time.perf_counter() - tic_gen

        keep_idx = []
        end_idx = []
        responses = []
        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache_out = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            if finish_reason is not None:
                cache_out = batch.extract_cache(e)
            responses.append(
                self.Response(uid, t, logprobs[e], finish_reason, cache_out)
            )

        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def _chunked_next(self=batch_gen):  # noqa: C901
        """
        Replacement for _next() that chunks large prefills.

        Only intercepts when:
        1. A partial prefill is in progress (_partial is not None)
        2. The next prompt batch exceeds the budget

        Everything else delegates to the original _next().
        """
        # ----- Continue a partial prefill -----
        if self._partial is not None:
            # Check for pending aborts BEFORE processing next chunk
            if pending_abort_ids is not None and uid_to_request_id is not None:
                partial_rids = {uid_to_request_id.get(u) for u in self._partial["uids"]}
                aborted_rids = partial_rids & pending_abort_ids
                if aborted_rids:
                    logger.info(
                        f"[chunked_prefill] abort detected mid-prefill, "
                        f"clearing partial for: {aborted_rids}"
                    )
                    self._partial = None
                    mx.clear_cache()
                    return _generation_step()

            tic = _time.perf_counter()
            partial = self._partial
            inputs = partial["inputs"]
            prompt_cache = partial["cache"]
            remaining = inputs.shape[1]

            n_to_process = min(budget, remaining - 1) if remaining > 1 else 0

            if n_to_process > 0:
                self.model(inputs[:, :n_to_process], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                partial["inputs"] = inputs
                partial["processed"] += n_to_process

                self.prompt_progress_callback(
                    [
                        (uid, partial["processed"], partial["total"])
                        for uid in partial["uids"]
                    ]
                )

                # Save intermediate cache for disconnect resilience
                if mid_prefill_save is not None and len(partial["uids"]) == 1:
                    mid_prefill_save(
                        partial["uids"][0], partial["processed"], prompt_cache
                    )

                if partial.get("is_cached"):
                    mx.clear_cache()

            # Check if prefill is done (only 1 token left or 0)
            if inputs.shape[1] <= 1:
                # Finalize
                if partial.get("is_cached"):
                    mx.eval([c.state for c in prompt_cache])
                    inputs = partial["last_inputs"]

                for c in prompt_cache:
                    c.finalize()
                mx.clear_cache()

                y, logprobs = self._step(
                    inputs,
                    prompt_cache,
                    partial["samplers"],
                    partial["logits_processors"],
                    partial["tokens"],
                )
                mx.async_eval(y, logprobs)

                new_batch = Batch(
                    list(partial["uids"]),
                    y,
                    logprobs,
                    list(partial["max_tokens"]),
                    [0] * len(partial["uids"]),
                    prompt_cache,
                    list(partial["samplers"]),
                    list(partial["logits_processors"]),
                    partial["tokens"],
                )

                # Save prompt-only cache BEFORE merging into active batch.
                # This is the chunked-prefill equivalent of the
                # _patched_process_prompts hook — at this point the cache
                # contains the exact prompt-only state (num_tokens == 0).
                if prompt_cache_save is not None and len(partial["uids"]) == 1:
                    uid = partial["uids"][0]
                    try:
                        prompt_cache_save(uid, new_batch.extract_cache(0))
                    except Exception:
                        pass

                if self.active_batch is None:
                    self.active_batch = new_batch
                else:
                    self.active_batch.extend(new_batch)

                self._partial = None
                self._stats.prompt_time += _time.perf_counter() - tic
            else:
                # Not done yet — record prompt time for this chunk
                self._stats.prompt_time += _time.perf_counter() - tic

            # Generation step for active requests between chunks
            return _generation_step()

        # ----- No partial — check if next prompt batch needs chunking -----
        num_active = len(self.active_batch) if self.active_batch else 0
        num_to_add = self.completion_batch_size - num_active

        if num_to_add >= self.prefill_batch_size and self.unprocessed_prompts:
            batch_prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            if batch_prompts:
                total_tokens = sum(len(p[1]) for p in batch_prompts)

                # Check if any prompt has a prefix_boundary that
                # requires two-phase prefill for cache save at that boundary.
                _needs_boundary_split = False
                if requests is not None and uid_to_request_id is not None:
                    for _uid, _toks, *_ in batch_prompts:
                        _rid = uid_to_request_id.get(_uid)
                        _req = requests.get(_rid) if _rid else None
                        if _req and getattr(_req, "prefix_boundary", 0) > 0:
                            _needs_boundary_split = True
                            break

                if total_tokens > budget or _needs_boundary_split:
                    # Large prompt batch or prefix boundary — start partial prefill
                    tic = _time.perf_counter()

                    # Eval outstanding generation tokens before switching
                    if self.active_batch is not None:
                        mx.eval(self.active_batch.y, self.active_batch.logprobs)
                        self._stats.generation_time += _time.perf_counter() - tic
                        tic = _time.perf_counter()

                    (
                        uids,
                        inputs_raw,
                        max_tokens_list,
                        caches,
                        samplers,
                        logits_processors,
                    ) = zip(*batch_prompts)
                    lengths = [len(p) for p in inputs_raw]
                    max_length = max(lengths)
                    padding = [max_length - ln for ln in lengths]
                    tokens = [mx.array(inp) for inp in inputs_raw]
                    is_cached = not all(c[0].empty() for c in caches)

                    self._stats.prompt_tokens += sum(lengths)

                    if not is_cached:
                        padded = _left_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _make_cache(self.model, padding)
                    else:
                        last_inputs = mx.array([p[-1:] for p in inputs_raw])
                        padded = _right_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _merge_caches(caches)
                        for c in prompt_cache:
                            c.prepare(
                                lengths=[ln - 1 for ln in lengths],
                                right_padding=padding,
                            )

                    # Remove from unprocessed
                    self.unprocessed_prompts = self.unprocessed_prompts[
                        self.prefill_batch_size :
                    ]

                    # Process first chunk — if prefix_boundary is set,
                    # use it as the first chunk size so that mid_prefill_save
                    # can capture the exact prefix cache state (critical for
                    # hybrid Mamba+Transformer models where trim is unsafe).
                    # When the request already has cached tokens (cache hit),
                    # adjust the boundary relative to the remaining tokens.
                    _first_chunk = budget
                    if _needs_boundary_split and len(batch_prompts) == 1:
                        _uid0 = uids[0]
                        _rid0 = uid_to_request_id.get(_uid0)
                        _req0 = requests.get(_rid0) if _rid0 else None
                        _pb = getattr(_req0, "prefix_boundary", 0) if _req0 else 0
                        _cached = getattr(_req0, "cached_tokens", 0) if _req0 else 0
                        _adjusted_pb = _pb - _cached
                        if 0 < _adjusted_pb < padded.shape[1]:
                            _first_chunk = _adjusted_pb
                    n_to_process = min(_first_chunk, padded.shape[1] - 1)
                    if n_to_process > 0:
                        self.model(padded[:, :n_to_process], cache=prompt_cache)
                        mx.eval([c.state for c in prompt_cache])
                        padded = padded[:, n_to_process:]
                        if is_cached:
                            mx.clear_cache()

                    self._partial = {
                        "uids": list(uids),
                        "inputs": padded,
                        "cache": prompt_cache,
                        "tokens": tokens,
                        "max_tokens": list(max_tokens_list),
                        "samplers": list(samplers),
                        "logits_processors": list(logits_processors),
                        "processed": n_to_process,
                        "total": max_length,
                        "is_cached": is_cached,
                    }
                    if is_cached:
                        self._partial["last_inputs"] = last_inputs

                    self.prompt_progress_callback(
                        [
                            (uid, n_to_process, max_length)
                            for uid in self._partial["uids"]
                        ]
                    )

                    # Save intermediate cache for disconnect resilience
                    if mid_prefill_save is not None and len(uids) == 1:
                        mid_prefill_save(uids[0], n_to_process, prompt_cache)

                    self._stats.prompt_time += _time.perf_counter() - tic

                    # Generation step for active requests
                    return _generation_step()

        # Small prompts, pure generation, or no work — delegate to original
        return _orig_next()

    def _patched_remove(uids_to_remove, _self=batch_gen):
        """Clear partial state if aborted request is being prefilled."""
        if _self._partial is not None:
            partial_uids = set(_self._partial["uids"])
            if partial_uids & set(uids_to_remove):
                logger.info(
                    f"[chunked_prefill] clearing partial state for aborted uids: "
                    f"{partial_uids & set(uids_to_remove)}"
                )
                _self._partial = None
                mx.clear_cache()  # flush Metal encoders after dropping partial state
        _orig_remove(uids_to_remove)

    batch_gen._next = _chunked_next
    batch_gen.remove = _patched_remove

    logger.info(f"[chunked_prefill] installed with budget={budget} tokens per step")


def _resolve_mtp_model_path(model_name: str, num_hidden_layers: int) -> str:
    """Resolve an HF model name to a local path, downloading only MTP-related shards.

    Instead of downloading the entire model (which can be 600GB+), this
    downloads only the safetensors index and the specific shard files that
    contain MTP layer weights.

    Args:
        model_name: HuggingFace model name (e.g. 'deepseek-ai/DeepSeek-V3-0324').
        num_hidden_layers: Number of hidden layers in the main model.

    Returns:
        Local directory path containing the downloaded MTP weight files.
    """
    import json

    from huggingface_hub import hf_hub_download

    # Known MTP prefix patterns
    mtp_prefixes = [
        f"model.layers.{num_hidden_layers}.",  # DeepSeek V3/V3.2, GLM
        "model.mtp_layers.",  # MiMo
        "model.mtp.",  # Kimi, MiMo V2
    ]

    # Step 1: Download the safetensors index
    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    except Exception:
        # No index file — try downloading entire model as fallback
        logger.warning(
            "No safetensors index found for '%s', falling back to snapshot_download",
            model_name,
        )
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name)

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Step 2: Find shard files containing MTP weights (try all known prefixes)
    mtp_files: set[str] = set()
    for prefix in mtp_prefixes:
        for key, filename in weight_map.items():
            if key.startswith(prefix):
                mtp_files.add(filename)
        if mtp_files:
            break

    if not mtp_files:
        raise ValueError(
            f"No MTP weights found in {model_name}'s weight map. "
            f"Tried prefixes: {mtp_prefixes}. "
            f"This model may not have MTP layers."
        )

    logger.info(
        "MTP weights span %d shard files out of %d total — downloading selectively",
        len(mtp_files),
        len(set(weight_map.values())),
    )

    # Step 3: Download only the needed shard files
    for filename in sorted(mtp_files):
        logger.info("Downloading MTP shard: %s", filename)
        hf_hub_download(model_name, filename)

    # Return the directory containing the downloaded files
    # hf_hub_download caches to the same directory structure
    from pathlib import Path

    cache_dir = Path(index_path).parent
    return str(cache_dir)


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: Optional[BatchGenerator] = None
        self._current_sampler_params: Optional[Tuple] = None

        self._mtp_hidden_states: dict[str, Any] = (
            {}
        )  # Per-request hidden states for MTP

        # Prefix cache for KV state reuse
        self.prefix_cache: Optional[PrefixCacheManager] = None
        self.memory_aware_cache: Optional[MemoryAwarePrefixCache] = None
        self.paged_cache_manager: Optional[PagedCacheManager] = None
        self.block_aware_cache: Optional[BlockAwarePrefixCache] = None

        if self.config.enable_prefix_cache:
            if self.config.use_paged_cache:
                # Use paged cache for memory efficiency
                self.paged_cache_manager = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.config.max_cache_blocks,
                )
                self.block_aware_cache = BlockAwarePrefixCache(
                    model=model,
                    paged_cache_manager=self.paged_cache_manager,
                )
                logger.info(
                    f"Paged cache enabled: block_size={self.config.paged_cache_block_size}, "
                    f"max_blocks={self.config.max_cache_blocks}"
                )
            elif self.config.use_memory_aware_cache:
                # Use memory-aware cache (recommended for large models)
                cache_config = MemoryCacheConfig(
                    max_memory_mb=self.config.cache_memory_mb,
                    max_memory_percent=self.config.cache_memory_percent,
                    kv_quantize=self.config.kv_cache_quantization,
                    kv_bits=self.config.kv_cache_quantization_bits,
                    kv_group_size=self.config.kv_cache_quantization_group_size,
                    kv_min_quantize_tokens=self.config.kv_cache_min_quantize_tokens,
                )
                self.memory_aware_cache = MemoryAwarePrefixCache(
                    model=model,
                    config=cache_config,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
                )
            else:
                # Use legacy entry-count based prefix cache
                self.prefix_cache = PrefixCacheManager(
                    model=model,
                    max_entries=self.config.prefix_cache_size,
                )
                logger.info(
                    f"Prefix cache enabled with max_entries={self.config.prefix_cache_size}"
                )

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: Set[str] = set()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Memory management: periodic mx.clear_cache() to free Metal command buffers
        # Lower interval = less VRAM spike during generation but slight throughput cost
        self._step_count = 0
        self._clear_cache_interval = 32
        self._memory_log_interval = 256

        # Speculative decoding (initialized lazily on first spec-eligible step)
        self._spec_decode_runtime: Optional[SpecDecodeRuntime] = None
        self._spec_decode_enabled = self.config.speculative_method is not None
        if self._spec_decode_enabled:
            self._init_spec_decode()

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        """
        Get the actual tokenizer from a processor or tokenizer.

        MLLM models use processors (e.g., Qwen3VLProcessor) which wrap
        the tokenizer. This method extracts the actual tokenizer.
        """
        # If it has encode method, it's already a tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        # If it's a processor, get the wrapped tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        # Fallback to the original
        return tokenizer

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids)

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer or processor."""
        stop_tokens = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    # Handle case where eos_token_ids is a single int
                    stop_tokens.add(tok.eos_token_ids)
        return stop_tokens

    # -----------------------------------------------------------------
    # Speculative decoding
    # -----------------------------------------------------------------

    def _init_spec_decode(self) -> None:
        """Initialize speculative decoding components."""
        spec_config = SpecDecodeConfig(
            method=self.config.speculative_method,
            num_speculative_tokens=self.config.num_speculative_tokens,
            disable_by_batch_size=self.config.spec_decode_disable_batch_size,
            auto_disable_threshold=self.config.spec_decode_auto_disable_threshold,
            auto_disable_window=self.config.spec_decode_auto_disable_window,
        )

        # Create proposer based on method
        if self.config.speculative_method == "ngram":
            from .spec_decode.ngram_proposer import NgramProposer, NgramProposerConfig

            proposer_config = NgramProposerConfig(
                num_speculative_tokens=self.config.num_speculative_tokens,
            )
            proposer = NgramProposer(proposer_config)
        elif self.config.speculative_method == "draft_model":
            from .spec_decode.draft_model_proposer import (
                DraftModelProposer,
                DraftModelProposerConfig,
            )

            if not self.config.draft_model_name:
                raise ValueError(
                    "draft_model_name must be set when using speculative_method='draft_model'"
                )
            proposer_config = DraftModelProposerConfig(
                num_speculative_tokens=self.config.num_speculative_tokens,
                draft_model_name=self.config.draft_model_name,
            )
            proposer = DraftModelProposer(proposer_config)
            proposer.load()
        elif self.config.speculative_method == "mtp":
            # External MTP adapters (DeepSeek V3, GLM-5, etc.)
            from .spec_decode.mtp_module import (
                detect_mtp_prefix,
                detect_mtp_style,
                load_mtp_weights,
            )
            from .spec_decode.mtp_proposer import MTPProposer, MTPProposerConfig

            model_config = self.model.args

            # Resolve MTP weight source
            mtp_source = self.config.mtp_model_name or self.config.model_name
            if not mtp_source:
                raise ValueError(
                    "model_name or mtp_model_name must be set in SchedulerConfig "
                    "for MTP weight loading"
                )
            import os

            if os.path.isdir(mtp_source):
                mtp_model_path = mtp_source
            else:
                mtp_model_path = _resolve_mtp_model_path(
                    mtp_source, model_config.num_hidden_layers
                )

            # Auto-detect MTP prefix and style from weights
            mtp_prefix, mtp_keys = detect_mtp_prefix(
                mtp_model_path, model_config.num_hidden_layers
            )
            mtp_style = detect_mtp_style(mtp_keys, mtp_prefix)
            logger.info(
                "Detected MTP style='%s' with prefix='%s'",
                mtp_style,
                mtp_prefix,
            )

            # Get decoder layer class from the loaded model
            decoder_layer_cls = type(self.model.model.layers[0])

            # Create appropriate MTP module based on detected style
            if mtp_style == "standard":
                from .spec_decode.mtp_adapters import StandardMTPModule

                mtp_module = StandardMTPModule.from_model_config(
                    model_config,
                    decoder_layer_cls=decoder_layer_cls,
                )
            else:
                from .spec_decode.mtp_adapters import SimpleMTPModule

                mtp_module = SimpleMTPModule.from_model_config(
                    model_config,
                    decoder_layer_cls=decoder_layer_cls,
                )

            # Load MTP weights (and quantize to match main model if needed)
            load_mtp_weights(
                mtp_model_path,
                model_config.num_hidden_layers,
                mtp_module,
                model_config=model_config,
                mtp_prefix=mtp_prefix,
                main_model=self.model,
            )

            # Create proposer with shared embed_fn and lm_head
            proposer_config = MTPProposerConfig(
                num_speculative_tokens=self.config.num_speculative_tokens,
            )
            proposer = MTPProposer(
                config=proposer_config,
                mtp_module=mtp_module,
                embed_fn=self.model.model.embed_tokens,
                lm_head=self.model.lm_head,
            )
        else:
            raise ValueError(
                f"Unknown speculative method: {self.config.speculative_method}"
            )

        # Create rejection sampler (greedy for now, stochastic needs draft model logits)
        sampler = RejectionSampler(method="greedy")

        self._spec_decode_runtime = SpecDecodeRuntime(
            model=self.model,
            proposer=proposer,
            rejection_sampler=sampler,
            config=spec_config,
        )
        logger.info(
            f"Speculative decoding enabled: method={self.config.speculative_method}, "
            f"k={self.config.num_speculative_tokens}"
        )

    def _can_spec_decode(self) -> bool:
        """Check if speculative decoding can run this step.

        Returns False when:
        - Spec decode is not enabled
        - No active batch exists
        - There are waiting requests (prefill needed -- don't mix with speculation)
        - Batch size exceeds threshold
        - Model is not transformer-only (Mamba/hybrid not supported)
        """
        if not self._spec_decode_enabled or self._spec_decode_runtime is None:
            return False
        if self.batch_generator is None:
            return False
        if self.batch_generator.active_batch is None:
            return False
        if self.waiting:  # Pending prefills -- run normal step
            return False
        # Don't spec decode if there are unprocessed prompts waiting for prefill
        if (
            hasattr(self.batch_generator, "unprocessed_prompts")
            and self.batch_generator.unprocessed_prompts
        ):
            return False
        # Check batch size threshold
        batch_size = len(self.running)
        if self._spec_decode_runtime.should_disable(batch_size):
            return False
        # Check for transformer-only model (Mamba/hybrid not supported for spec decode)
        if hasattr(self.model, "args") and hasattr(self.model.args, "model_type"):
            model_type = self.model.args.model_type
            if model_type in ("mamba", "jamba", "nemotron"):
                return False
        # Check cache supports per-sequence trim (CacheList layers may not)
        batch = self.batch_generator.active_batch
        if batch is not None and not can_per_seq_trim(batch.cache):
            return False
        return True

    def _step_spec_decode(self) -> Optional[list]:
        """Execute one speculative decoding step.

        Flow:
        1. Build request states from running requests
        2. Propose k draft tokens per request via NgramProposer
        3. Build input tensor: [y, d1, ..., dk] per request
        4. Run target model forward: logits = model(input_tokens, cache)
        5. Build VerifyResult with per-request sliced logits
        6. Run rejection sampling via runtime.accept_and_commit()
        7. Build Response objects for each committed token
        8. Rollback KV cache for rejected draft positions
        9. Update batch state for next step

        Returns:
            List of Response-like objects, or None if no drafts were
            generated (caller should fall back to the normal path).
        """
        batch = self.batch_generator.active_batch
        runtime = self._spec_decode_runtime
        k = runtime.config.num_speculative_tokens

        # 1. Build request states
        batch_y_list = batch.y.tolist()
        request_states = {}
        uid_to_request_id_local = {}
        for i, uid in enumerate(batch.uids):
            request_id = self.uid_to_request_id.get(uid)
            if request_id is None:
                continue
            request = self.running.get(request_id)
            if request is None:
                continue
            token_ids = (
                list(request.prompt_token_ids or [])
                + list(request.output_token_ids)
                + [batch_y_list[i]]
            )
            request_states[request_id] = RequestState(
                request_id=request_id,
                token_ids=token_ids,
                batch_uid=uid,
                hidden_states=self._mtp_hidden_states.get(request_id),
            )
            uid_to_request_id_local[uid] = request_id

        if not request_states:
            return []

        # 2. Propose drafts
        draft_metadata = runtime.propose_drafts(request_states)

        # 3. Build input tensor for verification
        # For each request: [y_i, draft_1, ..., draft_k] (padded to max_k+1)
        # batch.y contains the last sampled token (not yet fed to model)
        batch_y = batch.y.tolist()  # (B,)

        # Map request order to batch indices
        batch_request_ids = []
        batch_indices = []
        for i, uid in enumerate(batch.uids):
            rid = uid_to_request_id_local.get(uid)
            if rid and rid in request_states:
                batch_request_ids.append(rid)
                batch_indices.append(i)

        if not batch_request_ids:
            return []

        # Build padded input tokens: shape (len(batch_request_ids), max_k + 1)
        max_draft_len = 0
        draft_per_request = {}
        for rid in batch_request_ids:
            drafts = draft_metadata.get_draft_tokens(rid)
            draft_per_request[rid] = drafts
            max_draft_len = max(max_draft_len, len(drafts))

        if max_draft_len == 0 and self.config.speculative_method != "mtp":
            # No drafts generated -- fall back to normal step
            # (MTP continues to capture hidden states for bootstrap)
            return None  # Signal caller to use normal path

        input_rows = []
        draft_lengths = []
        for rid, batch_idx in zip(batch_request_ids, batch_indices):
            y_token = batch_y[batch_idx]
            drafts = draft_per_request[rid]
            # Pad drafts to max_draft_len with 0 (causal mask makes padding harmless)
            padded_drafts = list(drafts) + [0] * (max_draft_len - len(drafts))
            row = [y_token] + padded_drafts  # length = max_draft_len + 1
            input_rows.append(row)
            draft_lengths.append(len(drafts))

        input_tokens = mx.array(input_rows, dtype=mx.int32)  # (B_spec, max_k+1)

        # 4. Run target model forward pass
        # Feed the full [y, d1, ..., dk] sequence through the model with
        # the existing KV cache. The model returns logits for each position.
        _is_mtp = self.config.speculative_method == "mtp"
        if _is_mtp:
            # Capture PRE-NORM hidden states for MTP.
            # model.model.__call__() applies self.norm(h) at the end,
            # but MTP's hnorm expects raw decoder layer output.
            inner = self.model.model  # DeepseekV32Model (or equivalent backbone)
            h = inner.embed_tokens(input_tokens)
            # Build attention mask using the first layer's cache offset.
            # batch.cache[0] may be CacheList (deepseek_v32/glm_moe_dsa)
            # or a plain KVCache; [0] accesses first KVCache in CacheList.
            _c0 = batch.cache[0]
            try:
                _cache_for_mask = _c0[0] if _c0 else None
            except (TypeError, IndexError):
                _cache_for_mask = _c0
            from mlx_lm.models.base import create_attention_mask

            mask = create_attention_mask(h, _cache_for_mask, return_array=True)
            for i in range(inner.num_layers):
                h = inner.layers[inner.start_idx + i](h, mask, batch.cache[i])
            hidden_states = h  # Pre-norm hidden states for MTP
            logits = self.model.lm_head(inner.norm(h))
        else:
            logits = self.model(input_tokens, cache=batch.cache)
            hidden_states = None
        # logits shape: (B_spec, max_k+1, vocab_size)

        mx.eval(logits)

        # 5. Build VerifyResult
        verify_result = VerifyResult()
        verify_result.request_ids = list(batch_request_ids)

        for i, rid in enumerate(batch_request_ids):
            num_drafts = draft_lengths[i]
            # Slice logits for this request: positions 0..num_drafts (inclusive)
            # Position j verifies draft token j (0-indexed)
            # Position num_drafts is the bonus position
            req_logits = logits[i, : num_drafts + 1, :]  # (num_drafts+1, vocab)
            verify_result.target_logits[rid] = req_logits

        # 6. Run rejection sampling
        accept_results = runtime.accept_and_commit(verify_result, draft_metadata)

        # 6a. Store hidden states for MTP (at accepted positions)
        if _is_mtp and hidden_states is not None:
            for i, rid in enumerate(batch_request_ids):
                result = accept_results.get(rid)
                if result is not None:
                    # The accepted position index in the logits/hidden tensor
                    accepted_pos = result.num_accepted
                    # Extract hidden state at the accepted position (shape: 1, 1, hidden_size)
                    self._mtp_hidden_states[rid] = hidden_states[
                        i : i + 1, accepted_pos : accepted_pos + 1, :
                    ]

        # Log spec decode stats periodically
        if runtime.stats.num_drafts % 50 == 0 and runtime.stats.num_drafts > 0:
            logger.info(
                f"[SpecDecode] steps={runtime.stats.num_drafts}, "
                f"alpha={runtime.stats.acceptance_rate():.3f}, "
                f"mean_accepted={runtime.stats.mean_accepted_length():.2f}/{k}, "
                f"per_pos={[f'{r:.2f}' for r in runtime.stats.acceptance_rate_per_position]}"
            )

        # 7. Build Response objects and apply stop/max-token checks
        responses = []
        stop_tokens = self._get_stop_tokens()

        rollback_counts = {}
        finished_in_spec = []
        canonical_committed = {}  # rid -> list of actually emitted token IDs

        for i, rid in enumerate(batch_request_ids):
            batch_idx = batch_indices[i]
            uid = batch.uids[batch_idx]
            result = accept_results.get(rid)
            if result is None:
                continue

            request = self.running.get(rid)
            if request is None:
                continue

            tokens_remaining = (
                request.sampling_params.max_tokens - request.num_output_tokens
            )

            # Build committed tokens: old batch.y + accepted drafts (excluding
            # the final bonus/correction token which becomes new batch.y).
            committed_tokens = (
                [batch_y[batch_idx]] + result.accepted_tokens[:-1]
                if result.accepted_tokens
                else []
            )

            emitted_for_rid = []

            for t_idx, token in enumerate(committed_tokens):
                if tokens_remaining <= 0:
                    unemitted = len(committed_tokens) - t_idx
                    rollback_counts[rid] = rollback_counts.get(rid, 0) + unemitted
                    break

                finish_reason = None
                if token in stop_tokens:
                    finish_reason = "stop"
                elif tokens_remaining == 1:
                    finish_reason = "length"

                resp = type(
                    "Response",
                    (),
                    {
                        "uid": uid,
                        "token": token,
                        "finish_reason": finish_reason,
                        "prompt_cache": None,
                    },
                )()

                responses.append(resp)
                emitted_for_rid.append(token)
                tokens_remaining -= 1

                if finish_reason is not None:
                    unemitted = len(committed_tokens) - t_idx - 1
                    if unemitted > 0:
                        rollback_counts[rid] = rollback_counts.get(rid, 0) + unemitted
                    finished_in_spec.append(rid)
                    break

            canonical_committed[rid] = emitted_for_rid
            # Add original rollback count from rejection
            rollback_counts[rid] = rollback_counts.get(rid, 0) + result.rollback_count

        # 8. Rollback KV cache for rejected positions
        # trim_per_sequence expects an mx.array of per-sequence trim amounts
        trim_amounts = []
        for batch_idx in range(len(batch.uids)):
            uid = batch.uids[batch_idx]
            rid = uid_to_request_id_local.get(uid)
            if rid and rid in rollback_counts:
                # Also account for padding: max_draft_len - actual_draft_len
                actual_drafts = (
                    draft_lengths[batch_request_ids.index(rid)]
                    if rid in batch_request_ids
                    else 0
                )
                padding_trim = max_draft_len - actual_drafts
                total_trim = rollback_counts[rid] + padding_trim
                trim_amounts.append(total_trim)
            else:
                # Requests not in spec decode path: trim the padding only
                trim_amounts.append(max_draft_len)

        if any(t > 0 for t in trim_amounts):
            trim_array = mx.array(trim_amounts, dtype=mx.int32)
            batch_variable_trim(batch.cache, trim_array)
            # Materialize and fix _idx after trim
            if batch.cache:
                _c0 = _inner_cache(batch.cache[0])
                mx.eval(_c0.offset, _c0.left_padding)
            from vllm_mlx.spec_decode.cache_utils import fixup_cache_after_filter

            fixup_cache_after_filter(batch.cache)

        # 9. Update batch state for next step
        # After spec decode, we need batch.y and batch.logprobs set correctly
        # for the NEXT normal step. batch.y = the last committed token for
        # each request. batch.logprobs = logits at the correction/bonus
        # position, normalized.

        new_y = []
        new_logprobs = []
        for batch_idx in range(len(batch.uids)):
            uid = batch.uids[batch_idx]
            rid = uid_to_request_id_local.get(uid)
            if rid and rid in accept_results:
                result = accept_results[rid]
                i = batch_request_ids.index(rid)

                if result.accepted_tokens:
                    # Last committed token becomes new y
                    new_y.append(result.accepted_tokens[-1])
                    # Logprobs at the position after last accepted token
                    # This is the correction/bonus position
                    correction_pos = result.num_accepted  # 0-indexed position
                    req_logits = verify_result.target_logits[rid]
                    if correction_pos < req_logits.shape[0]:
                        log_probs = mx.softmax(req_logits[correction_pos], axis=-1)
                        log_probs = mx.log(log_probs + 1e-12)
                        new_logprobs.append(log_probs)
                    else:
                        # Fallback: use last position
                        log_probs = mx.softmax(req_logits[-1], axis=-1)
                        log_probs = mx.log(log_probs + 1e-12)
                        new_logprobs.append(log_probs)
                else:
                    # No tokens accepted -- keep original y
                    new_y.append(batch_y[batch_idx])
                    new_logprobs.append(
                        batch.logprobs[batch_idx] if batch.logprobs else mx.zeros((1,))
                    )
            else:
                new_y.append(batch_y[batch_idx])
                new_logprobs.append(
                    batch.logprobs[batch_idx] if batch.logprobs else mx.zeros((1,))
                )

        batch.y = mx.array(new_y, dtype=mx.int32)
        batch.logprobs = new_logprobs

        # Update tokens list using canonical committed (post-clipping)
        for rid in accept_results:
            if rid in batch_request_ids and rid in canonical_committed:
                i = batch_request_ids.index(rid)
                batch_idx = batch_indices[i]
                emitted = canonical_committed[rid]
                if emitted:
                    batch.tokens[batch_idx] = mx.concatenate(
                        (batch.tokens[batch_idx], mx.array(emitted))
                    )
                batch.num_tokens[batch_idx] += len(emitted)

        mx.eval(batch.y, *batch.tokens)

        return responses

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
        )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        def _prefill_progress(progress_list):
            """Log prefill progress for each uid chunk."""
            for uid, processed, total in progress_list:
                rid = self.uid_to_request_id.get(uid, "?")
                logger.info(
                    f"[prefill] request={rid[:12] if isinstance(rid, str) else rid} "
                    f"tokens={processed}/{total}"
                )

        bg = BatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            prompt_progress_callback=_prefill_progress,
        )

        # Install chunked prefill when explicitly configured OR when
        # memory-aware cache is active (needed for prefix_boundary saves
        # in agentic multi-turn workloads with hybrid Mamba+Transformer models).
        chunked_budget = self.config.chunked_prefill_tokens
        need_chunked = chunked_budget > 0 or self.memory_aware_cache is not None
        if need_chunked:
            if chunked_budget <= 0:
                # No explicit budget — use a very large value so normal
                # prompts pass through unchanged.  Prefix boundary splits
                # still trigger via _needs_boundary_split.
                chunked_budget = 999_999
            mid_prefill_cb = None
            save_interval = self.config.mid_prefill_save_interval
            if save_interval > 0 and self.memory_aware_cache is not None:
                mid_prefill_cb = self._make_mid_prefill_save_callback(save_interval)
                logger.info(f"[mid_prefill_cache] enabled, interval={save_interval}")
            prompt_cache_cb = None
            if self.memory_aware_cache is not None:
                prompt_cache_cb = self._make_prompt_cache_save_callback()
            _install_chunked_prefill(
                bg,
                chunked_budget,
                mid_prefill_cb,
                prompt_cache_save=prompt_cache_cb,
                pending_abort_ids=self._pending_abort_ids,
                uid_to_request_id=self.uid_to_request_id,
                requests=self.requests,
            )

        return bg

    def _make_prompt_cache_save_callback(self):
        """Create a callback that stores prompt-only KV/Mamba cache.

        Called from ``_generation_step`` right before the first output token
        is fed into the model.  At that point ``num_tokens == 0`` and the
        batch cache contains the exact prompt-only state (correct for both
        KVCache and MambaCache/ArraysCache layers).

        The cache is stored with key = prompt_token_ids so that a future
        request with the identical prompt gets an exact hit.
        """
        import time as _time

        def _prompt_cache_save(uid, extracted_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            prompt_tokens = list(request.prompt_token_ids)
            _t0 = _time.monotonic()
            # evict_prefixes=False: keep mid-prefill boundary entries so
            # that future requests with the same prefix but different
            # suffix get a prefix cache hit (critical for agentic multi-turn).
            stored = self.memory_aware_cache.store(
                prompt_tokens, extracted_cache, evict_prefixes=False
            )
            _dt = _time.monotonic() - _t0
            if stored:
                logger.info(
                    f"[prompt_cache_save] request={request_id[:12]} "
                    f"prompt_tokens={len(prompt_tokens)} "
                    f"store_time={_dt:.3f}s"
                )

        return _prompt_cache_save

    def _make_mid_prefill_save_callback(self, save_interval: int):
        """Create a callback for saving intermediate KV cache during chunked prefill.

        The callback is called after each chunk with (uid, processed_tokens,
        prompt_cache).  It extracts the cache state (immutable MLX array
        snapshots), reconstructs KVCache objects, and stores them in the
        memory-aware prefix cache so that a subsequent request with the same
        prompt prefix can skip the already-computed tokens.
        """
        import time as _time

        def _mid_prefill_save(uid, processed_tokens, prompt_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            total_cached = (request.cached_tokens or 0) + processed_tokens

            # Always save at prefix_boundary (message boundary for cache
            # reuse with different final user messages).
            prefix_boundary = getattr(request, "prefix_boundary", 0)
            at_prefix_boundary = prefix_boundary > 0 and total_cached == prefix_boundary

            # Throttle: only save every save_interval tokens,
            # unless we're at the prefix boundary.
            last_save = getattr(request, "_mid_prefill_last_save", 0)
            if not at_prefix_boundary and total_cached - last_save < save_interval:
                return

            # Extract immutable state snapshots
            extracted = self._extract_cache_states(prompt_cache)
            if not extracted:
                return

            # Reconstruct cache objects (directly usable by BatchGenerator)
            reconstructed = self._reconstruct_cache_from_states(extracted)
            if not reconstructed:
                return

            prefix_tokens = list(request.prompt_token_ids[:total_cached])

            # Remove previous intermediate entry to avoid memory waste
            old_key = getattr(request, "_mid_prefill_cache_key", None)
            if old_key is not None:
                self.memory_aware_cache.remove(list(old_key))

            _t0 = _time.monotonic()
            stored = self.memory_aware_cache.store(prefix_tokens, reconstructed)
            _dt = _time.monotonic() - _t0

            if stored:
                request._mid_prefill_last_save = total_cached
                request._mid_prefill_cache_key = tuple(prefix_tokens)
                logger.info(
                    f"[mid_prefill_cache] request={request_id[:12]} "
                    f"saved {total_cached}/{len(request.prompt_token_ids)} tokens "
                    f"({total_cached * 100 // len(request.prompt_token_ids)}%) "
                    f"store_time={_dt:.3f}s"
                )
            else:
                logger.debug(
                    f"[mid_prefill_cache] request={request_id[:12]} "
                    f"store rejected for {total_cached} tokens"
                )

        return _mid_prefill_save

    def _close_batch_generator(self) -> None:
        """Properly close BatchGenerator to restore wired_limit."""
        if self.batch_generator is not None:
            try:
                if hasattr(self.batch_generator, "close"):
                    self.batch_generator.close()
            except Exception as e:
                logger.debug(f"Error closing BatchGenerator: {e}")
            self.batch_generator = None

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> None:
        """Ensure BatchGenerator exists with compatible settings."""
        sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, we need to drain it first
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Sampling parameters changed with active requests. "
                    "New requests will use new parameters after current batch completes."
                )
                return

            # Keep prefix cache across BatchGenerator recreations.
            # KV cache entries depend only on the input tokens, not on
            # sampling params (temperature, top_p, min_p).  Since the
            # server runs a single model, the cache is always valid.
            if self.batch_generator is not None:
                n_entries = 0
                if self.memory_aware_cache is not None:
                    n_entries = len(self.memory_aware_cache._entries)
                elif self.prefix_cache is not None:
                    n_entries = (
                        len(self.prefix_cache)
                        if hasattr(self.prefix_cache, "__len__")
                        else 0
                    )
                logger.info(
                    f"[batch_generator] recreating (sampler params changed), "
                    f"keeping {n_entries} cache entries"
                )

            self._close_batch_generator()
            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params

    def _validate_cache(self, cache: Any) -> bool:
        """
        Validate that a cache object is usable.

        Checks for None references AND shape compatibility.  Restored
        cache entries must have batch_size == 1 (single sequence) so
        they can be merged into the running batch by _merge_caches.
        A shape mismatch here (e.g. batch=2 from a stale entry) would
        cause a concatenation crash inside _merge_caches.

        Args:
            cache: The cache object to validate

        Returns:
            True if cache is valid and usable
        """
        if cache is None:
            return False

        # Check if it's a list of cache layers
        if isinstance(cache, list):
            if len(cache) == 0:
                return False
            # Check each layer
            for layer_cache in cache:
                if layer_cache is None:
                    return False
                # Check if layer has expected structure
                if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                    return False
                if hasattr(layer_cache, "values") and layer_cache.values is None:
                    return False
                # Validate batch dimension == 1 for KVCache layers
                if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                    if layer_cache.keys.shape[0] != 1:
                        logger.debug(
                            f"Cache layer invalid: keys batch={layer_cache.keys.shape[0]}, expected 1"
                        )
                        return False
                # Validate batch dimension for MambaCache layers
                if hasattr(layer_cache, "cache") and isinstance(
                    layer_cache.cache, list
                ):
                    for arr in layer_cache.cache:
                        if arr is not None and arr.shape[0] != 1:
                            logger.debug(
                                f"Cache layer invalid: mamba batch={arr.shape[0]}, expected 1"
                            )
                            return False

        # Check BatchKVCache structure
        if hasattr(cache, "caches"):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False

        return True

    def _extract_cache_states(self, raw_cache: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache.

        This extracts the real KV data using mlx-lm's cache.state property,
        allowing the data to be stored and reconstructed later even after
        the BatchGenerator is recreated.

        Args:
            raw_cache: List of KVCache objects from mlx-lm

        Returns:
            List of dicts with {state: (keys, values), meta_state: (offset,), class_name: str}
        """
        if not raw_cache:
            return []

        extracted = []
        for layer_cache in raw_cache:
            try:
                if hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    state = layer_cache.state  # (keys, values) or more for Mamba
                    meta = layer_cache.meta_state  # (offset,) as strings
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": type(layer_cache).__name__,
                            "class_ref": type(layer_cache),
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to extract state from cache layer: {e}")
                continue

        return extracted if len(extracted) == len(raw_cache) else []

    def _reconstruct_cache_from_states(
        self, extracted_states: List[Dict[str, Any]]
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from extracted cache states.

        This is the inverse of _extract_cache_states(). Uses mlx-lm's
        _BaseCache.from_state() to reconstruct any cache type (KVCache,
        MambaCache, etc.) from its state/meta_state.

        Args:
            extracted_states: List of dicts from _extract_cache_states()

        Returns:
            List of cache objects, or None if reconstruction fails
        """
        if not extracted_states:
            return None

        try:
            caches = []
            for layer_state in extracted_states:
                state = layer_state.get("state")
                meta_state = layer_state.get("meta_state")
                cache_cls = layer_state.get("class_ref")
                if state is None:
                    return None

                if cache_cls is not None and hasattr(cache_cls, "from_state"):
                    # BatchKVCache doesn't inherit from KVCache, so
                    # _merge_caches can't handle it. Convert to KVCache
                    # (safe because mid-prefill save is always batch_size=1).
                    from mlx_lm.models.cache import BatchKVCache as _BatchKVCache
                    from mlx_lm.models.cache import KVCache as _KVCache

                    if cache_cls is _BatchKVCache:
                        # BatchKVCache.state = (keys, values, offset, left_padding)
                        keys, values = state[0], state[1]
                        cache = _KVCache()
                        cache.keys = keys
                        cache.values = values
                        cache.offset = keys.shape[2]
                    else:
                        cache = cache_cls.from_state(state, meta_state)
                else:
                    # Fallback: try KVCache manual reconstruction
                    from mlx_lm.models.cache import KVCache

                    if len(state) != 2:
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    cache.offset = (
                        int(meta_state[0]) if meta_state else cache.keys.shape[2]
                    )

                caches.append(cache)

            return caches

        except Exception as e:
            logger.info(f"[mid_prefill_cache] reconstruct EXCEPTION: {e}")
            return None

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self.tokenizer.tokenizer.encode(
                        request.prompt
                    )
                else:
                    raise AttributeError(
                        f"Tokenizer {type(self.tokenizer)} has no 'encode' method. "
                        "Continuous batching requires a tokenizer with encode support."
                    )
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Check prefix cache for cached KV state
        if self.block_aware_cache is not None:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                request.prompt_token_ids,
            )
            if block_table and block_table.num_tokens > 0:
                request.cache_hit_type = "hit"
                # Reconstruct actual KVCache objects from stored tensor data
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    request.prompt_cache = reconstructed
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    request.remaining_tokens = remaining
                    logger.debug(
                        f"Request {request.request_id}: paged cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                        f"{len(remaining)} tokens remaining, cache reconstructed"
                    )
                else:
                    # Reconstruction failed, treat as cache miss
                    request.cache_hit_type = "miss"
                    request.remaining_tokens = request.prompt_token_ids
                    logger.debug(
                        f"Request {request.request_id}: paged cache reconstruction failed"
                    )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        elif self.memory_aware_cache is not None:
            # Use memory-aware prefix cache
            import time as _time

            _fetch_t0 = _time.monotonic()
            cache, remaining = self.memory_aware_cache.fetch(request.prompt_token_ids)
            _fetch_dt = _time.monotonic() - _fetch_t0
            request.cache_hit_type = self.memory_aware_cache._last_match_type
            if cache:
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} HIT "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"cached={request.cached_tokens} remaining={len(remaining)} "
                    f"time={_fetch_dt:.3f}s"
                )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} MISS "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"time={_fetch_dt:.3f}s entries={len(self.memory_aware_cache._entries)}"
                )
        elif self.prefix_cache is not None:
            # Use legacy prefix cache
            cache, remaining = self.prefix_cache.fetch_cache(request.prompt_token_ids)
            if cache:
                request.cache_hit_type = "hit"
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.debug(
                    f"Request {request.request_id}: cache hit, "
                    f"{request.cached_tokens} tokens cached, "
                    f"{len(remaining)} tokens remaining"
                )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        else:
            request.cache_hit_type = "miss"
            request.remaining_tokens = request.prompt_token_ids

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def abort_request(self, request_id: str) -> bool:
        """
        Queue request for abort. Thread-safe, called from any thread.

        The actual abort is deferred to the executor thread (inside step())
        to avoid race conditions with in-flight Metal GPU operations.

        Args:
            request_id: The request ID to abort

        Returns:
            True (abort is always enqueued)
        """
        self._pending_abort_ids.add(request_id)
        logger.info(f"[abort_request] {request_id[:12]} enqueued for deferred abort")
        return True

    def _process_pending_aborts(self) -> None:
        """Drain and process pending abort requests. Called from executor thread."""
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _do_abort_request(self, request_id: str) -> bool:
        """
        Actually abort a request. Must be called from the executor thread.

        Handles the case where the request was already removed from
        self.requests by _cleanup_request() but still lives in the
        BatchGenerator (e.g. in _partial or active_batch).

        Args:
            request_id: The request ID to abort

        Returns:
            True if any cleanup was performed, False otherwise
        """
        request = self.requests.get(request_id)
        was_waiting = False
        was_running = False
        removed_from_batch = False

        # Remove from waiting queue
        if request is not None and request.status == RequestStatus.WAITING:
            was_waiting = True
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from running (BatchGenerator) — do this even if request
        # was already cleaned up from self.requests, because the UID may
        # still be live inside the BatchGenerator (_partial / active_batch).
        if request_id in self.request_id_to_uid:
            was_running = True
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
                if self.batch_generator.active_batch is not None:
                    from vllm_mlx.spec_decode.cache_utils import (
                        fixup_cache_after_filter,
                    )

                    fixup_cache_after_filter(self.batch_generator.active_batch.cache)
                removed_from_batch = True
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        if request is not None:
            request.set_finished(RequestStatus.FINISHED_ABORTED)
        self.finished_req_ids.add(request_id)

        # Clean up spec decode / MTP state
        if self._spec_decode_runtime is not None:
            self._spec_decode_runtime.remove_request(request_id)
        self._mtp_hidden_states.pop(request_id, None)

        # Flush Metal encoders after removing arrays from batch
        mx.clear_cache()

        logger.info(
            f"[abort_request] {request_id[:12]} ABORTED "
            f"was_waiting={was_waiting} was_running={was_running} "
            f"removed_from_batch={removed_from_batch} "
            f"remaining_running={len(self.running)} remaining_waiting={len(self.waiting)}"
        )
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> List[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Determine tokens to process and cache to use
            # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
            # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
            # pass just the last token so BatchGenerator can start generation.
            if (
                request.remaining_tokens is not None
                and len(request.remaining_tokens) == 0
            ):
                # Exact cache match - pass only last token for generation kickoff
                tokens_to_process = request.prompt_token_ids[-1:]
            elif request.remaining_tokens:
                tokens_to_process = request.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request.prompt_cache  # May be None

            # Validate cache before using it
            if cache_to_use is not None and not self._validate_cache(cache_to_use):
                logger.debug(
                    f"Request {request.request_id}: invalid cache detected, "
                    f"proceeding without cache"
                )
                cache_to_use = None
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                tokens_to_process = request.prompt_token_ids

            # Insert into BatchGenerator with optional cache.
            # Wrap in try/except: if cache shapes are incompatible
            # (e.g. stale entry after BatchGenerator recreation),
            # fall back to no-cache insert instead of crashing.
            try:
                uids = self.batch_generator.insert(
                    [tokens_to_process],
                    max_tokens=[request.sampling_params.max_tokens],
                    caches=[cache_to_use] if cache_to_use else None,
                )
            except Exception as e:
                if cache_to_use is not None:
                    logger.warning(
                        f"[cache_insert_error] request={request.request_id[:12]} "
                        f"cache insert failed ({e}), retrying without cache"
                    )
                    cache_to_use = None
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=None,
                    )
                else:
                    raise

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                self.running[request.request_id] = request
                scheduled.append(request)

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request.cached_tokens} cached"
                    if request.cached_tokens > 0
                    else ""
                )
                tokens_to_prefill = len(tokens_to_process)
                logger.info(
                    f"[schedule] request={request.request_id[:12]} uid={uid} "
                    f"prompt_tokens={request.num_prompt_tokens} "
                    f"tokens_to_prefill={tokens_to_prefill}{cache_info} "
                    f"max_tokens={request.sampling_params.max_tokens} "
                    f"running={len(self.running)} waiting={len(self.waiting)}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: List[Any]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Append token to request
            request.append_output_token(response.token)

            # Record first token time for TTFT metric
            if request.first_token_time is None and request.num_output_tokens > 0:
                import time as _time

                request.first_token_time = _time.time()

            # Decode the new token (skip stop tokens — they are not content)
            if response.finish_reason == "stop":
                new_text = ""
            else:
                new_text = self._decode_tokens([response.token])

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
            )

            # Check if finished
            if response.finish_reason is not None:
                if response.finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif response.finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

                output.finished = True
                output.finish_reason = response.finish_reason
                finished_ids.add(request_id)

                # Decode full output
                output.output_text = self._decode_tokens(request.output_token_ids)
                request.output_text = output.output_text

                # Extract cache for future reuse (critical for agentic multi-turn)
                if hasattr(response, "prompt_cache"):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            # For paged cache, extract actual tensor states
                            # This allows cache to survive BatchGenerator recreation
                            if self.block_aware_cache is not None:
                                extracted_cache = self._extract_cache_states(raw_cache)
                                if extracted_cache:
                                    request._extracted_cache = extracted_cache
                                    logger.debug(
                                        f"Extracted {len(extracted_cache)} layer states "
                                        f"for request {request_id}"
                                    )
                            else:
                                # Standard cache stores object references
                                request._extracted_cache = raw_cache
                    except Exception as e:
                        logger.debug(f"Failed to extract cache for {request_id}: {e}")

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse
            if request is not None and request.prompt_token_ids:
                if self.block_aware_cache is not None:
                    # Store in paged cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.block_aware_cache.store_cache(
                                request_id,
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored paged cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to store paged cache for {request_id}: {e}"
                            )
                    # NOTE: Do NOT call release_cache here - blocks should persist
                    # for future requests to share. The LRU eviction will clean up
                    # unused blocks when under memory pressure.

                elif self.memory_aware_cache is not None:
                    # Keep mid-prefill entry as prefix cache for future
                    # requests that share a common prefix (e.g. same system
                    # prompt + tools but different user message).  LRU
                    # eviction handles memory pressure.

                    # Store in memory-aware prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            import time as _time

                            _store_t0 = _time.monotonic()
                            stored = self.memory_aware_cache.store(
                                full_token_sequence,
                                request._extracted_cache,
                                evict_prefixes=False,
                            )
                            _store_dt = _time.monotonic() - _store_t0
                            # NOTE: We intentionally do NOT store a prompt-only
                            # cache entry.  Hybrid Mamba+Transformer models
                            # (like Qwen3-Coder-Next) have MambaCache layers
                            # whose state is cumulative and cannot be trimmed
                            # back to "prompt only".  Reusing such state causes
                            # the model to immediately produce EOS.
                            # The full prompt+output entry is stored above; a
                            # future request with the same prompt will hit the
                            # supersequence match path in the fetch, which is
                            # now disabled for safety (see memory_cache.py).

                            logger.info(
                                f"[cache_store] request={request_id[:12]} "
                                f"tokens={len(full_token_sequence)} "
                                f"({len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output) "
                                f"stored={stored} time={_store_dt:.3f}s "
                                f"cache_entries={len(self.memory_aware_cache._entries)} "
                                f"cache_mem={self.memory_aware_cache._current_memory / 1e6:.0f}MB"
                            )
                            # Release the original FP16 cache reference so
                            # memory can be reclaimed (the quantized copy
                            # lives inside the prefix cache now).
                            request._extracted_cache = None
                        except Exception as e:
                            logger.debug(
                                f"Failed to store memory-aware cache for {request_id}: {e}"
                            )

                elif self.prefix_cache is not None:
                    # Store in legacy prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    # The next turn's prompt will include the previous response
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.prefix_cache.store_cache(
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")

            # Evaluate stored cache tensors incrementally (per-layer) to prevent
            # a deferred batch evaluation spike when all lazy ops resolve at once.
            # This spreads the VRAM cost across smaller per-layer evaluations.
            if (
                request is not None
                and hasattr(request, "_extracted_cache")
                and request._extracted_cache
            ):
                for layer in request._extracted_cache:
                    if isinstance(layer, dict) and "state" in layer:
                        keys, values = layer["state"]
                        mx.eval(keys, values)
                    elif hasattr(layer, "keys") and hasattr(layer, "values"):
                        keys_attr = layer.keys
                        values_attr = layer.values
                        if not callable(keys_attr) and not callable(values_attr):
                            mx.eval(keys_attr, values_attr)

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

                # Remove from batch generator (needed for spec decode path which
                # bypasses BatchGenerator.next() internal removal)
                if self.batch_generator is not None:
                    self.batch_generator.remove([uid])
                    # Fix stale _idx after filter and evaluate cache metadata
                    if self.batch_generator.active_batch is not None:
                        from vllm_mlx.spec_decode.cache_utils import (
                            fixup_cache_after_filter,
                        )

                        fixup_cache_after_filter(
                            self.batch_generator.active_batch.cache
                        )

            # Clean up spec decode state
            if self._spec_decode_runtime is not None:
                self._spec_decode_runtime.remove_request(request_id)
            self._mtp_hidden_states.pop(request_id, None)

            # Track as finished
            self.finished_req_ids.add(request_id)

        # Free Metal command buffers after cleanup (prevents end-of-generation spike)
        if finished_ids:
            mx.clear_cache()

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        error_str = str(error)
        return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Properly close batch generator (this is the source of the corruption)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Clear MTP hidden states to prevent stale state after recovery
        if hasattr(self, "_mtp_hidden_states") and self._mtp_hidden_states:
            self._mtp_hidden_states.clear()

        logger.info("Cache recovery completed")

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        # Clear MTP hidden states - rescheduled requests will recompute from scratch
        if hasattr(self, "_mtp_hidden_states") and self._mtp_hidden_states:
            self._mtp_hidden_states.clear()

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 1) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache corruption errors

        Args:
            max_retries: Number of times to retry on cache errors (default 1)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (in executor thread, safe for MLX)
        self._process_pending_aborts()

        for attempt in range(max_retries + 1):
            try:
                # Schedule waiting requests
                scheduled = self._schedule_waiting()
                output.scheduled_request_ids = [r.request_id for r in scheduled]
                output.num_scheduled_tokens = sum(
                    r.num_prompt_tokens for r in scheduled
                )

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    if self._can_spec_decode():
                        # Speculative decoding path
                        spec_responses = self._step_spec_decode()
                        if spec_responses is not None:
                            responses = spec_responses
                            output.has_work = True
                            if responses:
                                outputs, finished_ids = self._process_batch_responses(
                                    responses
                                )
                                output.outputs = outputs
                                output.finished_request_ids = finished_ids
                                self._cleanup_finished(finished_ids)
                        else:
                            # Spec decode returned None — fall back to normal decode
                            responses = self.batch_generator.next()
                            output.has_work = True
                            if responses:
                                # Invalidate stale MTP hidden states after normal decode
                                if (
                                    self._mtp_hidden_states
                                    and self.config.speculative_method == "mtp"
                                ):
                                    for resp in responses:
                                        rid = self.uid_to_request_id.get(resp.uid)
                                        if rid:
                                            self._mtp_hidden_states.pop(rid, None)
                                outputs, finished_ids = self._process_batch_responses(
                                    responses
                                )
                                output.outputs = outputs
                                output.finished_request_ids = finished_ids
                                self._cleanup_finished(finished_ids)
                    else:
                        # Normal decode path
                        responses = self.batch_generator.next()
                        output.has_work = True
                        if responses:
                            # Invalidate stale MTP hidden states after normal decode
                            if (
                                self._mtp_hidden_states
                                and self.config.speculative_method == "mtp"
                            ):
                                for resp in responses:
                                    rid = self.uid_to_request_id.get(resp.uid)
                                    if rid:
                                        self._mtp_hidden_states.pop(rid, None)
                            outputs, finished_ids = self._process_batch_responses(
                                responses
                            )
                            output.outputs = outputs
                            output.finished_request_ids = finished_ids
                            self._cleanup_finished(finished_ids)

                # Success - break out of retry loop
                break

            except TypeError as e:
                # Catch the NoneType error specifically
                if self._is_cache_corruption_error(e):
                    if attempt < max_retries:
                        logger.warning(
                            f"Cache corruption detected (attempt {attempt + 1}), "
                            f"performing recovery and retry..."
                        )
                        # Deep reset to recover
                        self._recover_from_cache_error()
                        # Re-add any running requests back to waiting
                        self._reschedule_running_requests()
                    else:
                        logger.error(
                            f"Cache corruption not recoverable after "
                            f"{max_retries + 1} attempts"
                        )
                        raise
                else:
                    raise
            except Exception as e:
                logger.error(f"Error in batch generation step: {e}")
                raise

        # Clear finished tracking for next step
        old_finished = self.finished_req_ids
        self.finished_req_ids = set()

        # Periodically clear Metal cache to prevent memory accumulation
        self._step_count += 1
        if self._step_count % self._clear_cache_interval == 0:
            mx.clear_cache()

        # Periodically log memory stats for monitoring
        if self._step_count % self._memory_log_interval == 0:
            try:
                if mx.metal.is_available():
                    active_gb = mx.get_active_memory() / 1e9
                    peak_gb = mx.get_peak_memory() / 1e9
                    cache_gb = mx.get_cache_memory() / 1e9
                    logger.info(
                        f"[Metal memory] active={active_gb:.1f}GB "
                        f"peak={peak_gb:.1f}GB cache={cache_gb:.1f}GB "
                        f"step={self._step_count} "
                        f"running={len(self.running)} waiting={len(self.waiting)}"
                    )
            except Exception:
                pass

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_running_requests_info(self) -> List[Dict[str, Any]]:
        """Per-request details for status endpoint."""
        import time as _time

        now = _time.time()
        result = []

        # Waiting requests
        for req in self.waiting:
            result.append(
                {
                    "request_id": req.request_id,
                    "status": "waiting",
                    "phase": "queued",
                    "elapsed_s": round(now - req.arrival_time, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": 0,
                    "max_tokens": req.max_tokens,
                    "progress": 0.0,
                    "tokens_per_second": None,
                    "ttft_s": None,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        # Running requests
        for req in self.running.values():
            n_out = req.num_output_tokens
            elapsed = now - req.arrival_time

            # Phase detection
            if n_out == 0:
                phase = "prefill"
            else:
                phase = "generation"

            # Tokens per second (generation phase only)
            tok_s = None
            ttft = None
            if req.first_token_time is not None:
                ttft = round(req.first_token_time - req.arrival_time, 3)
                gen_elapsed = now - req.first_token_time
                if gen_elapsed > 0 and n_out > 0:
                    tok_s = round(n_out / gen_elapsed, 1)

            # Progress: completion_tokens / max_tokens
            progress = round(n_out / req.max_tokens, 3) if req.max_tokens > 0 else 0.0

            result.append(
                {
                    "request_id": req.request_id,
                    "status": "running",
                    "phase": phase,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": n_out,
                    "max_tokens": req.max_tokens,
                    "progress": min(progress, 1.0),
                    "tokens_per_second": tok_s,
                    "ttft_s": ttft,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        # Include cache stats
        if self.block_aware_cache is not None:
            stats["paged_cache"] = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()

        # Add spec decode stats
        if self._spec_decode_runtime is not None:
            stats["spec_decode"] = {
                "enabled": self._spec_decode_enabled,
                "auto_disabled": self._spec_decode_runtime.auto_disabled,
                "method": self.config.speculative_method,
                "num_speculative_tokens": self.config.num_speculative_tokens,
                "total_drafts": self._spec_decode_runtime.stats.num_drafts,
                "total_draft_tokens": self._spec_decode_runtime.stats.num_draft_tokens,
                "total_accepted": self._spec_decode_runtime.stats.num_accepted_tokens,
                "acceptance_rate": self._spec_decode_runtime.stats.acceptance_rate(),
                "recent_acceptance_rate": self._spec_decode_runtime.stats.recent_acceptance_rate(),
                "mean_accepted_length": self._spec_decode_runtime.stats.mean_accepted_length(),
                "per_position_acceptance": self._spec_decode_runtime.stats.acceptance_rate_per_position,
            }

        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            return self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return None

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Drain any pending deferred aborts
        self._pending_abort_ids.clear()

        # Abort all requests directly (reset is synchronous)
        for request_id in list(self.requests.keys()):
            self._do_abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    # -----------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.save_to_disk(cache_dir)
        logger.info("[cache_persist] no memory-aware cache to save")
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.load_from_disk(cache_dir)
        logger.info("[cache_persist] no memory-aware cache to load into")
        return 0
