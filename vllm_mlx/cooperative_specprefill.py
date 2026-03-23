# SPDX-License-Identifier: Apache-2.0
"""
Cooperative SpecPrefill — chunked draft model scoring with yield points.

Draft scoring (the slowest phase) is broken into chunks. Between chunks,
the caller yields to the event loop so active requests can generate tokens.
Selection + sparse prefill + generation run monolithically after scoring.

The draft model is separate from the target model — no shared mutable state
(RoPE, cache). Safe to interleave draft scoring with target generation.
"""

import logging
import math
import time

import mlx.core as mx

logger = logging.getLogger(__name__)


class ChunkedDraftScorer:
    """Break draft model token scoring into chunks for cooperative scheduling.

    Usage:
        scorer = ChunkedDraftScorer(draft_model, tokens, chunk_size=4096)
        while scorer.is_scoring:
            scorer.step()       # Process one chunk of draft prefill
            await asyncio.sleep(0)  # Yield to event loop
        importance = scorer.finalize()  # Lookahead + importance (fast)
    """

    def __init__(
        self,
        draft_model,
        tokens,
        chunk_size: int = 4096,
        n_lookahead: int = 8,
        pool_kernel: int = 13,
        temp: float = 0.6,
        top_p: float = 0.95,
        query_extractor=None,
    ):
        self._model = draft_model
        self._tokens = tokens if isinstance(tokens, list) else tokens.tolist()
        self._chunk_size = chunk_size
        self._n_lookahead = n_lookahead
        self._pool_kernel = pool_kernel
        self._temp = temp
        self._top_p = top_p
        self._query_extractor = query_extractor

        # State
        self._cache = None
        self._prompt = mx.array(self._tokens)  # Create once, reuse in step()
        self._processed = 0
        self._logits = None
        self._done_scoring = False
        self._importance = None
        self._t_start = time.monotonic()

    @property
    def is_scoring(self) -> bool:
        """True while draft prefill chunks remain."""
        return not self._done_scoring

    @property
    def is_done(self) -> bool:
        """True after finalize() completes."""
        return self._importance is not None

    @property
    def tokens_processed(self) -> int:
        return self._processed

    @property
    def chunks_remaining(self) -> int:
        n = len(self._tokens)
        remaining = n - self._processed
        if remaining <= 1:
            return 0
        return math.ceil((remaining - 1) / self._chunk_size)

    def step(self) -> bool:
        """Process one chunk of draft prefill. Returns True if more chunks remain."""
        from mlx_lm.models.cache import make_prompt_cache

        if self._done_scoring:
            return False

        # Initialize cache on first step
        if self._cache is None:
            self._cache = make_prompt_cache(self._model)

        n = len(self._tokens)
        remaining = n - self._processed

        if remaining > 1:
            chunk = min(self._chunk_size, remaining - 1)
            self._model(
                self._prompt[self._processed : self._processed + chunk][None],
                cache=self._cache,
            )
            mx.eval([c.state for c in self._cache])
            self._processed += chunk
            mx.clear_cache()
            return (n - self._processed) > 1  # More chunks?
        else:
            # Last token — get logits
            self._logits = self._model(
                self._prompt[self._processed :][None], cache=self._cache
            )
            mx.eval(self._logits)
            self._processed = n
            self._done_scoring = True
            return False

    def finalize(self):
        """Run lookahead decode + importance computation. Returns importance scores.

        This is fast (~100ms) — no yield needed. Must be called after
        all scoring chunks are processed (is_scoring == False).
        """
        if not self._done_scoring:
            raise RuntimeError("Cannot finalize before scoring is complete")
        if self._importance is not None:
            return self._importance

        from .specprefill import (
            _build_layer_to_cache_map,
            _compute_importance,
            _find_attention_layers,
            _get_attn_module,
            _llama_extract_queries,
            _lookahead_decode,
            _nemotron_h_extract_queries,
            _patch_attention_for_capture,
            _qwen35_extract_queries,
            _unpatch_attention_capture,
        )

        attn_layers = _find_attention_layers(self._model)
        n_attn_layers = len(attn_layers)
        attn_obj = _get_attn_module(attn_layers[0][1])
        n_attn_heads = getattr(
            attn_obj,
            "num_attention_heads",
            getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
        )
        n_kv_heads = getattr(
            attn_obj,
            "num_key_value_heads",
            getattr(attn_obj, "n_kv_heads", None),
        )

        # Auto-detect query extractor
        qe = self._query_extractor
        if qe is None:
            if hasattr(attn_obj, "q_norm"):
                qe = _qwen35_extract_queries
            elif not hasattr(attn_obj, "rope"):
                qe = _nemotron_h_extract_queries
            else:
                qe = _llama_extract_queries

        # Lookahead decode with query capture
        query_buffer = [[] for _ in range(n_attn_layers)]
        patches, attn_indices = _patch_attention_for_capture(
            self._model,
            query_buffer,
            query_extractor=qe,
        )
        try:
            _lookahead_decode(
                self._model,
                self._logits,
                self._cache,
                self._n_lookahead,
                temp=self._temp,
                top_p=self._top_p,
            )
            mx.eval(query_buffer)
        finally:
            _unpatch_attention_capture(self._model, patches)

        # Compute importance
        layer_to_cache = _build_layer_to_cache_map(self._model)
        attn_caches = [self._cache[layer_to_cache[i]] for i in attn_indices]
        self._importance = _compute_importance(
            query_buffer,
            attn_caches,
            len(self._tokens),
            n_attn_heads,
            n_kv_heads,
            pool_kernel=self._pool_kernel if self._pool_kernel > 0 else None,
        )
        mx.eval(self._importance)

        t_total = time.monotonic() - self._t_start
        logger.info(
            "ChunkedDraftScorer: scored %d tokens in %.1fs (%d chunks of %d)",
            len(self._tokens),
            t_total,
            math.ceil(len(self._tokens) / self._chunk_size),
            self._chunk_size,
        )

        # Free draft resources
        del self._cache, self._logits, self._prompt
        self._cache = None
        self._logits = None
        self._prompt = None
        mx.clear_cache()

        return self._importance

    def cleanup(self):
        """Clean up on cancellation. Frees draft cache and intermediates."""
        if self._cache is not None:
            del self._cache
            self._cache = None
        if self._prompt is not None:
            del self._prompt
            self._prompt = None
        if self._logits is not None:
            del self._logits
            self._logits = None
        if self._importance is not None:
            del self._importance
            self._importance = None
        mx.clear_cache()
