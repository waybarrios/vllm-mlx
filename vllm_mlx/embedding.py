# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine using mlx-embeddings.

Provides lazy-loaded model management and batch embedding generation
for the OpenAI-compatible /v1/embeddings endpoint.
"""

import logging
import time
from typing import Any

import mlx.core as mx

from vllm_mlx.metrics import metrics
from vllm_mlx.utils.truncation import inner_tokenizer, resolve_max_length

logger = logging.getLogger(__name__)

# Resolved max_length above this, with no operator-set ceiling, gets a
# startup warning: attention cost scales roughly quadratically with sequence
# length, and a batch pads every input to its longest member, so large
# unbounded context windows can use disproportionate memory.
LARGE_CONTEXT_WARNING_THRESHOLD = 4096

# Default cap on padded token positions for multi-input forward passes. A
# single longer input still runs alone, up to the effective max length.
DEFAULT_EMBEDDING_TOKEN_BUDGET = 4096


class EmbeddingLengthExceededError(Exception):
    """Raised under the "error" overflow policy when an input exceeds the
    effective embedding max length."""

    def __init__(self, text_index: int, token_count: int, max_length: int):
        self.text_index = text_index
        self.token_count = token_count
        self.max_length = max_length
        super().__init__(
            f"Input {text_index} has {token_count} tokens, "
            f"exceeding the effective embedding max length of {max_length}"
        )


class EmbeddingEngine:
    """
    Wrapper around mlx-embeddings for text embedding generation.

    Supports lazy model loading and batch embedding with proper
    tokenization and pooling.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length_ceiling: int | None = None,
        overflow_policy: str = "truncate",
        token_budget: int = DEFAULT_EMBEDDING_TOKEN_BUDGET,
    ):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._max_length: int | None = None
        self._max_length_ceiling = max_length_ceiling
        self._overflow_policy = overflow_policy
        self._token_budget = token_budget

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_max_length(self) -> int | None:
        """The truncation length actually applied by embed(), resolved from
        the loaded model's config and clamped by the configured ceiling
        (see resolve_max_length()). Unlike the raw ceiling, this reflects
        what the engine really uses — e.g. a 512-token model with an 8192
        ceiling still reports 512 here. None before the model has loaded.
        """
        if not self.is_loaded:
            return None
        return self._resolve_max_length()

    def load(self) -> None:
        """Load the embedding model and tokenizer."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s: {self.model_name}")

        max_length = self._resolve_max_length()
        if (
            self._max_length_ceiling is None
            and max_length > LARGE_CONTEXT_WARNING_THRESHOLD
        ):
            logger.warning(
                "Embedding model %s reports a %d-token context window and no "
                "--embedding-max-length ceiling is set. Batches are padded to "
                "their longest input, so large inputs near this length can use "
                "significant memory. Consider setting --embedding-max-length "
                "to cap it for memory-constrained deployments.",
                self.model_name,
                max_length,
            )

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def _resolve_max_length(self) -> int:
        """Tokenizer truncation length from the model config (cached)."""
        if self._max_length is None:
            self._max_length = resolve_max_length(
                getattr(self._model, "config", None),
                self._tokenizer,
                ceiling=self._max_length_ceiling,
            )
        return self._max_length

    def _exact_token_length(self, text: str) -> int:
        """Raw (untruncated) token count for a single text, via the real
        tokenizer. Used to enforce the overflow policy — that decision needs
        an accurate count, not an estimate that could silently let an
        over-limit input through (or reject a valid one). Propagates any
        tokenizer failure instead of masking it with a heuristic; see
        _estimated_token_length() for the reporting-only fallback.
        """
        tokens = self._tokenizer.encode(text)
        if isinstance(tokens, list):
            return len(tokens)
        elif hasattr(tokens, "__len__"):
            return len(tokens)
        return tokens.size

    def _estimated_token_length(self, text: str) -> int:
        """Approximate token count for usage reporting only (count_tokens()).
        Falls back to a rough ~4-chars-per-token estimate if the tokenizer
        can't encode the text — acceptable for a reported number, but not for
        the overflow-policy decision in embed() (see _exact_token_length()).
        """
        try:
            return self._exact_token_length(text)
        except Exception:
            return max(1, len(text) // 4)

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: A single string or list of strings.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            EmbeddingLengthExceededError: If ``overflow_policy="error"`` and
                any input exceeds the effective max length.
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        max_length = self._resolve_max_length()
        raw_counts: list[int] = []
        overflows: list[tuple[int, int]] = []
        for i, text in enumerate(texts):
            token_count = self._exact_token_length(text)
            raw_counts.append(token_count)
            if token_count <= max_length:
                continue
            if self._overflow_policy == "error":
                raise EmbeddingLengthExceededError(i, token_count, max_length)
            overflows.append((i, token_count))

        if overflows:
            # One aggregated line per embed() call instead of one per
            # over-limit text, so a single large batch can't flood the log.
            shown = [i for i, _ in overflows[:5]]
            indices = f"{shown}{', ...' if len(overflows) > len(shown) else ''}"
            longest = max(token_count for _, token_count in overflows)
            logger.warning(
                "Embedding: %d/%d inputs truncated (indices %s), longest has "
                "%d tokens, exceeding effective max length %d for model %s",
                len(overflows),
                len(texts),
                indices,
                longest,
                max_length,
                self.model_name,
            )
            for _ in overflows:
                metrics.observe_embedding_truncated(model=self.model_name)

        # Tokenize directly instead of using mlx_embeddings.generate(),
        # which has compatibility issues with newer tokenizers (e.g.
        # GemmaTokenizer lacks batch_encode_plus, and the model's __call__
        # expects positional `inputs` not `input_ids` as a kwarg).
        inner_tok = inner_tokenizer(self._tokenizer)

        # Pack into sub-batches bounded by self._token_budget instead of one
        # batch padded to the single longest input: `padding=True` pads
        # every text in a batch to its longest member, and attention cost
        # scales roughly quadratically with sequence length, so an unbounded
        # request (one huge input, or many inputs near max_length) can
        # otherwise blow up memory/compute for that one request.
        # Budgeted on the post-truncation length, since that's what's
        # actually padded/computed — mirrors RerankEngine.score_pairs().
        effective_counts = [min(c, max_length) for c in raw_counts]
        result: list[list[float] | None] = [None] * len(texts)
        for batch_indices in self._token_budget_batches(effective_counts):
            batch_texts = [texts[i] for i in batch_indices]
            try:
                batch_embeds = self._embed_batch(inner_tok, batch_texts, max_length)
            finally:
                # MLX caches freed buffers by size. Clear after every pass so
                # varying padded shapes cannot accumulate in the allocator,
                # including when tokenization or model execution fails.
                mx.clear_cache()

            for idx, embed in zip(batch_indices, batch_embeds):
                result[idx] = embed

        return result  # type: ignore[return-value]

    def _embed_batch(
        self, tokenizer: Any, texts: list[str], max_length: int
    ) -> list[list[float]]:
        """Run one padded embedding pass and return JSON-serializable vectors."""
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        output = self._model(input_ids, attention_mask=attention_mask)
        return output.text_embeds.tolist()

    def _token_budget_batches(self, effective_counts: list[int]) -> list[list[int]]:
        """Group text indices into sub-batches whose padded token positions
        stay at or under self._token_budget, in input order. A single text over
        budget still forms a batch of one because max_length bounds individual
        sequence length.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for i, count in enumerate(effective_counts):
            prospective_max = max(current_max, count)
            prospective_size = len(current) + 1
            if current and prospective_size * prospective_max > self._token_budget:
                batches.append(current)
                current = []
                current_max = 0
            current.append(i)
            current_max = max(current_max, count)
        if current:
            batches.append(current)
        return batches

    def count_tokens(self, texts: str | list[str]) -> int:
        """Approximate token count for usage reporting."""
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        return sum(self._estimated_token_length(text) for text in texts)
