# SPDX-License-Identifier: Apache-2.0
"""
Reranker engine for cross-encoder models.

Provides a dedicated RerankEngine with adapter-based scoring for the
OpenAI/Jina-compatible /v1/rerank endpoint. Cross-encoder models use
AutoModelForSequenceClassification-style loading, not mlx_lm.load.
"""

import logging
import math
import time
from abc import ABC, abstractmethod

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Adapter contract
# =============================================================================


class RerankAdapter(ABC):
    """
    Per-family adapter for reranker models.

    Different cross-encoder families use different tokenization patterns,
    score extraction logic, and normalization functions. This contract
    isolates those differences so RerankEngine stays family-agnostic.
    """

    @abstractmethod
    def tokenize_pair(self, tokenizer, query: str, document: str) -> dict:
        """
        Tokenize a (query, document) pair for the cross-encoder.

        Args:
            tokenizer: The HuggingFace tokenizer instance.
            query: The query string.
            document: The document string.

        Returns:
            Dict with 'input_ids' and 'attention_mask' as numpy arrays.
        """
        ...

    @abstractmethod
    def extract_score(self, logits) -> float:
        """
        Extract a raw relevance score from model output logits.

        Args:
            logits: Model output logits (list or array), shape varies by model.

        Returns:
            A single float raw score.
        """
        ...

    @abstractmethod
    def normalize(self, raw_score: float) -> float:
        """
        Normalize a raw score to [0, 1] range.

        Args:
            raw_score: The raw score from extract_score().

        Returns:
            Normalized relevance score in [0, 1].
        """
        ...


class SigmoidAdapter(RerankAdapter):
    """
    Default adapter for single-logit sigmoid rerankers.

    Works with Jina Reranker v2, BGE Reranker v2, and MS-MARCO MiniLM
    families. These models output a single relevance logit at position 0,
    normalized via sigmoid.
    """

    def tokenize_pair(self, tokenizer, query: str, document: str) -> dict:
        """Tokenize as a sentence pair (query, document)."""
        return tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

    def extract_score(self, logits) -> float:
        """Extract the first logit as the relevance score."""
        return float(logits[0])

    def normalize(self, raw_score: float) -> float:
        """Apply sigmoid normalization."""
        return 1.0 / (1.0 + math.exp(-raw_score))
