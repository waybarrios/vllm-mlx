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

import asyncio

import mlx.core as mx

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


# =============================================================================
# Engine
# =============================================================================

# Default adapter registry — extend for new model families.
_ADAPTER_REGISTRY: dict[str, type[RerankAdapter]] = {
    "default": SigmoidAdapter,
}


def get_adapter(model_name: str) -> RerankAdapter:
    """
    Return the appropriate adapter for a model.

    Falls back to SigmoidAdapter (works for Jina, BGE, MS-MARCO families).
    Extend _ADAPTER_REGISTRY for families that need different scoring.
    """
    # Future: inspect model config to select adapter automatically.
    # For now, all known MLX reranker models use the sigmoid pattern.
    return _ADAPTER_REGISTRY["default"]()


class RerankEngine:
    """
    Reranker engine for cross-encoder sequence classification models.

    Loads cross-encoder models via transformers + MLX (safetensors weights).
    Scores (query, document) pairs using the adapter contract for
    family-specific tokenization, score extraction, and normalization.

    Supports token-budget batching to avoid OOM on large document lists.
    """

    def __init__(
        self,
        model_name: str,
        token_budget: int = 4096,
        max_concurrency: int = 1,
    ):
        self.model_name = model_name
        self.token_budget = token_budget
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._model = None
        self._tokenizer = None
        self._adapter: RerankAdapter | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """
        Load the cross-encoder model and tokenizer.

        Uses transformers AutoTokenizer and loads MLX weights from safetensors
        via the model's from_pretrained or equivalent MLX loading path.
        """
        from transformers import AutoTokenizer

        logger.info(f"Loading reranker model: {self.model_name}")
        start = time.perf_counter()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = self._load_mlx_model(self.model_name)
        self._adapter = get_adapter(self.model_name)

        elapsed = time.perf_counter() - start
        logger.info(f"Reranker model loaded in {elapsed:.2f}s: {self.model_name}")

    @staticmethod
    def _load_mlx_model(model_name: str):
        """
        Load an MLX cross-encoder model from HuggingFace Hub.

        Attempts mlx-community weights first (safetensors), then falls back
        to transformers AutoModelForSequenceClassification with MLX conversion.
        """
        try:
            from huggingface_hub import snapshot_download
            from safetensors import safe_open

            model_path = snapshot_download(model_name)

            import glob
            import json
            import os

            # Load model config
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                config = json.load(f)

            # Load weights from safetensors
            weight_files = glob.glob(os.path.join(model_path, "*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")

            weights = {}
            for wf in weight_files:
                with safe_open(wf, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = mx.array(f.get_tensor(key))

            # Build model based on architecture
            model_type = config.get("model_type", "")
            num_labels = config.get("num_labels", 1)

            model = _build_classifier_model(model_type, config, weights, num_labels)
            mx.eval(model.parameters())
            return model

        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def score_pairs(self, query: str, documents: list[str]) -> tuple[list[float], int]:
        """
        Score each (query, document) pair and return normalized relevance scores.

        Pairs are batched by token budget to control memory usage. Each batch
        is tokenized together and scored in a single forward pass.
        Returns (scores, total_tokens) where total_tokens reflects the
        actual tokenization used for scoring (consistent with adapter).

        Args:
            query: The query string.
            documents: List of document strings.

        Returns:
            List of normalized relevance scores, one per document,
            in the same order as the input documents.
        """
        self._ensure_loaded()

        # Tokenize each pair individually to measure token counts
        pair_encodings = []
        pair_token_counts = []
        for doc in documents:
            enc = self._adapter.tokenize_pair(self._tokenizer, query, doc)
            pair_encodings.append(enc)
            seq_len = (
                len(enc["input_ids"][0])
                if hasattr(enc["input_ids"][0], "__len__")
                else enc["input_ids"].shape[1]
            )
            pair_token_counts.append(seq_len)

        # Build batches by token budget
        batches = []
        current_batch = []
        current_tokens = 0
        for i, (enc, tok_count) in enumerate(zip(pair_encodings, pair_token_counts)):
            if current_batch and current_tokens + tok_count > self.token_budget:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append((i, enc))
            current_tokens += tok_count
        if current_batch:
            batches.append(current_batch)

        # Score each batch
        all_scores: list[tuple[int, float]] = []
        for batch in batches:
            if len(batch) == 1:
                # Single pair — use encoding directly
                idx, enc = batch[0]
                input_ids = mx.array(enc["input_ids"])
                attention_mask = mx.array(enc["attention_mask"])
            else:
                # Pad and stack multiple pairs
                max_len = max(
                    (
                        len(enc["input_ids"][0])
                        if hasattr(enc["input_ids"][0], "__len__")
                        else enc["input_ids"].shape[1]
                    )
                    for _, enc in batch
                )
                padded_ids = []
                padded_mask = []
                for _, enc in batch:
                    raw_ids = enc["input_ids"][0]
                    ids = (
                        raw_ids.tolist()
                        if hasattr(raw_ids, "tolist")
                        else list(raw_ids)
                    )
                    raw_mask = enc["attention_mask"][0]
                    mask = (
                        raw_mask.tolist()
                        if hasattr(raw_mask, "tolist")
                        else list(raw_mask)
                    )
                    pad_len = max_len - len(ids)
                    padded_ids.append(ids + [0] * pad_len)
                    padded_mask.append(mask + [0] * pad_len)
                input_ids = mx.array(padded_ids)
                attention_mask = mx.array(padded_mask)

            output = self._model(input_ids, attention_mask=attention_mask)
            logits_list = output.logits.tolist()

            for j, (idx, _enc) in enumerate(batch):
                logits_row = logits_list[j] if len(batch) > 1 else logits_list[0]
                raw_score = self._adapter.extract_score(logits_row)
                normalized = self._adapter.normalize(raw_score)
                all_scores.append((idx, normalized))

        # Sort by original index to restore input order
        all_scores.sort(key=lambda x: x[0])
        total_tokens = sum(pair_token_counts)
        return [score for _, score in all_scores], total_tokens


def _build_classifier_model(model_type, config, weights, num_labels):
    """
    Build an MLX sequence classification model from config and weights.

    This is a thin wrapper that constructs the appropriate encoder
    architecture with a classification head on top.
    """
    # Import here to avoid top-level dependency on specific model implementations
    return _MLXClassifierWrapper(config, weights, num_labels)


class _MLXClassifierWrapper:
    """
    Minimal MLX wrapper for sequence classification models.

    Wraps loaded safetensors weights into a callable that returns
    logits for (input_ids, attention_mask) pairs. Supports BERT-family
    and XLM-RoBERTa-family architectures commonly used as cross-encoders.
    """

    def __init__(self, config: dict, weights: dict, num_labels: int):
        self.config = config
        self.weights = weights
        self.num_labels = num_labels
        self._params = list(weights.values())

    def parameters(self):
        """Return model parameters for mx.eval."""
        return self._params

    def __call__(self, input_ids: mx.array, attention_mask: mx.array = None):
        """
        Forward pass through the classifier.

        For encoder-only cross-encoders, this runs the full transformer
        encoder and classification head. The exact layer wiring depends
        on the model architecture.

        This initial implementation uses a weight-lookup forward pass
        that works for standard BERT/XLM-RoBERTa classifiers. For
        models with non-standard architectures, register a custom
        adapter via _ADAPTER_REGISTRY.
        """
        # Use the transformers-style weight naming convention
        # to walk through embeddings -> encoder layers -> classifier
        from vllm_mlx.rerank_forward import classifier_forward

        logits = classifier_forward(
            input_ids, attention_mask, self.weights, self.config
        )
        return _ClassifierOutput(logits=logits)


class _ClassifierOutput:
    """Simple container for classifier output logits."""

    def __init__(self, logits: mx.array):
        self.logits = logits
