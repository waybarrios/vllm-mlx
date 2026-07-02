# SPDX-License-Identifier: Apache-2.0
"""
Shared resolution of the tokenizer truncation length for embedding and
reranker models.

The input token limit follows each model's own context window
(``max_position_embeddings``) instead of a hard-coded 512, capped per engine.
Falls back to the tokenizer's ``model_max_length`` and finally to a default.
"""

from typing import Any

# Shared bounds for the tokenizer truncation length. Both the embedding and
# reranker engines mean the same thing here; a per-engine override, if ever
# needed, is passed at the call site via ``cap=``/``default=``.
MAX_LENGTH_CAP = 8192
MAX_LENGTH_DEFAULT = 512


def _config_get(config: Any, key: str) -> Any:
    """Read ``key`` from a model config that may be a dict or an object."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def inner_tokenizer(tokenizer: Any) -> Any:
    """Unwrap a wrapping tokenizer to its inner ``_tokenizer`` when present."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def resolve_max_length(config: Any, tokenizer: Any, *, cap: int, default: int) -> int:
    """
    Resolve the tokenizer truncation length for a model.

    Source order: ``config.max_position_embeddings`` →
    ``tokenizer.model_max_length`` → ``default``. The result is capped at
    ``cap``, which also guards against the HuggingFace sentinel value
    (``model_max_length`` is often a huge integer when unset).

    Args:
        config: Model config as a dict (reranker) or object (embeddings).
        tokenizer: The tokenizer (possibly wrapping an inner ``_tokenizer``).
        cap: Upper bound for the resolved length.
        default: Fallback when no usable value is found.

    Returns:
        The truncation length to pass as ``max_length``.
    """
    val = _config_get(config, "max_position_embeddings")
    if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
        val = getattr(inner_tokenizer(tokenizer), "model_max_length", None)
    if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
        return default
    return min(val, cap)
