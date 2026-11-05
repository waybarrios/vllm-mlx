# SPDX-License-Identifier: Apache-2.0
"""
Shared resolution of the tokenizer truncation length for embedding and
reranker models.

The input token limit follows each model's own context window
(``max_position_embeddings``) instead of a hard-coded 512. A finite tokenizer
limit further constrains that architecture value, while HuggingFace's huge
unset placeholder is ignored. This keeps position-table offsets safe for
RoBERTa-family models without losing model-derived limits for sentinel values.

``declared_context_limit`` is the separate bound for generation prompts.
"""

from typing import Any

MAX_LENGTH_DEFAULT = 512

# HuggingFace tokenizers default model_max_length to a huge sentinel (e.g.
# int(1e30)) when unset. Any tokenizer-derived value at or above this is
# treated as "unset" rather than a real context length. No real model
# context window is anywhere near this size, so it does not affect trusted
# config values.
TOKENIZER_SENTINEL_THRESHOLD = 1_000_000


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


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def declared_context_limit(model: Any) -> int | None:
    """
    The context length a loaded model declares, or ``None`` when it declares none.

    This is the generation limit: ``max_position_embeddings`` from the model's
    own config. The first of ``model.config``, ``model.args`` and
    ``model.model_config`` that carries a usable value wins. A tokenizer's
    ``model_max_length`` does not constrain it (that rule is the embedding
    truncation length of ``resolve_max_length``), and a value nested under
    ``text_config`` is not read.
    """
    for attr in ("config", "args", "model_config"):
        limit = _positive_int(
            _config_get(getattr(model, attr, None), "max_position_embeddings")
        )
        if limit is not None:
            return limit
    return None


def resolve_max_length(
    config: Any,
    tokenizer: Any,
    *,
    default: int = MAX_LENGTH_DEFAULT,
    sentinel_threshold: int = TOKENIZER_SENTINEL_THRESHOLD,
    ceiling: int | None = None,
) -> int:
    """
    Resolve the tokenizer truncation length for a model.

    Source order:
      1. ``config.max_position_embeddings`` — an explicit value supplied by
         the model's own architecture.
      2. A finite ``tokenizer.model_max_length`` further constrains the
         architecture value. This matters for RoBERTa-family models, whose
         position table includes reserved padding positions.
      3. ``default``, when neither source yields a usable value.
      4. ``ceiling`` — when set, clamps the resolved value down (never up),
         letting an operator impose a lower deployment-wide limit than a
         model's own declared context window.

    Args:
        config: Model config as a dict (reranker) or object (embeddings).
        tokenizer: The tokenizer (possibly wrapping an inner ``_tokenizer``).
        default: Fallback when no usable value is found.
        sentinel_threshold: Tokenizer-derived values at or above this are
            treated as an unset HuggingFace sentinel, not a real length.
        ceiling: Optional operator-configured upper bound. ``None`` (the
            default) leaves the model-derived value uncapped.

    Returns:
        The truncation length to pass as ``max_length``.
    """
    config_val = _positive_int(_config_get(config, "max_position_embeddings"))
    tok_val = _positive_int(
        getattr(inner_tokenizer(tokenizer), "model_max_length", None)
    )
    if tok_val is not None and tok_val < sentinel_threshold:
        resolved = min(config_val, tok_val) if config_val is not None else tok_val
    else:
        resolved = config_val
    if resolved is None:
        resolved = default
    if ceiling is not None:
        resolved = min(resolved, ceiling)
    return resolved
