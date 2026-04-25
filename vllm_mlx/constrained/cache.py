# SPDX-License-Identifier: Apache-2.0
"""
Cache of ``TokenEnforcerTokenizerData`` objects keyed by tokenizer identity.

Building ``TokenEnforcerTokenizerData`` requires iterating over the entire
vocabulary (up to 200k tokens on MiniMax/GLM) and decoding each token.  The
cost is ~1-2 seconds per model and the result is independent of the JSON
schema, so we cache it for the lifetime of the process.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Keyed by ``id(tokenizer)`` because tokenizer instances are not hashable
# across HF/MLX wrappers, but each server loads a single tokenizer per model.
_CACHE: dict[int, Any] = {}
_CACHE_LOCK = threading.Lock()


def _resolve_inner_tokenizer(tokenizer: Any) -> Any:
    """
    VLM processors wrap the actual tokenizer under ``processor.tokenizer``.
    ``mlx_lm.tokenizer_utils.TokenizerWrapper`` exposes it via ``_tokenizer``.
    Return the most-unwrapped tokenizer that still has the HF
    ``all_special_ids`` / ``eos_token_id`` surface.

    Note: on HF ``PreTrainedTokenizerFast``, ``_tokenizer`` points at the
    rust-level object which lacks ``all_special_ids``; unwrapping to that
    level would cause every special token (``<eos>``, ``<pad>``, ``\\n``,
    ``<|think|>`` …) to leak into ``regular_tokens`` and end up in
    ``TokenizerPrefixTree.root`` as an always-allowed token.  We only
    unwrap when the inner layer still exposes ``all_special_ids``.
    """
    # VLM processor wrapper exposes the HF tokenizer under ``tokenizer``.
    inner = getattr(tokenizer, "tokenizer", None)
    if (
        inner is not None
        and inner is not tokenizer
        and hasattr(inner, "all_special_ids")
    ):
        tokenizer = inner
    # mlx_lm TokenizerWrapper keeps the raw HF tokenizer under ``_tokenizer``.
    # Only unwrap if the inner object still exposes the HF tokenizer surface.
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "all_special_ids"):
        tokenizer = inner
    return tokenizer


def _build_regular_tokens_list(
    tokenizer: Any, vocab_size: int
) -> list[tuple[int, str, bool]]:
    """
    Enumerate the regular (non-special) tokens in the vocabulary and produce
    the ``(token_id, decoded_with_leading_space_marker, is_word_start)`` tuples
    required by ``TokenEnforcerTokenizerData``.

    Mirrors the reference implementation in ``lmformatenforcer.integrations.
    transformers`` but works with the HF tokenizer surface only (so we do not
    need a hard transformers dependency at the right version).
    """
    try:
        special_ids = set(tokenizer.all_special_ids)
    except AttributeError:
        special_ids = set()

    try:
        token_0 = tokenizer.encode("0")[-1]
    except Exception:
        token_0 = None

    regular_tokens: list[tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        if token_idx in special_ids:
            continue
        try:
            decoded_regular = tokenizer.decode([token_idx])
        except Exception:
            continue
        if token_0 is not None:
            try:
                decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
            except Exception:
                decoded_after_0 = decoded_regular
        else:
            decoded_after_0 = decoded_regular
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def _get_eos_token_id(tokenizer: Any) -> int | list[int]:
    # Some tokenizers expose multiple EOS candidates (e.g. Gemma 4 has
    # [1 <eos>, 106 <end_of_turn>, 50 <|think|>] in generation_config.json).
    # Prefer the list form so all stop tokens are treated as EOS by the
    # enforcer; otherwise the model may emit an out-of-schema stop token
    # that the enforcer did not mask (because it's a special token not in
    # ``regular_tokens``), yet the inference runtime still treats as stop.
    eos_list = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_list, (list, tuple)) and eos_list:
        return list(eos_list)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return eos
    return 0


def _get_vocab_size(tokenizer: Any) -> int:
    vs = getattr(tokenizer, "vocab_size", None)
    if isinstance(vs, int) and vs > 0:
        return vs
    try:
        return len(tokenizer)
    except TypeError:
        pass
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return len(get_vocab())
    raise ValueError("Cannot determine tokenizer vocab size")


def _decode_function(tokenizer: Any, tokens: list[int]) -> str:
    try:
        decoded = tokenizer.decode(tokens)
    except Exception:
        return ""
    return decoded.rstrip("\ufffd") if isinstance(decoded, str) else ""


def get_tokenizer_data(tokenizer: Any) -> Any | None:
    """
    Return a cached ``TokenEnforcerTokenizerData`` for ``tokenizer``.

    Returns ``None`` if ``lm-format-enforcer`` is not installed or the
    tokenizer cannot be adapted.
    """
    try:
        from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    except ImportError:
        return None

    inner = _resolve_inner_tokenizer(tokenizer)
    key = id(inner)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    try:
        vocab_size = _get_vocab_size(inner)
    except Exception as exc:
        logger.warning(
            "Could not determine vocab size for constrained decoding: %s", exc
        )
        return None

    try:
        regular_tokens = _build_regular_tokens_list(inner, vocab_size)
        decode_fn = functools.partial(_decode_function, inner)
        eos_token_id = _get_eos_token_id(inner)
        data = TokenEnforcerTokenizerData(
            regular_tokens,
            decode_fn,
            eos_token_id,
            False,  # use_bitmask
            vocab_size,
        )
    except Exception as exc:
        logger.warning("Failed to build TokenEnforcerTokenizerData: %s", exc)
        return None

    with _CACHE_LOCK:
        _CACHE[key] = data
    return data


def clear_cache() -> None:
    """Drop the cache (mainly for tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()
