# SPDX-License-Identifier: Apache-2.0
"""Strict JSON Schema token masking backed by ``llguidance``."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .json_schema_processor import ConstrainedDecodingError

_MAX_NONPROGRESS_WHITESPACE_CHARS = 256
_tokenizer_cache: dict[int, tuple[Any, Any, frozenset[int]]] = {}


def is_available() -> bool:
    """Return whether the strict JSON Schema backend is importable."""
    try:
        import llguidance  # noqa: F401
        import llguidance.hf  # noqa: F401
        import llguidance.mlx  # noqa: F401
    except ImportError:
        return False
    return True


def _token_list(tokens: mx.array) -> list[int]:
    values = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    if isinstance(values, int):
        return [values]
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token) for token in values]


def _ll_tokenizer(tokenizer: Any):
    import llguidance.hf

    candidates = [tokenizer]
    for attribute in ("tokenizer", "_tokenizer"):
        candidate = getattr(tokenizer, attribute, None)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    errors: list[str] = []
    for candidate in candidates:
        cached = _tokenizer_cache.get(id(candidate))
        if cached is not None and cached[0] is candidate:
            return cached
        try:
            ll_tokenizer = llguidance.hf.from_tokenizer(candidate)
        except (TypeError, ValueError) as exc:
            errors.append(f"{type(candidate).__name__}: {exc}")
            continue
        whitespace_tokens = frozenset(
            token_id
            for token_id in range(ll_tokenizer.vocab_size)
            if _decoded_token_is_whitespace(candidate, token_id)
        )
        result = (candidate, ll_tokenizer, whitespace_tokens)
        _tokenizer_cache[id(candidate)] = result
        return result
    detail = "; ".join(errors)
    raise ConstrainedDecodingError(
        "strict JSON Schema requires a fast Hugging Face tokenizer"
        + (f" ({detail})" if detail else "")
    )


def _decoded_token_is_whitespace(tokenizer: Any, token_id: int) -> bool:
    try:
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([token_id])
    except Exception:
        return False
    return bool(text) and text.isspace()


def _trailing_json_whitespace(text: str) -> int:
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
    if in_string:
        return 0
    return len(text) - len(text.rstrip())


def _clear_mask_token(mask: np.ndarray, token_id: int) -> None:
    word_index, bit_index = divmod(token_id, 32)
    if word_index >= mask.shape[1]:
        return
    value = int(np.uint32(mask[0, word_index])) & ~(1 << bit_index)
    mask[0, word_index] = np.array(value, dtype=np.uint32).view(np.int32)


class LLGuidanceJSONSchemaLogitsProcessor:
    """Apply a request-local JSON Schema grammar to MLX logits."""

    def __init__(self, schema: dict, tokenizer: Any) -> None:
        if not is_available():
            raise ConstrainedDecodingError(
                "llguidance is required for strict JSON Schema response_format"
            )

        import llguidance

        self._tokenizer, ll_tokenizer, whitespace_tokens = _ll_tokenizer(tokenizer)
        self._vocab_size = ll_tokenizer.vocab_size
        self._whitespace_tokens = whitespace_tokens
        eos_tokens = ll_tokenizer.eos_tokens
        self._eos_tokens = {int(token) for token in eos_tokens}
        try:
            grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
            self._matcher = llguidance.LLMatcher(ll_tokenizer, grammar)
        except Exception as exc:
            raise ConstrainedDecodingError(
                f"failed to compile strict JSON Schema grammar: {exc}"
            ) from exc

        self._prompt_len: int | None = None
        self._consumed_suffix: list[int] = []
        self._terminal = False

    def _consume_suffix(self, suffix: list[int]) -> None:
        previous = self._consumed_suffix
        if len(suffix) >= len(previous) and suffix[: len(previous)] == previous:
            delta = suffix[len(previous) :]
        else:
            self._matcher.reset()
            delta = suffix

        if delta and all(token in self._eos_tokens for token in delta):
            if not self._matcher.is_accepting():
                raise ConstrainedDecodingError(
                    "received an EOS token before the JSON Schema was complete"
                )
            self._terminal = True
            self._consumed_suffix = list(suffix)
            return
        if delta and not self._matcher.consume_tokens(delta):
            try:
                decoded_tail = self._tokenizer.decode(suffix[-12:])
            except Exception:
                decoded_tail = "<decode-error>"
            raise ConstrainedDecodingError(
                "generated token prefix violated the declared JSON Schema: "
                f"delta={delta!r}, suffix_tail={suffix[-12:]!r}, "
                f"decoded_tail={decoded_tail!r}"
            )
        if self._matcher.is_error():
            detail = self._matcher.get_error() or self._matcher.stop_reason()
            raise ConstrainedDecodingError(
                f"strict JSON Schema matcher entered an error state: {detail}"
            )
        self._consumed_suffix = list(suffix)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        import llguidance.mlx

        tokens_list = _token_list(tokens)
        if self._prompt_len is None:
            self._prompt_len = len(tokens_list)
        suffix = tokens_list[self._prompt_len :]
        self._consume_suffix(suffix)
        if self._terminal:
            mask = mx.full(logits.shape, -float("inf"))
            for token in self._eos_tokens:
                if token < logits.shape[-1]:
                    if logits.ndim == 1:
                        mask[token] = 0.0
                    else:
                        mask[..., token] = 0.0
            return logits + mask

        try:
            mask = llguidance.mlx.allocate_token_bitmask(1, self._vocab_size)
            llguidance.mlx.fill_next_token_bitmask(self._matcher, mask)
            decoded = self._tokenizer.decode(
                suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if _trailing_json_whitespace(decoded) >= _MAX_NONPROGRESS_WHITESPACE_CHARS:
                for token_id in self._whitespace_tokens:
                    _clear_mask_token(mask, token_id)
            masked = llguidance.mlx.apply_token_bitmask(logits, mask)
        except Exception as exc:
            raise ConstrainedDecodingError(
                f"failed to construct strict JSON Schema token mask: {exc}"
            ) from exc
        return masked[0] if logits.ndim == 1 else masked
