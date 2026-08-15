# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared truncation-length resolver."""

from types import SimpleNamespace

from vllm_mlx.utils.truncation import (
    MAX_LENGTH_DEFAULT,
    TOKENIZER_SENTINEL_THRESHOLD,
    inner_tokenizer,
    resolve_max_length,
)


class _Tok:
    """Minimal tokenizer stub with a configurable model_max_length."""

    def __init__(self, model_max_length=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length


def test_resolve_from_dict_config():
    """Positive: reranker-style dict config is honored."""
    cfg = {"max_position_embeddings": 8192}
    assert resolve_max_length(cfg, _Tok()) == 8192


def test_resolve_from_object_config():
    """Positive: embeddings-style object config is honored."""
    cfg = SimpleNamespace(max_position_embeddings=2048)
    assert resolve_max_length(cfg, _Tok()) == 2048


def test_resolve_trusts_large_explicit_config_value():
    """Regression: an explicit model config limit is never silently
    reduced, no matter how large."""
    cfg = {"max_position_embeddings": 100000}
    assert resolve_max_length(cfg, _Tok()) == 100000


def test_resolve_uses_finite_tokenizer_limit_as_safe_upper_bound():
    """A finite tokenizer limit constrains the architecture table size."""
    cfg = {"max_position_embeddings": 514}
    assert resolve_max_length(cfg, _Tok(model_max_length=512)) == 512


def test_resolve_falls_back_to_tokenizer():
    """No config value falls back to tokenizer.model_max_length."""
    assert resolve_max_length(None, _Tok(model_max_length=2048)) == 2048


def test_resolve_unwraps_inner_tokenizer():
    """The inner _tokenizer is consulted when present."""
    outer = SimpleNamespace(_tokenizer=_Tok(model_max_length=1024))
    assert resolve_max_length(None, outer) == 1024


def test_resolve_defaults_to_512():
    """Negative: no usable value anywhere returns the 512 default."""
    assert resolve_max_length(None, _Tok()) == MAX_LENGTH_DEFAULT


def test_resolve_ignores_non_int_config_and_rejects_tokenizer_sentinel():
    """Non-int config values are rejected and fall through to the
    tokenizer; a huge tokenizer sentinel is rejected too, in favor of the
    default -- only the tokenizer-derived value is sentinel-checked."""
    cfg = {"max_position_embeddings": "nope"}
    tok = _Tok(model_max_length=10**30)
    assert resolve_max_length(cfg, tok) == MAX_LENGTH_DEFAULT


def test_resolve_rejects_bool():
    """bool is a subclass of int but must not be treated as a length."""
    cfg = {"max_position_embeddings": True}
    assert resolve_max_length(cfg, _Tok()) == MAX_LENGTH_DEFAULT


def test_resolve_config_value_bypasses_sentinel_threshold():
    """An explicit config value at or above the tokenizer sentinel
    threshold is still trusted -- sentinel detection only applies to the
    tokenizer-derived fallback, never to an explicit architecture value."""
    cfg = {"max_position_embeddings": TOKENIZER_SENTINEL_THRESHOLD + 1}
    assert resolve_max_length(cfg, _Tok()) == TOKENIZER_SENTINEL_THRESHOLD + 1


def test_resolve_tokenizer_value_rejected_at_sentinel_threshold():
    """A tokenizer-derived value at or above the sentinel threshold is
    treated as an unset HuggingFace placeholder, not a real length."""
    tok = _Tok(model_max_length=TOKENIZER_SENTINEL_THRESHOLD)
    assert resolve_max_length(None, tok) == MAX_LENGTH_DEFAULT


def test_resolve_honors_default_override():
    """default is a caller-supplied override."""
    assert resolve_max_length(None, _Tok(), default=256) == 256


def test_resolve_ceiling_clamps_config_value():
    """A ceiling below the model-derived value clamps it down."""
    cfg = {"max_position_embeddings": 8192}
    assert resolve_max_length(cfg, _Tok(), ceiling=1024) == 1024


def test_resolve_ceiling_is_noop_when_above_resolved_value():
    """A ceiling above (or equal to) the resolved value changes nothing."""
    cfg = {"max_position_embeddings": 512}
    assert resolve_max_length(cfg, _Tok(), ceiling=4096) == 512
    assert resolve_max_length(cfg, _Tok(), ceiling=512) == 512


def test_resolve_ceiling_none_is_unchanged():
    """ceiling=None (the default) preserves pre-existing behavior exactly."""
    cfg = {"max_position_embeddings": 8192}
    assert resolve_max_length(cfg, _Tok(), ceiling=None) == resolve_max_length(
        cfg, _Tok()
    )


def test_inner_tokenizer_unwraps_and_passes_through():
    """inner_tokenizer returns the wrapped tokenizer, else the object itself."""
    inner = _Tok()
    assert inner_tokenizer(SimpleNamespace(_tokenizer=inner)) is inner
    plain = _Tok()
    assert inner_tokenizer(plain) is plain
