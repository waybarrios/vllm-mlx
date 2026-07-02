# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared truncation-length resolver."""

from functools import partial
from types import SimpleNamespace

from vllm_mlx.utils.truncation import (
    MAX_LENGTH_CAP,
    MAX_LENGTH_DEFAULT,
    inner_tokenizer,
    resolve_max_length,
)

# Resolve with the shared cap/default unless a test overrides them.
resolve = partial(resolve_max_length, cap=MAX_LENGTH_CAP, default=MAX_LENGTH_DEFAULT)


class _Tok:
    """Minimal tokenizer stub with a configurable model_max_length."""

    def __init__(self, model_max_length=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length


def test_resolve_from_dict_config():
    """Positive: reranker-style dict config is honored."""
    cfg = {"max_position_embeddings": 8192}
    assert resolve(cfg, _Tok()) == 8192


def test_resolve_from_object_config():
    """Positive: embeddings-style object config is honored."""
    cfg = SimpleNamespace(max_position_embeddings=2048)
    assert resolve(cfg, _Tok()) == 2048


def test_resolve_caps_at_8192():
    """A model reporting a huge window is capped at 8192."""
    cfg = {"max_position_embeddings": 100000}
    assert resolve(cfg, _Tok()) == MAX_LENGTH_CAP


def test_resolve_falls_back_to_tokenizer():
    """No config value falls back to tokenizer.model_max_length."""
    assert resolve(None, _Tok(model_max_length=2048)) == 2048


def test_resolve_unwraps_inner_tokenizer():
    """The inner _tokenizer is consulted when present."""
    outer = SimpleNamespace(_tokenizer=_Tok(model_max_length=1024))
    assert resolve(None, outer) == 1024


def test_resolve_defaults_to_512():
    """Negative: no usable value anywhere returns the 512 default."""
    assert resolve(None, _Tok()) == MAX_LENGTH_DEFAULT


def test_resolve_ignores_non_int_and_sentinel():
    """Non-int config values are rejected; huge sentinels are capped."""
    # Non-int config -> falls through to a huge tokenizer sentinel -> capped.
    cfg = {"max_position_embeddings": "nope"}
    tok = _Tok(model_max_length=10**30)
    assert resolve(cfg, tok) == MAX_LENGTH_CAP


def test_resolve_rejects_bool():
    """bool is a subclass of int but must not be treated as a length."""
    cfg = {"max_position_embeddings": True}
    assert resolve(cfg, _Tok()) == MAX_LENGTH_DEFAULT


def test_resolve_honors_call_site_overrides():
    """cap/default are caller-supplied, so an engine can override them."""
    # Custom default when nothing usable is found.
    assert resolve_max_length(None, _Tok(), cap=4096, default=256) == 256
    # Custom cap clamps the model's window.
    cfg = {"max_position_embeddings": 100000}
    assert resolve_max_length(cfg, _Tok(), cap=4096, default=256) == 4096


def test_inner_tokenizer_unwraps_and_passes_through():
    """inner_tokenizer returns the wrapped tokenizer, else the object itself."""
    inner = _Tok()
    assert inner_tokenizer(SimpleNamespace(_tokenizer=inner)) is inner
    plain = _Tok()
    assert inner_tokenizer(plain) is plain
