# SPDX-License-Identifier: Apache-2.0
"""Tests for tokenizer utility helpers."""

import types
from unittest.mock import patch


def test_load_model_with_fallback_returns_successful_load_result():
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    fake_model = object()
    fake_tokenizer = object()
    fake_mlx_lm = types.SimpleNamespace(
        load=lambda *args, **kwargs: (fake_model, fake_tokenizer)
    )

    with (
        patch("vllm_mlx.utils.tokenizer._needs_tokenizer_fallback", return_value=False),
        patch("vllm_mlx.utils.tokenizer._needs_strict_false", return_value=False),
        patch("vllm_mlx.utils.tokenizer._try_inject_mtp_post_load"),
        patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm}),
    ):
        model, tokenizer = load_model_with_fallback("mlx-community/Qwen3.5-4B")

    assert model is fake_model
    assert tokenizer is fake_tokenizer


def test_load_model_with_fallback_uses_tokenizer_fallback_for_tokenizer_errors():
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    fake_model = object()
    fake_tokenizer = object()

    def _raise(*args, **kwargs):
        raise ValueError("Tokenizer class Foo does not exist")

    fake_mlx_lm = types.SimpleNamespace(load=_raise)

    with (
        patch("vllm_mlx.utils.tokenizer._needs_tokenizer_fallback", return_value=False),
        patch("vllm_mlx.utils.tokenizer._needs_strict_false", return_value=False),
        patch("vllm_mlx.utils.tokenizer._try_inject_mtp_post_load"),
        patch(
            "vllm_mlx.utils.tokenizer._load_with_tokenizer_fallback",
            return_value=(fake_model, fake_tokenizer),
        ) as fallback,
        patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm}),
    ):
        model, tokenizer = load_model_with_fallback("example/model")

    fallback.assert_called_once_with("example/model")
    assert model is fake_model
    assert tokenizer is fake_tokenizer
