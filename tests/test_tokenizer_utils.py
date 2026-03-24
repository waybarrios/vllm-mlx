# SPDX-License-Identifier: Apache-2.0
"""Tests for tokenizer utility helpers."""

import platform
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class TestLoadModelWithFallback:
    def test_returns_successful_load_result(self):
        from vllm_mlx.utils.tokenizer import load_model_with_fallback

        fake_model = object()
        fake_tokenizer = object()

        with patch("mlx_lm.load", return_value=(fake_model, fake_tokenizer)) as load:
            model, tokenizer = load_model_with_fallback("mlx-community/Qwen3.5-4B")

        load.assert_called_once()
        assert model is fake_model
        assert tokenizer is fake_tokenizer

    def test_uses_tokenizer_fallback_for_tokenizer_errors(self):
        from vllm_mlx.utils.tokenizer import load_model_with_fallback

        fake_model = object()
        fake_tokenizer = object()

        with patch(
            "mlx_lm.load",
            side_effect=ValueError("Tokenizer class Foo does not exist"),
        ), patch(
            "vllm_mlx.utils.tokenizer._load_with_tokenizer_fallback",
            return_value=(fake_model, fake_tokenizer),
        ) as fallback:
            model, tokenizer = load_model_with_fallback("example/model")

        fallback.assert_called_once_with("example/model")
        assert model is fake_model
        assert tokenizer is fake_tokenizer
