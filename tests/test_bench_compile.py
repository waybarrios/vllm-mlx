# SPDX-License-Identifier: Apache-2.0
"""Integration test for mx.compile wrapper with mlx_lm models."""

import os

import pytest

try:
    import mlx.core as mx
    from mlx_lm import load

    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

from vllm_mlx.compile import apply_compile, is_compiled


@pytest.mark.skipif(not HAS_MLX_LM, reason="mlx_lm not available")
class TestCompileWithRealModel:
    @pytest.fixture
    def model_and_tokenizer(self):
        model_name = os.environ.get(
            "VLLM_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-8bit"
        )
        try:
            model, tokenizer = load(model_name)
            return model, tokenizer
        except Exception:
            pytest.skip(f"Model {model_name} not available")

    def test_compile_real_model_produces_output(self, model_and_tokenizer):
        from mlx_lm import stream_generate

        model, tokenizer = model_and_tokenizer

        baseline_tokens = []
        for resp in stream_generate(
            model, tokenizer, "Hello", max_tokens=5, temp=0.0
        ):
            baseline_tokens.append(resp.token)
            if resp.finish_reason:
                break

        apply_compile(model)
        assert is_compiled(model)

        compiled_tokens = []
        for resp in stream_generate(
            model, tokenizer, "Hello", max_tokens=5, temp=0.0
        ):
            compiled_tokens.append(resp.token)
            if resp.finish_reason:
                break

        assert len(baseline_tokens) > 0
        assert len(compiled_tokens) > 0
