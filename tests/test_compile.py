# SPDX-License-Identifier: Apache-2.0
"""Tests for mx.compile wrapper utility."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_mlx.compile import apply_compile, is_compiled


class SimpleModel(nn.Module):
    """Tiny model for testing compilation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def __call__(self, x):
        return self.linear(x)


class TestApplyCompile:
    def test_compile_wraps_model(self):
        """apply_compile returns a model whose __call__ is compiled."""
        model = SimpleModel()
        compiled_model = apply_compile(model)
        x = mx.ones((1, 8))
        original_out = model(x)
        compiled_out = compiled_model(x)
        assert mx.allclose(original_out, compiled_out).item()

    def test_compile_is_idempotent(self):
        """Applying compile twice doesn't double-wrap."""
        model = SimpleModel()
        compiled = apply_compile(model)
        double_compiled = apply_compile(compiled)
        assert compiled is double_compiled

    def test_is_compiled_flag(self):
        """is_compiled returns correct state."""
        model = SimpleModel()
        assert is_compiled(model) is False
        compiled = apply_compile(model)
        assert is_compiled(compiled) is True

    def test_no_compile_returns_original(self):
        """When compile=False, return model unchanged."""
        model = SimpleModel()
        result = apply_compile(model, enabled=False)
        assert result is model
        assert is_compiled(result) is False

    def test_compiled_model_handles_different_shapes(self):
        """shapeless=True means different input shapes don't crash."""
        model = SimpleModel()
        compiled = apply_compile(model)
        out1 = compiled(mx.ones((1, 8)))
        out2 = compiled(mx.ones((4, 8)))
        assert out1.shape == (1, 8)
        assert out2.shape == (4, 8)
