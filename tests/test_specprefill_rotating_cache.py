# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RotatingKVCache handling in sparse_prefill."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class _FakeAttention:
    def __init__(self):
        self.num_heads = 1
        self.q_proj = lambda x: x


class _FakeLayer:
    def __init__(self):
        self.block_type = "*"
        self.mixer = _FakeAttention()


class _FakeModel:
    def __init__(self):
        self.layers = [_FakeLayer()]
        self.calls: list[list[int]] = []

    def __call__(self, x, cache=None):
        self.calls.append(x.tolist())
        logits = mx.zeros((1, x.shape[1], 8), dtype=mx.float32)
        return logits


class RotatingKVCache:
    def __init__(self, max_size: int, keep: int = 0):
        self.max_size = max_size
        self.keep = keep
        self.offset = 0
        self.state = mx.array([0], dtype=mx.float32)


def _run_sparse_prefill(total_tokens: int, selected_indices: list[int], max_size: int):
    from vllm_mlx.specprefill import sparse_prefill

    model = _FakeModel()
    tokens = list(range(total_tokens))
    cache = [RotatingKVCache(max_size=max_size, keep=0)]
    sparse_prefill(
        model,
        tokens,
        selected_indices,
        cache,
        step_size=64,
    )
    return model.calls


def test_sparse_prefill_does_not_expand_tail_when_prompt_fits_window():
    calls = _run_sparse_prefill(
        total_tokens=6,
        selected_indices=[0, 2, 4],
        max_size=8,
    )

    flattened = [token for chunk in calls for row in chunk for token in row]
    assert flattened == [0, 2, 4]


def test_sparse_prefill_expands_tail_when_prompt_exceeds_window():
    calls = _run_sparse_prefill(
        total_tokens=10,
        selected_indices=[0, 2],
        max_size=8,
    )

    flattened = [token for chunk in calls for row in chunk for token in row]
    assert flattened == [0, 2, 3, 4, 5, 6, 7, 8, 9]


def test_prepare_rotating_caches_preserves_absolute_offset():
    """Window normalization must not reset the sequence position."""
    from mlx_lm.models.cache import RotatingKVCache

    from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

    max_size = 4
    cache = RotatingKVCache(max_size=max_size, keep=0)
    # Simulate prefix-restored state: buffer larger than window, offset past max
    cache.keys = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
    cache.values = mx.zeros((1, 1, 8, 4), dtype=mx.float32)
    cache.offset = 8
    cache._idx = 8

    assert MLLMBatchGenerator._prepare_rotating_caches([cache]) is True

    assert cache.keys.shape[2] == max_size
    assert cache.values.shape[2] == max_size
    assert cache._idx == max_size
    assert cache.offset == 8
