# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Qwen3.5 MLLM attention patch."""

import sys
import types
from typing import Any, cast

import mlx.core as mx

from vllm_mlx.patches.qwen3_5_mllm import patch_qwen35_attention_for_batching


def _position_ids(seq_len: int) -> mx.array:
    base = mx.arange(seq_len).reshape(1, 1, seq_len)
    return mx.tile(base, (3, 1, 1))


def _install_fake_qwen35_modules(monkeypatch):
    call_log: list[dict[str, int]] = []

    class _IdentityNorm:
        def __call__(self, x: mx.array) -> mx.array:
            return x

    class DummyCache:
        def __init__(self, offset=0):
            self.offset = offset

        def update_and_fetch(self, keys: mx.array, values: mx.array):
            return keys, values

    class DummyAttention:
        def __init__(self):
            self.num_attention_heads = 16
            self.num_key_value_heads = 16
            self.head_dim = 64
            self.scale = 1.0

            self.q_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.num_attention_heads * self.head_dim * 2),
                dtype=x.dtype,
            )
            self.k_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.num_key_value_heads * self.head_dim),
                dtype=x.dtype,
            )
            self.v_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.num_key_value_heads * self.head_dim),
                dtype=x.dtype,
            )
            self.q_norm = _IdentityNorm()
            self.k_norm = _IdentityNorm()
            self.o_proj = lambda x: x

            def _rotary_emb(values: mx.array, position_ids: mx.array):
                if position_ids.ndim == 3:
                    batch = int(position_ids.shape[1])
                else:
                    batch = int(position_ids.shape[0])
                seq_len = int(position_ids.shape[-1])
                head_dim = int(values.shape[-1])
                cos = mx.ones((batch, seq_len, head_dim), dtype=values.dtype)
                sin = mx.zeros((batch, seq_len, head_dim), dtype=values.dtype)
                return cos, sin

            self.rotary_emb = _rotary_emb

    def fake_apply_multimodal_rotary_pos_emb(
        queries: mx.array,
        keys: mx.array,
        cos: mx.array,
        sin: mx.array,
    ):
        del sin
        cos_expanded = mx.expand_dims(cos, axis=1)
        q_len = int(queries.shape[-2])
        cos_len = int(cos_expanded.shape[-2])
        call_log.append({"q_len": q_len, "cos_len": cos_len})

        if q_len != cos_len:
            raise ValueError(
                f"[broadcast_shapes] Shapes {tuple(queries.shape)} and "
                f"{tuple(cos_expanded.shape)} cannot be broadcast."
            )

        return queries, keys

    def fake_sdpa(queries: mx.array, keys: mx.array, values: mx.array, **kwargs):
        del keys, values, kwargs
        return queries

    fake_qwen35_mod = types.ModuleType("mlx_vlm.models.qwen3_5.language")
    setattr(fake_qwen35_mod, "Qwen3_5Attention", DummyAttention)
    setattr(
        fake_qwen35_mod,
        "apply_multimodal_rotary_pos_emb",
        fake_apply_multimodal_rotary_pos_emb,
    )

    fake_lm_base_mod = types.ModuleType("mlx_lm.models.base")
    setattr(fake_lm_base_mod, "scaled_dot_product_attention", fake_sdpa)

    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.language", fake_qwen35_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", fake_lm_base_mod)

    return DummyAttention, DummyCache, call_log


def test_qwen35_patch_generates_position_ids_when_missing(monkeypatch):
    attention_cls, cache_cls, call_log = _install_fake_qwen35_modules(monkeypatch)

    assert patch_qwen35_attention_for_batching() is True

    attn = cast(Any, attention_cls())
    x = mx.zeros((1, 11, 1024), dtype=mx.float32)
    cache = cache_cls(offset=0)

    out = attn(x, cache=cache, position_ids=None)

    assert out.shape[1] == 11
    assert call_log[-1] == {"q_len": 11, "cos_len": 11}


def test_qwen35_patch_accepts_matching_position_ids(monkeypatch):
    attention_cls, cache_cls, call_log = _install_fake_qwen35_modules(monkeypatch)

    assert patch_qwen35_attention_for_batching() is True

    attn = cast(Any, attention_cls())
    x = mx.zeros((1, 28, 1024), dtype=mx.float32)
    cache = cache_cls(offset=0)

    out = attn(x, cache=cache, position_ids=_position_ids(28))

    assert out.shape[1] == 28
    assert call_log[-1] == {"q_len": 28, "cos_len": 28}


def test_qwen35_patch_recovers_from_stale_position_ids_between_requests(monkeypatch):
    """Second call must not reuse shorter position_ids from a prior request."""
    attention_cls, cache_cls, call_log = _install_fake_qwen35_modules(monkeypatch)

    assert patch_qwen35_attention_for_batching() is True

    attn = cast(Any, attention_cls())
    cache = cache_cls(offset=0)

    short_position_ids = _position_ids(11)

    first_x = mx.zeros((1, 11, 1024), dtype=mx.float32)
    attn(first_x, cache=cache, position_ids=short_position_ids)

    second_x = mx.zeros((1, 28, 1024), dtype=mx.float32)
    out = attn(second_x, cache=cache, position_ids=short_position_ids)

    assert out.shape[1] == 28
    assert call_log[-1] == {"q_len": 28, "cos_len": 28}
