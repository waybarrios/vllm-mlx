# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Gemma 4 MLLM attention patch."""

import sys
import types
from typing import Any

import mlx.core as mx


def _install_fake_gemma4_modules(monkeypatch):
    """Stub mlx_vlm.models.gemma4 / .base so the patch can install without
    pulling in real weights or compilation."""

    base_module = types.ModuleType("mlx_vlm.models.base")

    def fake_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None):
        return queries

    base_module.scaled_dot_product_attention = fake_sdpa

    language_module = types.ModuleType("mlx_vlm.models.gemma4.language")

    class _IdentityNorm:
        def __call__(self, x):
            return x

    class DummyRope:
        def __call__(self, x, offset=0):
            return x

    class FakeGemma4Attention:
        n_heads = 8
        n_kv_heads = 8
        head_dim = 64
        scale = 1.0
        use_k_eq_v = False
        is_kv_shared_layer = False

        def __init__(self):
            self.q_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.n_heads * self.head_dim), dtype=x.dtype
            )
            self.k_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.n_kv_heads * self.head_dim), dtype=x.dtype
            )
            self.v_proj = lambda x: mx.zeros(
                (x.shape[0], x.shape[1], self.n_kv_heads * self.head_dim), dtype=x.dtype
            )
            self.q_norm = _IdentityNorm()
            self.k_norm = _IdentityNorm()
            self.v_norm = _IdentityNorm()
            self.rope = DummyRope()
            self.o_proj = lambda x: x

        def __call__(self, *args, **kwargs):
            raise AssertionError("unpatched __call__ should never run")

    language_module.Attention = FakeGemma4Attention

    gemma4_module = types.ModuleType("mlx_vlm.models.gemma4")
    gemma4_module.language = language_module
    models_module = types.ModuleType("mlx_vlm.models")
    models_module.base = base_module
    models_module.gemma4 = gemma4_module
    mlx_vlm_module = types.ModuleType("mlx_vlm")
    mlx_vlm_module.models = models_module

    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.base", base_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma4", gemma4_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma4.language", language_module)

    return FakeGemma4Attention


def test_patched_call_accepts_shared_kv_and_offset_kwargs(monkeypatch):
    """Patched __call__ accepts mlx-vlm 0.5.0 kwargs and returns a 3-tuple."""
    Attention = _install_fake_gemma4_modules(monkeypatch)

    from vllm_mlx.patches.gemma4_mllm import patch_gemma4_attention_for_batching

    if hasattr(Attention, "_batch_patched"):
        delattr(Attention, "_batch_patched")
    Attention.__call__ = Attention.__dict__["__call__"]

    assert patch_gemma4_attention_for_batching() is True
    assert Attention._batch_patched is True

    attn = Attention()
    x = mx.zeros((1, 4, Attention.n_heads * Attention.head_dim))
    shared = (
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
    )
    result = attn(x, mask=None, cache=None, shared_kv=shared, offset=mx.array([0]))
    assert isinstance(result, tuple) and len(result) == 3
    _output, kv, _offset = result
    assert isinstance(kv, tuple) and len(kv) == 2


def test_patched_call_snapshots_offset_before_update(monkeypatch):
    """RoPE on queries must see the pre-update offset, even with BatchKVCache."""
    Attention = _install_fake_gemma4_modules(monkeypatch)
    from vllm_mlx.patches.gemma4_mllm import patch_gemma4_attention_for_batching

    if hasattr(Attention, "_batch_patched"):
        delattr(Attention, "_batch_patched")
    Attention.__call__ = Attention.__dict__["__call__"]

    patch_gemma4_attention_for_batching()

    captured_offsets: list[Any] = []

    class CapturingRope:
        def __call__(self, x, offset=0):
            captured_offsets.append(offset)
            return x

    class MutatingBatchCache:
        # Mimics BatchKVCache: offset is mx.array, update_and_fetch advances it.
        def __init__(self):
            self.offset = mx.array([5])

        def update_and_fetch(self, keys, values):
            self.offset = self.offset + 1
            return keys, values

    attn = Attention()
    attn.rope = CapturingRope()
    cache = MutatingBatchCache()

    x = mx.zeros((1, 1, Attention.n_heads * Attention.head_dim))
    attn(x, mask=None, cache=cache)

    assert len(captured_offsets) == 2  # keys + queries
    for o in captured_offsets:
        assert int(o.tolist()[0]) == 5  # pre-update, not 6


def test_patched_call_uses_shared_kv_when_provided(monkeypatch):
    """shared_kv branch skips k_proj entirely and passes (k, v) through."""
    Attention = _install_fake_gemma4_modules(monkeypatch)
    from vllm_mlx.patches.gemma4_mllm import patch_gemma4_attention_for_batching

    if hasattr(Attention, "_batch_patched"):
        delattr(Attention, "_batch_patched")
    Attention.__call__ = Attention.__dict__["__call__"]
    patch_gemma4_attention_for_batching()

    attn = Attention()

    k_proj_calls = []
    orig_k_proj = attn.k_proj

    def spy_k_proj(x):
        k_proj_calls.append(x.shape)
        return orig_k_proj(x)

    attn.k_proj = spy_k_proj

    x = mx.zeros((1, 4, Attention.n_heads * Attention.head_dim))
    shared = (
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
    )
    _output, kv, _offset = attn(
        x, mask=None, cache=None, shared_kv=shared, offset=mx.array([3])
    )
    assert k_proj_calls == []
    assert kv[0] is shared[0]
    assert kv[1] is shared[1]
