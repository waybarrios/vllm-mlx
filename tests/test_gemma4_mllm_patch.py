# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Gemma 4 MLLM attention patch.

Two distinct properties this test guards against:

1. **Signature compatibility with mlx-vlm 0.5.0+.** The patched call must
   accept ``shared_kv: Optional[tuple]`` and ``offset: Optional[Any]`` and
   return a 3-tuple ``(output, (keys, values), offset)``. The pre-fix
   version of this patch only accepted ``(x, mask, cache)`` and returned a
   single ``mx.array`` — every Gemma 4 request under
   ``--continuous-batching`` then crashed with::

       TypeError: patch_gemma4_attention_for_batching.<locals>._patched_call()
       got an unexpected keyword argument 'shared_kv'

2. **Offset snapshot before update_and_fetch (the original reason).**
   ``BatchKVCache`` holds offset as an ``mx.array``; the patched function
   must snapshot it BEFORE ``cache.update_and_fetch`` mutates it in place,
   so the RoPE applied to the queries afterwards still sees the pre-update
   value.
"""

import sys
import types
from typing import Any

import mlx.core as mx


def _install_fake_gemma4_modules(monkeypatch):
    """Replace mlx_vlm.models.gemma4 / .base in sys.modules with fakes whose
    ``Attention`` class accepts arbitrary kwargs and records its call.

    Lets us patch + invoke the wrapper without pulling in real mlx-vlm
    weights or compilation. The fakes are scoped via monkeypatch so they
    disappear after the test.
    """

    base_module = types.ModuleType("mlx_vlm.models.base")

    def fake_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None):
        # Return the queries unchanged — the test only cares about the
        # return-shape contract, not numeric correctness.
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
        """Minimal stand-in for mlx_vlm.models.gemma4.language.Attention.

        We do not redefine ``__call__`` here — the patch will inject the
        replacement. The orig call (used by the patch's _orig_call capture)
        is a simple identity so the patch installs cleanly.
        """

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

        def __call__(self, *args, **kwargs):  # noqa: D401
            raise AssertionError(
                "Unpatched __call__ should never run — the patch should "
                "have replaced it before the test calls it."
            )

    language_module.Attention = FakeGemma4Attention

    # Stitch the package hierarchy so `from mlx_vlm.models.gemma4.language
    # import Attention` resolves.
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
    """The patched __call__ must accept ``shared_kv`` and ``offset`` kwargs
    without raising TypeError. Pre-fix, this raised:

        TypeError: patch_gemma4_attention_for_batching.<locals>._patched_call()
        got an unexpected keyword argument 'shared_kv'
    """
    Attention = _install_fake_gemma4_modules(monkeypatch)

    # Import lazily so we re-import against the freshly faked mlx_vlm.
    from vllm_mlx.patches.gemma4_mllm import patch_gemma4_attention_for_batching

    # Clear the idempotency sentinel so the patch actually runs against our
    # fake class (the real one in production would only patch once).
    if hasattr(Attention, "_batch_patched"):
        delattr(Attention, "_batch_patched")
    Attention.__call__ = Attention.__dict__["__call__"]  # reset to AssertionError stub

    ok = patch_gemma4_attention_for_batching()
    assert ok is True
    assert Attention._batch_patched is True

    # Call with the new mlx-vlm 0.5.0 signature.
    attn = Attention()
    x = mx.zeros((1, 4, Attention.n_heads * Attention.head_dim))
    shared = (
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
        mx.zeros((1, Attention.n_kv_heads, 4, Attention.head_dim)),
    )
    result = attn(x, mask=None, cache=None, shared_kv=shared, offset=mx.array([0]))
    assert isinstance(result, tuple), "Patched __call__ must return a tuple"
    assert (
        len(result) == 3
    ), "Patched __call__ must return a 3-tuple (output, kv, offset)"
    output, kv, offset_out = result
    assert (
        isinstance(kv, tuple) and len(kv) == 2
    ), "Middle return slot must be (k, v) pair"


def test_patched_call_snapshots_offset_before_update(monkeypatch):
    """The patched __call__ must snapshot cache.offset BEFORE calling
    ``cache.update_and_fetch``, so the offset used for RoPE on queries
    reflects the *pre-update* state.

    Reproduces the bug where BatchKVCache's mx.array offset is mutated
    in-place by update_and_fetch — a naive read-after-update reads the
    advanced offset and applies the wrong RoPE position.
    """
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
        """Mimics BatchKVCache: offset is an mx.array, and update_and_fetch
        advances it in place via __iadd__ (which mx.array supports)."""

        def __init__(self):
            self.offset = mx.array([5])

        def update_and_fetch(self, keys, values):
            self.offset = self.offset + 1  # advance — pretend we appended a token
            return keys, values

    attn = Attention()
    attn.rope = CapturingRope()
    cache = MutatingBatchCache()

    x = mx.zeros((1, 1, Attention.n_heads * Attention.head_dim))
    attn(x, mask=None, cache=cache)  # no shared_kv → exercises the snapshot branch

    # captured_offsets[0] is the offset passed to RoPE for keys (pre-update),
    # captured_offsets[1] is for queries (must also be pre-update). Both
    # must equal 5, NOT the post-update 6.
    assert len(captured_offsets) == 2, "RoPE should be invoked twice (keys + queries)"
    for o in captured_offsets:
        # mx.array equality returns an array — compare to scalar 5 explicitly.
        assert (
            int(o.tolist()[0]) == 5
        ), "offset snapshot regression: RoPE saw a post-update offset"


def test_patched_call_uses_shared_kv_when_provided(monkeypatch):
    """When shared_kv is provided, the patched __call__ should skip the
    k_proj / v_proj / k_norm / v_norm / cache.update_and_fetch path entirely
    and reuse the supplied (keys, values).
    """
    Attention = _install_fake_gemma4_modules(monkeypatch)
    from vllm_mlx.patches.gemma4_mllm import patch_gemma4_attention_for_batching

    if hasattr(Attention, "_batch_patched"):
        delattr(Attention, "_batch_patched")
    Attention.__call__ = Attention.__dict__["__call__"]
    patch_gemma4_attention_for_batching()

    attn = Attention()

    # Spy on k_proj to confirm it's NOT called in the shared_kv branch.
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
    output, kv, offset_out = attn(
        x, mask=None, cache=None, shared_kv=shared, offset=mx.array([3])
    )
    assert k_proj_calls == [], "k_proj should be skipped when shared_kv is provided"
    # Inner arrays must be the SAME objects we passed in — confirming no
    # recompute happened (only the outer 2-tuple wrapper is new).
    assert kv[0] is shared[0], "shared_kv[0] should be passed through unchanged"
    assert kv[1] is shared[1], "shared_kv[1] should be passed through unchanged"
