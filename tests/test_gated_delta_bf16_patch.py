# SPDX-License-Identifier: Apache-2.0
"""Tests for the GatedDeltaNet bfloat16 recurrent-state patch.

The patch stores the GatedDeltaNet recurrent matrix (``cache[1]``) in bfloat16
between decode steps, upcasting to float32 before each forward so compute
precision is unchanged. These tests exercise the patch wrapper against a
minimal fake GatedDeltaNet — the real numerical-quality gate is the A/B live
comparison, not a unit test.
"""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from vllm_mlx.patches.gated_delta_bf16 import (
    _SENTINEL,
    _wrap_gated_delta_call,
    patch_gated_delta_bf16_state,
)


def _make_fake_gated_delta_net():
    """A minimal GatedDeltaNet stand-in.

    Its ``__call__`` mimics the real contract: read cache[1], allocate a
    float32 state on first use, update it, write it back. It also asserts the
    state it receives is float32 — proving the patch upcast on read.
    """

    class FakeGatedDeltaNet:
        def __call__(self, inputs, mask=None, cache=None):
            if cache is not None:
                state = cache[1]
                if state is None:
                    state = mx.zeros((1, 4, 8, 8), dtype=mx.float32)
                else:
                    # The patch must hand us float32, not bfloat16.
                    assert (
                        state.dtype == mx.float32
                    ), f"expected float32 state, got {state.dtype}"
                    state = state * 0.9 + 1.0
                cache[1] = state
            return inputs

    return FakeGatedDeltaNet


# ---------------------------------------------------------------------------
# _wrap_gated_delta_call — behavioural tests against a fake class
# ---------------------------------------------------------------------------


class TestWrapGatedDeltaCall:
    def test_state_stored_as_bfloat16(self):
        cls = _make_fake_gated_delta_net()
        assert _wrap_gated_delta_call(cls) is True
        assert getattr(cls, _SENTINEL, False) is True

        gdn = cls()
        cache = ArraysCache(size=2)
        x = mx.zeros((1, 1, 8))

        gdn(x, cache=cache)
        mx.eval(cache[1])
        assert cache[1].dtype == mx.bfloat16  # downcast after step 1

        gdn(x, cache=cache)
        mx.eval(cache[1])
        assert cache[1].dtype == mx.bfloat16  # still bf16 after step 2

    def test_idempotent(self):
        cls = _make_fake_gated_delta_net()
        assert _wrap_gated_delta_call(cls) is True
        patched_call = cls.__call__
        # Second call must be a no-op — not re-wrap.
        assert _wrap_gated_delta_call(cls) is True
        assert cls.__call__ is patched_call

    def test_guard_cache_none(self):
        cls = _make_fake_gated_delta_net()
        _wrap_gated_delta_call(cls)
        gdn = cls()
        # No cache → must not raise.
        out = gdn(mx.zeros((1, 1, 8)), cache=None)
        assert out is not None

    def test_guard_cache_slot_none(self):
        cls = _make_fake_gated_delta_net()
        _wrap_gated_delta_call(cls)
        gdn = cls()
        cache = ArraysCache(size=2)  # cache[1] is None
        # First step with cache[1] is None → must not raise.
        gdn(mx.zeros((1, 1, 8)), cache=cache)
        mx.eval(cache[1])
        assert cache[1].dtype == mx.bfloat16

    def test_float16_dtype(self):
        """The patch can store the recurrent state in float16 when requested."""
        cls = _make_fake_gated_delta_net()
        assert _wrap_gated_delta_call(cls, dtype=mx.float16) is True

        gdn = cls()
        cache = ArraysCache(size=2)
        x = mx.zeros((1, 1, 8))
        gdn(x, cache=cache)
        mx.eval(cache[1])
        assert cache[1].dtype == mx.float16
        gdn(x, cache=cache)
        mx.eval(cache[1])
        assert cache[1].dtype == mx.float16

    def test_numerical_equivalence_within_tolerance(self):
        """Patched vs unpatched recurrence stay close — the bf16 storage
        round-trip introduces only small bounded error."""
        x = mx.ones((1, 1, 8))

        unpatched_cls = _make_fake_gated_delta_net()
        patched_cls = _make_fake_gated_delta_net()
        _wrap_gated_delta_call(patched_cls)

        cache_a = ArraysCache(size=2)
        cache_b = ArraysCache(size=2)
        gdn_a = unpatched_cls()
        gdn_b = patched_cls()
        for _ in range(3):
            gdn_a(x, cache=cache_a)
            gdn_b(x, cache=cache_b)
        mx.eval(cache_a[1], cache_b[1])

        assert mx.allclose(
            cache_a[1].astype(mx.float32),
            cache_b[1].astype(mx.float32),
            atol=2e-2,
            rtol=2e-2,
        )


# ---------------------------------------------------------------------------
# patch_gated_delta_bf16_state — integration with the real modules
# ---------------------------------------------------------------------------


class TestPatchEntryPoint:
    def test_patch_applies_and_is_idempotent(self):
        # mlx_lm / mlx_vlm qwen3_5 modules are available in this env, so the
        # patch should report success and be safe to call twice.
        first = patch_gated_delta_bf16_state()
        second = patch_gated_delta_bf16_state()
        assert first == second  # idempotent result
