# SPDX-License-Identifier: Apache-2.0
"""Fail-closed regression for ``_trim_cache_offset`` (#678).

Callers are guarded by #680/#683/#691, but ``_trim_cache_offset`` itself
must never trim a non-trimmable layer even when called without a guard.
Pure-logic (no MLX): fakes + stubbed ``mlx`` modules.
"""

import sys
import types

import pytest

from vllm_mlx.memory_cache import (
    _QuantizedCacheWrapper,
    _is_cache_layer_trimmable,
    _trim_cache_offset,
)


@pytest.fixture
def _fake_mlx_modules(monkeypatch):
    """Stub ``mlx.core`` + ``mlx_lm.models.cache`` so the trim helper imports."""
    mx = types.ModuleType("mlx.core")
    mx.zeros = lambda *a, **k: None  # noqa: E731
    mx.concatenate = lambda *a, **k: None  # noqa: E731
    mx.eval = lambda *a, **k: None  # noqa: E731
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    cache_mod = types.ModuleType("mlx_lm.models.cache")

    class _FakeRotating:
        pass

    cache_mod.RotatingKVCache = _FakeRotating
    models_mod = types.ModuleType("mlx_lm.models")
    models_mod.cache = cache_mod
    lm_mod = types.ModuleType("mlx_lm")
    lm_mod.models = models_mod
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return _FakeRotating


class _FakeArray:
    """Minimal array stand-in supporting shape + ``[..., :n, :]`` slicing."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.dtype = "float32"

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        for part in key:
            if isinstance(part, slice) and part.stop is not None:
                new_len = part.stop
                shape = list(self.shape)
                shape[-2] = max(new_len, 0)
                return _FakeArray(tuple(shape))
        return _FakeArray(self.shape)


class _PlainTrimmable:
    """KV-like layer: offset + keys/values, no veto."""

    def __init__(self, length=10):
        self.keys = _FakeArray((1, 2, length, 4))
        self.values = _FakeArray((1, 2, length, 4))
        self.offset = length


class _NonTrimmableByMethod:
    """KV-like layer that explicitly vetoes rewind via ``is_trimmable``."""

    def __init__(self, length=10):
        self.keys = _FakeArray((1, 2, length, 4))
        self.values = _FakeArray((1, 2, length, 4))
        self.offset = length

    def is_trimmable(self):
        return False


def _quantized_wrapper(offset=10, with_max_size=True):
    wrapper = _QuantizedCacheWrapper.__new__(_QuantizedCacheWrapper)
    wrapper.keys = object()
    wrapper.values = object()
    wrapper.offset = offset
    wrapper.bits = 8
    wrapper.group_size = 64
    wrapper.orig_type = object
    wrapper.orig_attrs = {"max_size": 8} if with_max_size else {}
    return wrapper


class TestTrimCacheOffsetFailClosed:
    def test_non_trimmable_method_veto_is_not_trimmed(self, _fake_mlx_modules):
        layer = _NonTrimmableByMethod(length=10)
        assert not _is_cache_layer_trimmable(layer)

        out = _trim_cache_offset([layer], 4)

        assert out[0] is layer
        assert layer.offset == 10
        assert layer.keys.shape[-2] == 10

    def test_quantized_wrapper_with_max_size_is_not_trimmed(self, _fake_mlx_modules):
        wrapper = _quantized_wrapper(offset=10, with_max_size=True)
        assert not _is_cache_layer_trimmable(wrapper)

        out = _trim_cache_offset([wrapper], 4)

        assert out[0] is wrapper
        assert wrapper.offset == 10

    def test_container_layer_is_not_trimmed(self, _fake_mlx_modules):
        class _Container:
            def __init__(self):
                self.caches = [_PlainTrimmable(length=6)]

        layer = _Container()
        assert not _is_cache_layer_trimmable(layer)

        out = _trim_cache_offset([layer], 2)

        assert out[0] is layer

    def test_trimmable_plain_layer_still_trims(self, _fake_mlx_modules):
        layer = _PlainTrimmable(length=10)
        assert _is_cache_layer_trimmable(layer)

        out = _trim_cache_offset([layer], 4)
        trimmed = out[0]

        assert trimmed is not layer
        assert trimmed.offset == 6
        assert trimmed.keys.shape[-2] == 6
        assert trimmed.values.shape[-2] == 6
        # Source entry untouched.
        assert layer.offset == 10
        assert layer.keys.shape[-2] == 10

    def test_quantized_wrapper_without_max_size_still_trims(self, _fake_mlx_modules):
        wrapper = _quantized_wrapper(offset=10, with_max_size=False)
        assert _is_cache_layer_trimmable(wrapper)

        out = _trim_cache_offset([wrapper], 4)

        assert out[0] is not wrapper
        assert out[0].offset == 6
        assert wrapper.offset == 10

    def test_mixed_stack_is_whole_stack_veto(self, _fake_mlx_modules):
        """Whole-stack (P2-1): one veto freezes the entire stack."""
        trimmable = _PlainTrimmable(length=10)
        frozen = _NonTrimmableByMethod(length=10)

        out = _trim_cache_offset([trimmable, frozen], 4)

        assert out[0] is trimmable
        assert out[1] is frozen
        assert trimmable.offset == 10
        assert trimmable.keys.shape[-2] == 10
        assert frozen.offset == 10
        assert frozen.keys.shape[-2] == 10

    def test_mixed_stack_reversed_is_whole_stack_veto(self, _fake_mlx_modules):
        frozen = _NonTrimmableByMethod(length=10)
        trimmable = _PlainTrimmable(length=10)

        out = _trim_cache_offset([frozen, trimmable], 4)

        assert out[0] is frozen
        assert out[1] is trimmable
        assert trimmable.offset == 10
        assert frozen.offset == 10

    def test_empty_cache_returns_empty(self, _fake_mlx_modules):
        assert _trim_cache_offset([], 4) == []

    def test_trim_zero_and_negative_are_noop(self, _fake_mlx_modules):
        for trim_by in (0, -1, -5):
            layer = _PlainTrimmable(length=10)
            out = _trim_cache_offset([layer], trim_by)
            assert out[0] is layer
            assert layer.offset == 10
            assert layer.keys.shape[-2] == 10
