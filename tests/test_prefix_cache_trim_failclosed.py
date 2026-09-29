# SPDX-License-Identifier: Apache-2.0
"""Fail-closed regression for prefix trim with mixed layers (#678).

``PrefixCacheManager._can_trim_cache`` only inspected the first layer,
so ``[trimmable, RotatingKVCache]`` was reported trimmable and
``_trim_cache`` trimmed the non-trimmable layer. Mixed stacks must not
trim at all (fail-closed). Pure-logic (no MLX): fakes only.
"""

from vllm_mlx.prefix_cache import PrefixCacheManager


class _TrimmableCache:
    """Trimmable layer: supports trim + explicit opt-in."""

    def __init__(self, length=10):
        self.length = length
        self.trim_calls = 0

    def is_trimmable(self):
        return True

    def trim(self, num_tokens):
        self.trim_calls += 1
        self.length -= num_tokens


class _NonTrimmableCache:
    """RotatingKVCache-like layer: vetoes trim but exposes trim attr."""

    def __init__(self, length=10):
        self.length = length
        self.trim_calls = 0

    def is_trimmable(self):
        return False

    def trim(self, num_tokens):
        self.trim_calls += 1
        self.length -= num_tokens


class _NoTrimCache:
    """Layer without trim support (no trim method)."""

    def __init__(self, length=10):
        self.length = length


class _RaisingTrimmableCache:
    """Layer whose is_trimmable raises (fail-closed must return False)."""

    def __init__(self, length=10):
        self.length = length
        self.trim_calls = 0

    def is_trimmable(self):
        raise RuntimeError("boom")

    def trim(self, num_tokens):
        self.trim_calls += 1
        self.length -= num_tokens


class _CompatTrimCache:
    """Trim-capable layer without is_trimmable (compat contract -> True)."""

    def __init__(self, length=10):
        self.length = length
        self.trim_calls = 0

    def trim(self, num_tokens):
        self.trim_calls += 1
        self.length -= num_tokens


class _BoolNonTrimmableCache:
    """Layer with bool is_trimmable flag (non-callable) set to False."""

    is_trimmable = False

    def __init__(self, length=10):
        self.length = length
        self.trim_calls = 0

    def trim(self, num_tokens):
        self.trim_calls += 1
        self.length -= num_tokens


def _manager():
    return PrefixCacheManager(model=object())


class TestPrefixCacheTrimFailClosed:
    def test_mixed_trimmable_first_is_not_trimmable(self):
        mgr = _manager()
        layers = [_TrimmableCache(), _NonTrimmableCache()]

        assert mgr._can_trim_cache(layers) is False

    def test_mixed_non_trimmable_first_is_not_trimmable(self):
        mgr = _manager()
        layers = [_NonTrimmableCache(), _TrimmableCache()]

        assert mgr._can_trim_cache(layers) is False

    def test_mixed_stack_trim_is_noop(self):
        mgr = _manager()
        trimmable = _TrimmableCache(length=10)
        frozen = _NonTrimmableCache(length=10)

        out = mgr._trim_cache([trimmable, frozen], 4)

        assert out[0] is trimmable
        assert out[1] is frozen
        assert trimmable.trim_calls == 0
        assert frozen.trim_calls == 0
        assert trimmable.length == 10
        assert frozen.length == 10

    def test_trimmable_stack_still_trims(self):
        mgr = _manager()
        layers = [_TrimmableCache(length=10), _TrimmableCache(length=10)]

        assert mgr._can_trim_cache(layers) is True

        out = mgr._trim_cache(layers, 4)

        assert [c.length for c in out] == [6, 6]
        assert [c.trim_calls for c in out] == [1, 1]

    def test_empty_cache_is_not_trimmable_and_trim_noop(self):
        mgr = _manager()

        assert mgr._can_trim_cache([]) is False
        assert mgr._trim_cache([], 4) == []

    def test_trim_zero_and_negative_are_noop(self):
        mgr = _manager()
        for num_tokens in (0, -1, -5):
            layers = [_TrimmableCache(length=10), _TrimmableCache(length=10)]
            out = mgr._trim_cache(layers, num_tokens)

            assert out[0] is layers[0]
            assert out[1] is layers[1]
            assert [c.length for c in out] == [10, 10]
            assert [c.trim_calls for c in out] == [0, 0]

    def test_layer_without_trim_is_not_trimmable(self):
        mgr = _manager()
        layers = [_NoTrimCache(length=10)]

        assert mgr._can_trim_cache(layers) is False

        out = mgr._trim_cache(layers, 4)

        assert out[0] is layers[0]
        assert out[0].length == 10

    def test_is_trimmable_raising_is_fail_closed(self):
        mgr = _manager()
        layers = [_RaisingTrimmableCache(length=10)]

        assert mgr._can_trim_cache(layers) is False

        out = mgr._trim_cache(layers, 4)

        assert out[0] is layers[0]
        assert out[0].trim_calls == 0
        assert out[0].length == 10

    def test_layer_with_trim_without_is_trimmable_is_compat_trimmable(self):
        mgr = _manager()
        layers = [_CompatTrimCache(length=10), _CompatTrimCache(length=10)]

        assert mgr._can_trim_cache(layers) is True

        out = mgr._trim_cache(layers, 4)

        assert [c.length for c in out] == [6, 6]
        assert [c.trim_calls for c in out] == [1, 1]

    def test_bool_is_trimmable_false_is_fail_closed(self):
        mgr = _manager()
        layers = [_BoolNonTrimmableCache(length=10)]

        assert mgr._can_trim_cache(layers) is False

        out = mgr._trim_cache(layers, 4)

        assert out[0] is layers[0]
        assert out[0].trim_calls == 0
        assert out[0].length == 10
