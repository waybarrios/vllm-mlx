# SPDX-License-Identifier: Apache-2.0
"""Fail-closed ``_prompt_cache_save`` for non-trimmable stacks (#678 GAP2D).

``_prompt_cache_save`` trims the prompt-only cache by 1 before storing.
If any layer vetoes trimming (e.g. recurrent state), storing the trimmed
(or untrimmed-but-misaligned) entry would corrupt reuse. The callback must
skip the store entirely (fail-closed).

Pure-logic (no MLX): fakes + stubbed ``mlx``/``mlx_lm`` modules, lazy
``Scheduler`` import, ``object.__new__`` instantiation.
"""

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_mlx_modules(monkeypatch):
    """Stub ``mlx``/``mlx_lm`` so ``vllm_mlx.scheduler`` imports without MLX."""
    mx = types.ModuleType("mlx.core")
    mx.zeros = lambda *a, **k: None  # noqa: E731
    mx.concatenate = lambda *a, **k: None  # noqa: E731
    mx.eval = lambda *a, **k: None  # noqa: E731
    mx.array = lambda *a, **k: None  # noqa: E731
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx

    cache_mod = types.ModuleType("mlx_lm.models.cache")

    class _DummyCache:
        pass

    cache_mod.KVCache = _DummyCache
    cache_mod.RotatingKVCache = _DummyCache
    cache_mod.ArraysCache = _DummyCache
    cache_mod.MambaCache = _DummyCache
    cache_mod.CacheList = _DummyCache

    models_mod = types.ModuleType("mlx_lm.models")
    models_mod.cache = cache_mod

    gen_mod = types.ModuleType("mlx_lm.generate")

    class BatchGenerator:
        pass

    gen_mod.BatchGenerator = BatchGenerator

    sample_mod = types.ModuleType("mlx_lm.sample_utils")
    sample_mod.make_logits_processors = lambda *a, **k: []  # noqa: E731
    sample_mod.make_sampler = lambda *a, **k: None  # noqa: E731

    tok_mod = types.ModuleType("mlx_lm.tokenizer_utils")

    class NaiveStreamingDetokenizer:
        pass

    tok_mod.NaiveStreamingDetokenizer = NaiveStreamingDetokenizer

    lm_mod = types.ModuleType("mlx_lm")
    lm_mod.models = models_mod
    lm_mod.generate = gen_mod
    lm_mod.sample_utils = sample_mod
    lm_mod.tokenizer_utils = tok_mod

    for name, mod in [
        ("mlx", mlx_pkg),
        ("mlx.core", mx),
        ("mlx_lm", lm_mod),
        ("mlx_lm.models", models_mod),
        ("mlx_lm.models.cache", cache_mod),
        ("mlx_lm.generate", gen_mod),
        ("mlx_lm.sample_utils", sample_mod),
        ("mlx_lm.tokenizer_utils", tok_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    # Attribute linkage for ``import mlx.core`` style imports.
    monkeypatch.setattr(mlx_pkg, "core", mx, raising=False)
    monkeypatch.setattr(lm_mod, "models", models_mod, raising=False)
    monkeypatch.setattr(lm_mod, "generate", gen_mod, raising=False)
    monkeypatch.setattr(models_mod, "cache", cache_mod, raising=False)


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


class _TrimmableLayer:
    """KV-like layer: offset + keys/values, no veto."""

    def __init__(self, length=10):
        self.keys = _FakeArray((1, 2, length, 4))
        self.values = _FakeArray((1, 2, length, 4))
        self.offset = length


class _NonTrimmableLayer:
    """KV-like layer that vetoes rewind via ``is_trimmable``."""

    def __init__(self, length=10):
        self.keys = _FakeArray((1, 2, length, 4))
        self.values = _FakeArray((1, 2, length, 4))
        self.offset = length

    def is_trimmable(self):
        return False


class _FakeRequest:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = list(prompt_token_ids)


class _FakeStore:
    """Spy for ``memory_aware_cache`` capturing ``store`` calls."""

    def __init__(self):
        self.called = False
        self.calls = []
        self.remove_calls = []

    def store(self, tokens, cache, evict_prefixes=False):
        self.called = True
        self.calls.append((list(tokens), cache, evict_prefixes))
        return True

    def remove(self, tokens):
        self.remove_calls.append(list(tokens))
        return True


def _make_scheduler(prompt_token_ids):
    """Lazy-import Scheduler and build a bare instance with a store spy."""
    from vllm_mlx.scheduler import Scheduler

    sched = object.__new__(Scheduler)
    store = _FakeStore()
    request_id = "req-test-001"
    sched.uid_to_request_id = {7: request_id}
    sched.requests = {request_id: _FakeRequest(prompt_token_ids)}
    sched.memory_aware_cache = store
    return sched, store


class TestPromptCacheSaveFailClosed:
    def test_mixed_stack_skips_store(self, _stub_mlx_modules):
        sched, store = _make_scheduler([1, 2, 3, 4])
        callback = sched._make_prompt_cache_save_callback()
        mixed = [_TrimmableLayer(length=10), _NonTrimmableLayer(length=10)]

        callback(7, mixed)

        assert store.called is False, "mixed stack must skip store (fail-closed)"
        assert store.calls == []

    def test_all_trimmable_stores(self, _stub_mlx_modules):
        sched, store = _make_scheduler([1, 2, 3, 4])
        callback = sched._make_prompt_cache_save_callback()
        layers = [_TrimmableLayer(length=10), _TrimmableLayer(length=10)]

        callback(7, layers)

        assert store.called is True, "all-trimmable stack must store"
        assert len(store.calls) == 1
        stored_tokens, _trimmed, evict = store.calls[0]
        assert stored_tokens == [1, 2, 3, 4]
        assert evict is False


class TestMidPrefillSaveFailClosed:
    def _make_mid_scheduler(self, prompt_token_ids):
        sched, store = _make_scheduler(prompt_token_ids)
        request = sched.requests["req-test-001"]
        request.cached_tokens = 0
        request.prefix_boundary = 0
        request._mid_prefill_last_save = 0
        request._mid_prefill_cache_key = None
        return sched, store, request

    def test_mixed_stack_skips_store(self, _stub_mlx_modules):
        sched, store, _req = self._make_mid_scheduler([1, 2, 3, 4, 5, 6, 7, 8])
        mixed = [_TrimmableLayer(length=10), _NonTrimmableLayer(length=10)]
        sched._extract_cache_states = lambda _pc: [{"dummy": 1}]  # noqa: E731
        sched._reconstruct_cache_from_states = lambda _ex: mixed  # noqa: E731
        callback = sched._make_mid_prefill_save_callback(save_interval=2)

        callback(7, 4, object())

        assert store.called is False, "mixed stack must skip store (fail-closed)"
        assert store.calls == []

    def test_all_trimmable_stores(self, _stub_mlx_modules):
        sched, store, _req = self._make_mid_scheduler([1, 2, 3, 4, 5, 6, 7, 8])
        layers = [_TrimmableLayer(length=10), _TrimmableLayer(length=10)]
        sched._extract_cache_states = lambda _pc: [{"dummy": 1}]  # noqa: E731
        sched._reconstruct_cache_from_states = lambda _ex: layers  # noqa: E731
        callback = sched._make_mid_prefill_save_callback(save_interval=2)

        callback(7, 4, object())

        assert store.called is True, "all-trimmable stack must store"
        assert len(store.calls) == 1
