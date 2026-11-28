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


class TestSchedulerStatsContract:
    def _make_stats_scheduler(self, monkeypatch, metal_mode, calls):
        import vllm_mlx.scheduler as scheduler_module

        sched = object.__new__(scheduler_module.Scheduler)
        sched.waiting = [object(), object()]
        sched.running = {"running": object()}
        sched.num_requests_processed = 3
        sched.total_prompt_tokens = 11
        sched.total_completion_tokens = 7
        sched.batch_generator = types.SimpleNamespace()
        sched.block_aware_cache = None
        sched.memory_aware_cache = None
        sched.prefix_cache = None

        def available():
            calls.append("available")
            if metal_mode == "availability_error":
                raise RuntimeError("availability failed")
            return metal_mode != "unavailable"

        def active():
            calls.append("active")
            return 1_234_000_000

        def peak():
            calls.append("peak")
            if metal_mode == "peak_error":
                raise RuntimeError("peak failed")
            return 2_345_000_000

        def cache_memory():
            calls.append("cache_memory")
            return 3_456_000_000

        # Patch the imported module explicitly: its cached mx need not be the
        # autouse fixture's current sys.modules entry. Never call real Metal.
        monkeypatch.setattr(
            scheduler_module,
            "mx",
            types.SimpleNamespace(
                metal=types.SimpleNamespace(is_available=available),
                get_active_memory=active,
                get_peak_memory=peak,
                get_cache_memory=cache_memory,
            ),
        )
        return sched

    @pytest.mark.parametrize(
        "metal_mode", ["unavailable", "available", "availability_error", "peak_error"]
    )
    @pytest.mark.parametrize("cache_mode", ["none", "paged", "memory", "prefix", "all"])
    @pytest.mark.parametrize("mtp_mode", ["absent", "noncallable", "nested", "opaque"])
    def test_heterogeneous_snapshots_preserve_values_identity_and_order(
        self, monkeypatch, metal_mode, cache_mode, mtp_mode
    ):
        calls = []
        sched = self._make_stats_scheduler(monkeypatch, metal_mode, calls)
        cache_payloads = [
            {"hits": 3, "ratio": 0.25, "legacy": None},
            {"memory_mb": 1.5, "detail": {"unknown": [True, None]}},
            {"hit_rate": 0.5, "tokens_saved": 8},
        ]
        cache_fields = ("block_aware_cache", "memory_aware_cache", "prefix_cache")
        cache_keys = ("paged_cache", "memory_aware_cache", "prefix_cache")

        def provider(index):
            def get_stats():
                calls.append(cache_keys[index])
                return cache_payloads[index]

            return types.SimpleNamespace(get_stats=get_stats)

        for index, mode in enumerate(("paged", "memory", "prefix")):
            if cache_mode in (mode, "all"):
                setattr(sched, cache_fields[index], provider(index))
        providers = tuple(getattr(sched, field) for field in cache_fields)
        mtp_payload = {"enabled": True, "attempted": 2, "bypass_counts": {"x": 1}}
        opaque = object()

        def mtp_stats():
            calls.append("mtp")
            return mtp_payload if mtp_mode == "nested" else opaque

        if mtp_mode == "noncallable":
            sched.batch_generator.get_mtp_stats = "legacy"
        elif mtp_mode in ("nested", "opaque"):
            sched.batch_generator.get_mtp_stats = mtp_stats
        generator = sched.batch_generator
        expected = {
            "num_waiting": 2,
            "num_running": 1,
            "num_requests_processed": 3,
            "total_prompt_tokens": 11,
            "total_completion_tokens": 7,
        }
        expected_calls = []
        if mtp_mode in ("nested", "opaque"):
            expected["mtp"] = mtp_payload if mtp_mode == "nested" else opaque
            expected_calls.append("mtp")
        expected_calls.append("available")
        if metal_mode in ("available", "peak_error"):
            expected["metal_active_memory_gb"] = 1.23
            expected_calls.extend(["active", "peak"])
        if metal_mode == "available":
            expected["metal_peak_memory_gb"] = 2.35
            expected["metal_cache_memory_gb"] = 3.46
            expected_calls.append("cache_memory")
        selected = {"none": None, "paged": 0, "memory": 1, "prefix": 2, "all": 0}[
            cache_mode
        ]
        if selected is not None:
            expected[cache_keys[selected]] = cache_payloads[selected]
            expected_calls.append(cache_keys[selected])

        first = sched.get_stats()
        second = sched.get_stats()

        assert first is not second
        for result in (first, second):
            assert result == expected
            assert list(result) == list(expected)
            if selected is not None:
                assert result[cache_keys[selected]] is cache_payloads[selected]
            if mtp_mode in ("nested", "opaque"):
                assert result["mtp"] is expected["mtp"]
        assert calls == expected_calls * 2
        first["num_waiting"] = 999
        assert second["num_waiting"] == 2
        assert sched.batch_generator is generator
        assert all(
            getattr(sched, field) is original
            for field, original in zip(cache_fields, providers)
        )
        assert cache_payloads == [
            {"hits": 3, "ratio": 0.25, "legacy": None},
            {"memory_mb": 1.5, "detail": {"unknown": [True, None]}},
            {"hit_rate": 0.5, "tokens_saved": 8},
        ]
        assert mtp_payload == {
            "enabled": True,
            "attempted": 2,
            "bypass_counts": {"x": 1},
        }

    @pytest.mark.parametrize("source", ["mtp", "paged", "memory", "prefix"])
    def test_provider_errors_propagate_without_wrapping(self, monkeypatch, source):
        calls = []
        sched = self._make_stats_scheduler(monkeypatch, "unavailable", calls)
        error = ValueError("provider failure")

        def fail():
            calls.append(source)
            raise error

        provider = types.SimpleNamespace(get_stats=fail)
        if source == "mtp":
            sched.batch_generator.get_mtp_stats = fail
        else:
            field = {
                "paged": "block_aware_cache",
                "memory": "memory_aware_cache",
                "prefix": "prefix_cache",
            }[source]
            setattr(sched, field, provider)

        with pytest.raises(ValueError) as caught:
            sched.get_stats()

        assert caught.value is error
        assert calls == (["mtp"] if source == "mtp" else ["available", source])
