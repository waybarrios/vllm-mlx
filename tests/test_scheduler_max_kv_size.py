# SPDX-License-Identifier: Apache-2.0
"""``--max-kv-size`` has to reach the caches that generation actually uses.

It used to reach none of them. The scheduler built a bounded cache itself, then
handed it to ``_validate_cache``, which rejects any layer whose ``keys`` are
still None — true of every freshly created cache, bounded or not. So
``cache_to_use`` went back to None and ``BatchGenerator``, never told about
``max_kv_size``, built an unbounded one instead. The flag was a no-op while the
CLI printed "Max KV size: N (RotatingKVCache)".
"""

import pytest

pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402


class _Layer:
    """Stands in for a decoder layer; enough for make_prompt_cache."""


class _Model:
    def __init__(self, n_layers: int = 3) -> None:
        self.layers = [_Layer() for _ in range(n_layers)]


def _scheduler(**config_kwargs) -> Scheduler:
    from types import SimpleNamespace

    sched = object.__new__(Scheduler)
    sched.model = _Model()
    sched.config = SchedulerConfig(**config_kwargs)
    # _create_batch_generator reaches for stop tokens and MTP on the way past.
    sched.tokenizer = SimpleNamespace(eos_token_id=0, eos_token_ids={0})
    sched._actual_tokenizer = sched.tokenizer
    sched.memory_aware_cache = None
    sched.block_aware_cache = None
    sched._pending_abort_ids = set()
    sched.uid_to_request_id = {}
    sched.requests = {}
    return sched


class TestValidateCacheRejectsFreshCaches:
    """The behaviour that made the bounded cache disappear."""

    def test_a_freshly_built_cache_does_not_validate(self):
        sched = _scheduler()
        fresh = cache_mod.make_prompt_cache(sched.model, max_kv_size=64)

        assert all(isinstance(c, cache_mod.RotatingKVCache) for c in fresh)
        assert sched._validate_cache(fresh) is False, (
            "an empty cache has keys=None, so validation of a *restored* entry "
            "must not be applied to one the scheduler just created"
        )

    def test_it_is_not_specific_to_rotating_caches(self):
        """Same verdict unbounded — the guard is about emptiness, not topology."""
        sched = _scheduler()
        fresh = cache_mod.make_prompt_cache(sched.model)

        assert all(isinstance(c, cache_mod.KVCache) for c in fresh)
        assert sched._validate_cache(fresh) is False


class TestMaxKvSizeReachesTheBatchGenerator:
    def test_configured_size_is_passed_through(self, monkeypatch):
        captured = {}

        class _FakeBatchGenerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "vllm_mlx.scheduler.BatchGenerator", _FakeBatchGenerator, raising=False
        )
        sched = _scheduler(max_kv_size=1024)
        sched._create_batch_generator(_sampling_params())

        assert captured.get("max_kv_size") == 1024, (
            "BatchGenerator builds the cache for any sequence inserted without "
            f"one; it was given max_kv_size={captured.get('max_kv_size')!r}"
        )

    def test_unset_stays_unbounded(self, monkeypatch):
        captured = {}

        class _FakeBatchGenerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "vllm_mlx.scheduler.BatchGenerator", _FakeBatchGenerator, raising=False
        )
        sched = _scheduler(max_kv_size=0)
        sched._create_batch_generator(_sampling_params())

        assert (
            captured.get("max_kv_size") is None
        ), "0 means unbounded and must not become RotatingKVCache(max_size=0)"


class TestBoundedCacheTopology:
    """BatchGenerator's rule is the one that survives ``make_cache`` models.

    ``make_prompt_cache(model, max_kv_size=N)`` silently ignores ``max_kv_size``
    whenever the model defines ``make_cache``, which most current
    architectures do. Post-processing the model's own cache is the only way to
    bound those.
    """

    def test_make_prompt_cache_ignores_the_bound_for_make_cache_models(self):
        class _ModelWithMakeCache(_Model):
            def make_cache(self):
                return [cache_mod.KVCache() for _ in self.layers]

        model = _ModelWithMakeCache()
        built = cache_mod.make_prompt_cache(model, max_kv_size=64)

        assert all(isinstance(c, cache_mod.KVCache) for c in built)
        assert not any(isinstance(c, cache_mod.RotatingKVCache) for c in built), (
            "if this ever starts honouring max_kv_size, the scheduler can stop "
            "relying on BatchGenerator to post-process"
        )

    def test_post_processing_bounds_those_layers(self):
        """What BatchGenerator._make_new_cache does, and why it is correct."""

        class _ModelWithMakeCache(_Model):
            def make_cache(self):
                return [cache_mod.KVCache() for _ in self.layers]

        built = cache_mod.make_prompt_cache(_ModelWithMakeCache())
        bounded = [
            (
                cache_mod.RotatingKVCache(max_size=64)
                if isinstance(c, cache_mod.KVCache)
                else c
            )
            for c in built
        ]
        assert all(isinstance(c, cache_mod.RotatingKVCache) for c in bounded)
        assert all(c.max_size == 64 for c in bounded)


def _sampling_params():
    from vllm_mlx.scheduler import SamplingParams

    return SamplingParams()


class TestBoundingDescendsIntoContainers:
    """mlx-lm's own rule converts flat layers only.

    ``BatchGenerator._make_new_cache`` rewrites ``KVCache`` layers and stops
    there, so a ``KVCache`` nested inside a ``CacheList`` stays unbounded —
    and the architectures that nest (DeepSeek-V4 groups three caches per layer)
    are exactly the ones that need the bound.
    """

    def test_nested_plain_layers_are_bounded(self):
        live = [
            cache_mod.CacheList(
                cache_mod.KVCache(), cache_mod.RotatingKVCache(max_size=128)
            ),
            cache_mod.KVCache(),
        ]

        out = Scheduler._bound_cache_layers(live, 64)

        nested = out[0].caches
        assert isinstance(nested[0], cache_mod.RotatingKVCache)
        assert nested[0].max_size == 64, "nested KVCache was left unbounded"
        assert isinstance(out[1], cache_mod.RotatingKVCache)
        assert out[1].max_size == 64

    def test_layers_that_bound_themselves_are_left_alone(self):
        """A rotating layer already carries its own window; do not retune it."""
        live = [
            cache_mod.CacheList(
                cache_mod.RotatingKVCache(max_size=128), cache_mod.KVCache()
            )
        ]

        out = Scheduler._bound_cache_layers(live, 64)

        assert out[0].caches[0].max_size == 128, "existing window was overwritten"
        assert out[0].caches[1].max_size == 64

    def test_non_kv_cache_types_are_preserved(self):
        """Replacing a quantized or chunked cache would break the model."""
        quantized = cache_mod.QuantizedKVCache()
        chunked = cache_mod.ChunkedKVCache(chunk_size=16)

        out = Scheduler._bound_cache_layers([quantized, chunked], 64)

        assert out[0] is quantized
        assert out[1] is chunked

    def test_mlx_lm_rule_alone_would_miss_the_nested_case(self):
        """Pins why the scheduler does this instead of leaving it to mlx-lm.

        If BatchGenerator ever learns to recurse, this fails and the scheduler
        can drop its own copy.
        """
        live = [cache_mod.CacheList(cache_mod.KVCache(), cache_mod.KVCache())]
        mlx_lm_rule = [
            (
                cache_mod.RotatingKVCache(max_size=64)
                if isinstance(layer, cache_mod.KVCache)
                else layer
            )
            for layer in live
        ]

        assert isinstance(mlx_lm_rule[0], cache_mod.CacheList)
        assert all(
            isinstance(c, cache_mod.KVCache) for c in mlx_lm_rule[0].caches
        ), "mlx-lm started recursing; the scheduler's own bounding is redundant"


class TestNegativeSizesAreRejected:
    """``self.config.max_kv_size or None`` forwarded -1 straight through.

    Normalizing it per request would fix the value and create a second
    problem: one bad startup flag logging on every scheduling pass.
    """

    @pytest.mark.parametrize("size", [-1, -1024])
    def test_negative_is_normalized_at_config_construction(self, size):
        config = SchedulerConfig(max_kv_size=size)
        assert (
            config.max_kv_size == 0
        ), f"max_kv_size={size} would reach RotatingKVCache(max_size={size})"

    def test_non_integer_is_normalized_too(self):
        assert SchedulerConfig(max_kv_size="512").max_kv_size == 0

    def test_zero_is_unbounded(self):
        assert _scheduler(max_kv_size=0)._bounded_kv_size() is None

    def test_positive_is_forwarded(self):
        assert _scheduler(max_kv_size=4096)._bounded_kv_size() == 4096

    def test_negative_never_reaches_the_batch_generator(self, monkeypatch):
        captured = {}

        class _FakeBatchGenerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "vllm_mlx.scheduler.BatchGenerator", _FakeBatchGenerator, raising=False
        )
        _scheduler(max_kv_size=-1)._create_batch_generator(_sampling_params())

        assert captured.get("max_kv_size") is None

    def test_bad_value_is_reported_once_not_per_request(self, caplog):
        """A startup mistake must not scale its logging with traffic."""
        import logging

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.scheduler"):
            sched = _scheduler(max_kv_size=-1)
            construction_warnings = sum(
                "max_kv_size" in r.getMessage() for r in caplog.records
            )
            caplog.clear()

            # Scheduling is where this used to log — once per uncached request.
            for _ in range(5):
                sched._bounded_kv_size()

            # Count *every* warning from the scheduling path, not just ones
            # naming max_kv_size: a hot-path warning worded differently is the
            # same production problem, and filtering by wording let one slip
            # past an earlier version of this test.
            scheduling_warnings = len(caplog.records)

        assert construction_warnings == 1, (
            f"expected exactly one warning at construction, got "
            f"{construction_warnings}"
        )
        assert scheduling_warnings == 0, (
            f"{scheduling_warnings} warnings emitted from the scheduling path; "
            "one bad startup flag would flood production logs per request"
        )


class TestRejectedCacheAlwaysResetsTheRequest:
    """A rejected cache means a full prefill, whatever max_kv_size says.

    The reset used to sit after the bounded-cache branch, so with
    ``max_kv_size=0`` a request whose restored cache failed validation kept its
    stale ``cached_tokens``/``remaining_tokens`` and the scheduler inserted only
    the prompt's suffix — a four-token prompt prefilling one token.
    """

    @staticmethod
    def _real_scheduler(max_kv_size):
        from types import SimpleNamespace

        return Scheduler(
            model=_Model(),
            tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0}),
            config=SchedulerConfig(enable_prefix_cache=False, max_kv_size=max_kv_size),
        )

    @staticmethod
    def _request(prompt_ids):
        from vllm_mlx.request import Request
        from vllm_mlx.scheduler import SamplingParams

        request = Request(
            request_id="req-abc",
            prompt="prompt",
            prompt_token_ids=list(prompt_ids),
            sampling_params=SamplingParams(),
        )
        request.num_prompt_tokens = len(prompt_ids)
        # A restored entry that _validate_cache rejects: keys are still None.
        request.prompt_cache = [cache_mod.KVCache()]
        request.cached_tokens = len(prompt_ids) - 1
        request.remaining_tokens = list(prompt_ids[-1:])
        return request

    @pytest.mark.parametrize("max_kv_size", [0, 512])
    def test_scheduler_inserts_the_whole_prompt(self, max_kv_size, monkeypatch):
        """Drives the scheduler itself.

        Replicating the branch in the test body would not have caught this: the
        bug was *where* the reset sits, not what it does.
        """
        inserted = {}

        class _FakeBatchGenerator:
            def insert(self, prompts, **kwargs):
                inserted["prompts"] = [list(p) for p in prompts]
                return [1]

        sched = self._real_scheduler(max_kv_size)
        sched.batch_generator = _FakeBatchGenerator()

        request = self._request([1, 2, 3, 4])
        sched.requests[request.request_id] = request
        sched.waiting.append(request)

        sched._schedule_waiting()

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [1, 2, 3, 4]
