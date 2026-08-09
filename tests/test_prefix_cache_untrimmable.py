# SPDX-License-Identifier: Apache-2.0
"""Prefix cache behaviour for caches that cannot be trimmed.

A ``prompt + output`` cache key is only reusable via a trim: every later query
is shorter than such a key, so the generated tail has to come off first. Models
with sliding-window or pooled KV cannot do that, which makes those entries dead
weight — and expensive weight, since each one holds a full-length KV copy and
Metal runs out of *buffers* long before the cache's byte budget is reached.

These tests pin the two halves of the fix: skipping the unusable entry, and
storing a post-prefill snapshot that is reusable by strict prefix match.
"""

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.scheduler import Scheduler  # noqa: E402


def _rotating_past_its_window() -> list:
    """A RotatingKVCache that has wrapped, so trimming is impossible."""
    c = cache_mod.RotatingKVCache(max_size=8)
    # 12 > max_size: the ring buffer has physically overwritten older KV.
    c.update_and_fetch(mx.zeros((1, 1, 12, 4)), mx.zeros((1, 1, 12, 4)))
    return [c]


def _plain_kv_cache() -> list:
    c = cache_mod.KVCache()
    c.update_and_fetch(mx.zeros((1, 1, 4, 4)), mx.zeros((1, 1, 4, 4)))
    return [c]


class TestPromptOutputEntryIsUseless:
    def test_wrapped_rotating_cache_cannot_back_a_prompt_output_entry(self):
        cache = _rotating_past_its_window()
        assert not cache_mod.can_trim_prompt_cache(cache), "setup: must be untrimmable"
        assert Scheduler._prompt_output_entry_is_useless(cache) is True

    def test_plain_kv_cache_still_gets_its_entry(self):
        cache = _plain_kv_cache()
        assert cache_mod.can_trim_prompt_cache(cache), "setup: must be trimmable"
        assert Scheduler._prompt_output_entry_is_useless(cache) is False

    def test_cache_list_is_useless_when_any_member_is(self):
        """CacheList reports one verdict for the whole group."""
        grouped = cache_mod.CacheList(*_plain_kv_cache(), *_rotating_past_its_window())
        assert Scheduler._prompt_output_entry_is_useless([grouped]) is True

    def test_unknown_cache_objects_do_not_break_the_scheduler(self):
        """Anything that cannot answer the question keeps its entry."""

        class Opaque:
            pass

        assert Scheduler._prompt_output_entry_is_useless([Opaque()]) is False


class TestSnapshotOwnsItsMemory:
    """A snapshot that aliases the live cache is rewritten by the generation
    it is supposed to predate: RotatingKVCache writes into its ring buffer and
    PoolingCache into its remainder buffer, both in place.

    Ownership of copying lives at the storage boundary: ``store()`` detaches
    every array into a freshly allocated, evaluated copy, so the scheduler
    hands over a structure mirror whose state deliberately aliases the live
    caches.  These tests pin the guarantee where it is now enforced.
    (Container-shape coverage — nested lists/tuples/mappings, pass-through of
    non-array metadata — lives in tests/test_memory_cache_mlx.py.)
    """

    @staticmethod
    def _store(cache_layers, tokens=(1, 2, 3)):
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        class _Model:
            pass

        mc = MemoryAwarePrefixCache(
            _Model(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
        )
        assert mc.store(list(tokens), cache_layers)
        return mc

    def test_stored_snapshot_is_not_a_view_of_the_source(self):
        live = cache_mod.RotatingKVCache(max_size=8)
        live.update_and_fetch(mx.ones((1, 1, 4, 4)), mx.ones((1, 1, 4, 4)))

        mc = self._store([live])
        got, remaining = mc.fetch([1, 2, 3])

        assert remaining == []
        assert got[0] is not live
        assert got[0].keys is not live.keys
        assert mx.array_equal(got[0].keys, live.keys)

    def test_stored_snapshot_survives_in_place_mutation_of_the_source(self):
        live = cache_mod.RotatingKVCache(max_size=8)
        live.update_and_fetch(mx.zeros((1, 1, 4, 4)), mx.zeros((1, 1, 4, 4)))

        mc = self._store([live])
        got, _ = mc.fetch([1, 2, 3])
        before = mx.sum(got[0].keys).item()

        # Keep generating: the ring buffer is written in place past max_size.
        for _ in range(6):
            live.update_and_fetch(mx.ones((1, 1, 4, 4)) * 5, mx.ones((1, 1, 4, 4)) * 5)

        got, _ = mc.fetch([1, 2, 3])
        after = mx.sum(got[0].keys).item()
        assert before == after, "continued generation mutated the stored snapshot"

    def test_scheduler_mirror_flow_stores_detached_state(self):
        """The rewired snapshot path end-to-end THROUGH the real scheduler
        method: _store_prompt_only_cache mirrors structure, aliases state,
        and relies on store() as the sole copy/eval owner."""
        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        # Untrimmable: wrapped past max_size, 12 tokens total.
        live = cache_mod.RotatingKVCache(max_size=8)
        live.update_and_fetch(mx.ones((1, 1, 12, 4)), mx.ones((1, 1, 12, 4)))

        class _Model:
            pass

        sched = _bare_scheduler()
        sched.memory_aware_cache = MemoryAwarePrefixCache(
            _Model(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
        )
        request = _FakeRequest(list(range(1, 12)))  # 11 prompt tokens
        response = _FakeResponse(token=77, uid=0)
        response.prompt_cache = [live]

        Scheduler._store_prompt_only_cache(sched, request, response)

        key = list(range(1, 12)) + [77]  # prompt + first generated token
        got, remaining = sched.memory_aware_cache.fetch(key)
        assert got is not None and remaining == [], "snapshot was not stored"
        before = mx.sum(got[0].keys).item()

        # Ring buffer writes in place once wrapped: continued generation on
        # the live cache must not reach the stored snapshot.
        for _ in range(6):
            live.update_and_fetch(mx.ones((1, 1, 1, 4)) * 5, mx.ones((1, 1, 1, 4)) * 5)
        mx.eval(live.keys, live.values)

        got, _ = sched.memory_aware_cache.fetch(key)
        after = mx.sum(got[0].keys).item()
        assert before == after, "aliasing mirror leaked live mutation into the store"


class _FakeRequest:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = list(prompt_token_ids)
        self.request_id = "req-0123456789"
        self.cached_tokens = 0
        self.num_output_tokens = 0


class _FakeResponse:
    def __init__(self, token=None, uid=0):
        self.token = token
        self.uid = uid


def _bare_scheduler():
    sched = object.__new__(Scheduler)
    return sched


class TestSnapshotKeyAlignment:
    """The key must name exactly the tokens the snapshot holds.

    The snapshot is taken while processing the response carrying the first
    generated token, and the batch has already fed that token through the
    cache: measured ``prompt_len=5, cache_offset=6`` on a real scheduler run.
    Storing that under ``prompt_token_ids`` leaves warm reuse one token ahead
    of its key, and trimming the overshoot off is unavailable here — these are
    the caches that cannot be trimmed.
    """

    def test_key_is_extended_to_cover_the_generated_token(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5])
        cache = _plain_kv_cache()
        cache[0].offset = 6  # prompt + first generated token

        key = Scheduler._cache_key_for_snapshot(
            sched, request, _FakeResponse(token=99), cache
        )

        assert key == [1, 2, 3, 4, 5, 99]
        assert len(key) == Scheduler._cache_coverage(
            cache
        ), "key length must equal what the cache holds"

    def test_key_is_the_prompt_when_the_cache_stops_there(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5])
        cache = _plain_kv_cache()
        cache[0].offset = 5

        key = Scheduler._cache_key_for_snapshot(
            sched, request, _FakeResponse(token=99), cache
        )
        assert key == [1, 2, 3, 4, 5]

    def test_short_cache_is_refused_rather_than_stored_misaligned(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5])
        cache = _plain_kv_cache()
        cache[0].offset = 3

        assert (
            Scheduler._cache_key_for_snapshot(
                sched, request, _FakeResponse(token=99), cache
            )
            is None
        )

    def test_unnamed_overshoot_is_refused(self):
        """Two tokens past the prompt but only one is known — do not guess."""
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5])
        cache = _plain_kv_cache()
        cache[0].offset = 7

        assert (
            Scheduler._cache_key_for_snapshot(
                sched, request, _FakeResponse(token=99), cache
            )
            is None
        )


class TestSnapshotDestinationTopology:
    """A destination that cannot hold the live state means no entry at all.

    ``make_prompt_cache(model)`` yields plain ``KVCache`` layers; assigning a
    ``RotatingKVCache``'s state or meta_state to one raises, the store path's
    broad handler logs a warning, and the snapshot is silently never stored —
    on exactly the sliding-window configurations this feature exists for.
    """

    def test_rotating_layers_are_mirrored_not_replaced(self):
        sched = _bare_scheduler()
        live = _rotating_past_its_window()

        dest = Scheduler._make_snapshot_destination(sched, live)

        assert dest is not None
        assert type(dest[0]) is type(live[0])
        assert dest[0].max_size == live[0].max_size
        assert dest[0] is not live[0], "destination must not alias the live layer"

    def test_mixed_topology_is_preserved_layer_by_layer(self):
        sched = _bare_scheduler()
        live = _plain_kv_cache() + _rotating_past_its_window()

        dest = Scheduler._make_snapshot_destination(sched, live)

        assert [type(d).__name__ for d in dest] == [
            type(layer).__name__ for layer in live
        ]

    def test_state_assignment_round_trips_on_a_rotating_layer(self):
        """The assignment that used to raise and get swallowed.

        The mirror's state now aliases the live layer on purpose — store()
        owns the copy — so this pins only the structural round-trip.
        """
        sched = _bare_scheduler()
        live = _rotating_past_its_window()
        dest = Scheduler._make_snapshot_destination(sched, live)

        meta = getattr(live[0], "meta_state", None)
        if meta is not None:
            dest[0].meta_state = meta
        dest[0].state = live[0].state

        assert dest[0].offset == live[0].offset


class TestNestedCacheListInvariants:
    """DeepSeek-V4 groups several caches per layer in a ``CacheList``.

    A container carries no ``offset`` and copies shallowly, so both invariants
    this file exists for break at the outer layer while every flat-layer test
    still passes: coverage reads None and the key silently falls back to
    prompt-only, and the "snapshot" shares the live child objects and follows
    the generation it is supposed to predate.
    """

    @staticmethod
    def _nested(tokens=7, max_size=8):
        plain = cache_mod.KVCache()
        plain.update_and_fetch(mx.ones((1, 1, tokens, 4)), mx.ones((1, 1, tokens, 4)))
        rotating = cache_mod.RotatingKVCache(max_size=max_size)
        rotating.update_and_fetch(
            mx.ones((1, 1, tokens, 4)), mx.ones((1, 1, tokens, 4))
        )
        return cache_mod.CacheList(plain, rotating)

    def test_coverage_descends_into_the_container(self):
        live = [self._nested(tokens=7)]

        assert Scheduler._cache_coverage(live) == 7, (
            "a CacheList has no offset of its own; reading the attribute off "
            "the layer returns None and the key falls back to prompt-only"
        )

    def test_key_length_matches_nested_coverage(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5, 6])
        live = [self._nested(tokens=7)]

        key = Scheduler._cache_key_for_snapshot(
            sched, request, _FakeResponse(token=77), live
        )

        assert key == [1, 2, 3, 4, 5, 6, 77]
        assert len(key) == Scheduler._cache_coverage(live)

    def test_snapshot_does_not_share_children_with_the_live_cache(self):
        sched = _bare_scheduler()
        live = [self._nested()]

        dest = Scheduler._make_snapshot_destination(sched, live)

        assert type(dest[0]) is type(live[0])
        assert [type(c).__name__ for c in dest[0].caches] == [
            type(c).__name__ for c in live[0].caches
        ]
        for mirrored, original in zip(dest[0].caches, live[0].caches):
            assert mirrored is not original, "container copied shallowly"

    def test_snapshot_survives_continued_generation_on_the_live_cache(self):
        """The invariant that matters: the snapshot must predate what follows.

        The mirror's state aliases the live children by design; store() is the
        sole copy/eval owner, so the invariant is asserted on what fetch()
        returns after storing the aliasing mirror.
        """
        sched = _bare_scheduler()
        live = [self._nested(tokens=4, max_size=8)]
        dest = Scheduler._make_snapshot_destination(sched, live)

        for mirrored, original in zip(dest[0].caches, live[0].caches):
            meta = getattr(original, "meta_state", None)
            if meta is not None:
                mirrored.meta_state = meta
            mirrored.state = original.state

        mc = TestSnapshotOwnsItsMemory._store(dest)

        def _sums(entry):
            return [
                mx.sum(a).item()
                for child in entry[0].caches
                for a in child.state
                if isinstance(a, mx.array)
            ]

        got, _ = mc.fetch([1, 2, 3])
        before = _sums(got)
        for child in live[0].caches:
            for _ in range(6):
                child.update_and_fetch(
                    mx.ones((1, 1, 1, 4)) * 5, mx.ones((1, 1, 1, 4)) * 5
                )
        got, _ = mc.fetch([1, 2, 3])
        after = _sums(got)

        assert before == after, "live generation mutated the stored snapshot"


class TestSnapshotIsGatedToNonTrimmableTopologies:
    """The snapshot exists for caches whose completion-time entry is unusable.

    Running it for an ordinary ``KVCache`` is not merely wasted work. The
    completion path already stores a correct ``N``-token entry; an ``N+1``
    snapshot stored with ``evict_prefixes=True`` replaces it, and an identical
    ``N``-token prompt then matches a *supersequence* rather than an exact key.
    The scheduler only refuses ``exact`` hits, so it replays ``prompt[-1]`` on
    top of a cache that already holds it — the duplicated-token bug this file
    was written to avoid, reintroduced from the other side.
    """

    @staticmethod
    def _filled(layer, tokens=6):
        layer.update_and_fetch(mx.zeros((1, 1, tokens, 4)), mx.zeros((1, 1, tokens, 4)))
        return layer

    def test_trimmable_cache_is_left_to_the_completion_path(self):
        cache = [self._filled(cache_mod.KVCache())]
        assert cache_mod.can_trim_prompt_cache(cache) is True
        assert Scheduler._prompt_output_entry_is_useless(cache) is False, (
            "a trimmable cache must not be snapshotted here; its completion "
            "entry is reusable and this one would evict it"
        )

    def test_non_trimmable_cache_still_qualifies(self):
        cache = _rotating_past_its_window()
        assert cache_mod.can_trim_prompt_cache(cache) is False
        assert Scheduler._prompt_output_entry_is_useless(cache) is True

    def test_store_returns_early_for_a_trimmable_cache(self):
        """Drives the real store path rather than the predicate alone."""
        sched = _bare_scheduler()
        stored = []
        sched.memory_aware_cache = type(
            "C",
            (),
            {"store": lambda self, *a, **k: stored.append(a) or True, "_entries": {}},
        )()
        sched.model = object()
        sched._extract_cache_for_uid = lambda uid: [self._filled(cache_mod.KVCache())]

        request = _FakeRequest([1, 2, 3, 4, 5])
        Scheduler._store_prompt_only_cache(
            sched, request, _FakeResponse(token=99, uid=0)
        )

        assert stored == [], "a trimmable cache reached the snapshot store"


class TestUnknownCoverageFailsClosed:
    """A cache with no offset must not be stored under a guessed key.

    ``ArraysCache`` exposes no ``offset``, but by the time the first response
    arrives mlx-lm has already folded the first generated token into the
    recurrent state. Storing that under the prompt alone makes the next turn
    replay the token into cumulative state — and for these models the snapshot
    is the only entry that ever gets written, so nothing else corrects it.
    """

    def test_arrays_cache_has_no_offset_to_read(self):
        cache = [cache_mod.ArraysCache(2)]
        assert Scheduler._cache_coverage(cache) is None

    def test_key_is_refused_rather_than_guessed(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3, 4, 5])

        key = Scheduler._cache_key_for_snapshot(
            sched, request, _FakeResponse(token=99), [cache_mod.ArraysCache(2)]
        )

        assert key is None, (
            "unknown coverage fell back to prompt_ids; the stored state covers "
            "prompt + first generated token and the key would be one short"
        )

    def test_nested_container_of_arrays_caches_is_also_refused(self):
        sched = _bare_scheduler()
        request = _FakeRequest([1, 2, 3])
        nested = cache_mod.CacheList(cache_mod.ArraysCache(2), cache_mod.ArraysCache(2))

        assert (
            Scheduler._cache_key_for_snapshot(
                sched, request, _FakeResponse(token=7), [nested]
            )
            is None
        )
