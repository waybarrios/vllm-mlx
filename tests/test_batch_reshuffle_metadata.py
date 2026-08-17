# SPDX-License-Identifier: Apache-2.0
"""MLX-dependent regression tests for the batch-reshuffle metadata leak fix.

Regression: ``MLLMBatch.filter()``/``extend()`` rebuild per-layer cache
metadata (``offset``/``left_padding``/``lengths``) as lazy MLX ops, but the
decode loop only evaluates ``y``/``logprobs`` — and ``offset`` is never
read by the decode-time forward graph (``make_mask`` passes the Python-int
``_idx`` as the mask offset; ``ArraysCache`` metadata likewise has no
decode-time reader on hybrid models).  Left lazy, each membership change
extends an unevaluated graph chain that transitively retains every buffer
that ever fed it, for the lifetime of the batch.  Under continuous traffic
the batch never drains, so retained Metal buffer handles grow until the
per-process resource limit (499000) aborts the server.  The fix evaluates
these small arrays after every ``filter()``/``extend()``.

These tests use real ``mlx_lm.models.cache`` objects and run only on the
apple-silicon CI matrix (and locally on M-series hardware); the Linux
``test-matrix`` job excludes this file because MLX has no Linux
distribution.
"""

import os
import tempfile


def _make_batch(cache):
    import mlx.core as mx

    from vllm_mlx.mllm_batch_generator import MLLMBatch

    n = None
    for c in cache:
        if hasattr(c, "offset"):
            n = c.offset.size
            break
    n = n or 1
    return MLLMBatch(
        uids=list(range(n)),
        request_ids=[f"r{i}" for i in range(n)],
        y=mx.zeros((n,), mx.int32),
        logprobs=[mx.zeros((4,)) for _ in range(n)],
        max_tokens=[16] * n,
        num_tokens=[0] * n,
        cache=cache,
        requests=[None] * n,
    )


def _make_live_batch(n, layers=2, heads=2, seq=4, dim=4):
    """A batch mid-decode: prefilled ``BatchKVCache`` layers, fully materialized."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    caches = []
    for _ in range(layers):
        c = BatchKVCache([0] * n)
        k = mx.zeros((n, heads, seq, dim), mx.float16)
        c.update_and_fetch(k, k)
        caches.append(c)
    batch = _make_batch(caches)
    mx.eval(batch.y, *[c.keys for c in caches], *[c.values for c in caches])
    return batch


def _install_eval_spy(monkeypatch):
    """Route ``mx.eval`` through a spy that records every array it is passed
    while still evaluating normally.  Lets tests assert *what* a public call
    evaluated without ever touching the arrays themselves (observing a lazy
    array evaluates it, which would mask the leak under test)."""
    import mlx.core as mx

    captured = []
    real_eval = mx.eval

    def spy(*arrays):
        captured.extend(arrays)
        return real_eval(*arrays)

    monkeypatch.setattr(mx, "eval", spy)
    return captured


def _assert_metadata_evaluated(cache_obj, captured, context):
    """Assert every metadata array on this (possibly nested) cache is among
    the arrays the spied ``mx.eval`` received.  The attribute list mirrors
    the full set of reshuffle-rewritten, decode-unread arrays in
    ``mlx_lm.models.cache`` (audited against mlx-lm 0.31.3: BatchKVCache and
    BatchRotatingKVCache rewrite ``offset``/``left_padding``; ArraysCache
    rewrites ``left_padding``/``lengths``; everything else rewritten —
    keys/values/recurrent state — is read by the next forward step and
    materializes itself)."""
    import mlx.core as mx

    stack = [cache_obj]
    checked = 0
    while stack:
        c = stack.pop()
        if c is None:
            continue
        children = getattr(c, "caches", None)
        if children is not None:
            stack.extend(children)
        for name in ("offset", "left_padding", "lengths"):
            a = getattr(c, name, None)
            if isinstance(a, mx.array):
                checked += 1
                assert any(s is a for s in captured), (
                    f"{context}: {type(c).__name__}.{name} was rewritten but "
                    f"never evaluated by the public call — its lazy chain "
                    f"stays live and pins buffers"
                )
    assert checked, f"{context}: no metadata arrays found to check"


def _pending_edges(*arrays):
    """Count pending lazy-graph edges. An evaluated array serializes with
    zero ``->`` edges in ``mx.export_to_dot`` output; pending computation
    serializes with at least one."""
    import mlx.core as mx

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.dot")
        mx.export_to_dot(path, *arrays)
        with open(path, encoding="utf-8") as f:
            return f.read().count("->")


def _assert_instrument_works():
    import mlx.core as mx

    assert _pending_edges(mx.zeros((3,)) + 1) > 0, "edge counter is vacuous"


def _assert_no_pending_cache_arrays(batch, context):
    """Discovery-based invariant: after a reshuffle plus the decode loop's
    own evaluation, no mx.array anywhere on the caches may carry pending
    lazy work. Unlike the spy assertions this derives its expectations from
    the objects themselves (every array attribute, recursing into
    containers), so metadata added by a future cache class is covered — and
    it is strategy-agnostic: an implementation that materializes via
    ``mx.depends`` or ``mx.async_eval`` instead of ``mx.eval`` passes."""
    import mlx.core as mx

    arrays = []
    stack = list(batch.cache)
    while stack:
        c = stack.pop()
        if c is None:
            continue
        children = getattr(c, "caches", None)
        if children is not None:
            stack.extend(children)
        for a in vars(c).values():
            if isinstance(a, mx.array):
                arrays.append(a)
            elif isinstance(a, (list, tuple)):
                arrays.extend(x for x in a if isinstance(x, mx.array))
    assert arrays, f"{context}: no cache arrays discovered"
    edges = _pending_edges(*arrays)
    assert edges == 0, (
        f"{context}: {edges} pending lazy edges remain on cache arrays "
        f"after the decode-loop evaluation"
    )


class TestReshuffleEvaluatesMetadata:
    """Public-boundary regression tests for the leak itself.

    Each test drives the real ``filter()``/``extend()`` entry points (the
    only call sites the scheduler uses) and asserts, via the ``mx.eval``
    spy, that the metadata rewritten by the call was evaluated *by the call
    itself*.  On code without the fix these fail because a reshuffle
    evaluates nothing at all — which is precisely the defect, not a missing
    helper method.
    """

    def test_filter_evaluates_metadata(self, monkeypatch):
        """A request leaving the batch must not leave lazy metadata behind."""
        import mlx.core as mx

        _assert_instrument_works()
        batch = _make_live_batch(3)
        captured = _install_eval_spy(monkeypatch)
        batch.filter([1, 2])
        for c in batch.cache:
            _assert_metadata_evaluated(c, captured, "filter")
        # decode loop's own evaluation, then the discovery-based invariant
        mx.eval(
            batch.y, *(c.keys for c in batch.cache), *(c.values for c in batch.cache)
        )
        _assert_no_pending_cache_arrays(batch, "filter")

    def test_extend_evaluates_metadata(self, monkeypatch):
        """A request joining the batch must not leave lazy metadata behind."""
        import mlx.core as mx

        _assert_instrument_works()
        a = _make_live_batch(2)
        b = _make_live_batch(1)
        captured = _install_eval_spy(monkeypatch)
        a.extend(b)
        for c in a.cache:
            _assert_metadata_evaluated(c, captured, "extend")
        mx.eval(a.y, *(c.keys for c in a.cache), *(c.values for c in a.cache))
        _assert_no_pending_cache_arrays(a, "extend")

    def test_hybrid_cachelist_children_evaluated(self, monkeypatch):
        """Hybrid-model layers wrap caches in ``CacheList``; the children's
        metadata (including ``ArraysCache.lengths``, which plain-attention
        tests never exercise) must be evaluated through the public
        ``filter()`` as well.  Guards the container recursion — the
        works-for-pure-transformers, leaks-for-hybrids failure mode."""
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

        n = 2
        k = mx.zeros((n, 2, 4, 4), mx.float16)
        inner_kv = BatchKVCache([0] * n)
        inner_kv.update_and_fetch(k, k)
        recurrent = ArraysCache(1, left_padding=[0] * n)
        recurrent.lengths = mx.array([4] * n)
        recurrent.cache[0] = mx.zeros((n, 4))
        plain = BatchKVCache([0] * n)
        plain.update_and_fetch(k, k)
        batch = _make_batch([CacheList(inner_kv, recurrent), plain])
        mx.eval(inner_kv.keys, plain.keys, recurrent.cache[0])

        captured = _install_eval_spy(monkeypatch)
        batch.filter([1])
        for layer in batch.cache:
            _assert_metadata_evaluated(layer, captured, "hybrid filter")

        # the joining side of the hybrid shape: extend() must sweep the
        # CacheList children's freshly concatenated metadata as well
        other_kv = BatchKVCache([0])
        other_kv.update_and_fetch(k[:1], k[:1])
        other_rec = ArraysCache(1, left_padding=[3])
        other_rec.lengths = mx.array([2])
        other_rec.cache[0] = mx.zeros((1, 4))
        other_plain = BatchKVCache([0])
        other_plain.update_and_fetch(k[:1], k[:1])
        incoming = _make_batch([CacheList(other_kv, other_rec), other_plain])
        captured.clear()
        batch.extend(incoming)
        for layer in batch.cache:
            _assert_metadata_evaluated(layer, captured, "hybrid extend")

    def test_churn_keeps_metadata_materialized_every_cycle(self, monkeypatch):
        """The production failure mode in miniature: sustained join/leave
        churn with the batch never draining.  The invariant — every
        membership change evaluates its rewritten metadata immediately —
        must hold on *every* cycle, not just the first, so stateful
        regressions (a sweep that stops firing once the batch has history)
        also turn the suite red.  Each cycle ends with the decode loop's
        real evaluation pattern (``y`` + keys/values only)."""
        import mlx.core as mx

        _assert_instrument_works()
        resident = _make_live_batch(3)
        captured = _install_eval_spy(monkeypatch)
        for cycle in range(30):
            incoming = _make_live_batch(1)
            captured.clear()
            resident.extend(incoming)
            for c in resident.cache:
                _assert_metadata_evaluated(c, captured, f"cycle {cycle} extend")
            captured.clear()
            resident.filter([1, 2, 3])
            for c in resident.cache:
                _assert_metadata_evaluated(c, captured, f"cycle {cycle} filter")
            mx.eval(
                resident.y,
                *(c.keys for c in resident.cache),
                *(c.values for c in resident.cache),
            )
            _assert_no_pending_cache_arrays(resident, f"cycle {cycle}")


class TestSyncReshuffleMetadata:
    """Unit tests of the sweep helper's collection coverage."""

    def test_collects_batch_kv_cache_metadata(self, monkeypatch):
        """offset and left_padding of every BatchKVCache layer are evaluated."""
        from mlx_lm.models.cache import BatchKVCache

        batch = _make_batch([BatchKVCache([0, 2]), BatchKVCache([1, 0])])
        captured = _install_eval_spy(monkeypatch)
        batch._sync_reshuffle_metadata()
        for c in batch.cache:
            _assert_metadata_evaluated(c, captured, "sweep")

    def test_recurses_into_cache_list_children(self, monkeypatch):
        """Hybrid models wrapping layers in CacheList get the same protection."""
        from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

        batch = _make_batch(
            [CacheList(BatchKVCache([0, 0]), ArraysCache(2)), BatchKVCache([0, 3])]
        )
        captured = _install_eval_spy(monkeypatch)
        batch._sync_reshuffle_metadata()
        for c in batch.cache:
            _assert_metadata_evaluated(c, captured, "sweep")

    def test_none_cache_slots_are_skipped(self, monkeypatch):
        """None entries in the layer-cache list must not crash the sweep."""
        from mlx_lm.models.cache import BatchKVCache

        batch = _make_batch([None, BatchKVCache([0])])
        captured = _install_eval_spy(monkeypatch)
        batch._sync_reshuffle_metadata()
        _assert_metadata_evaluated(batch.cache[1], captured, "sweep with None slot")


class TestReshuffleValueCorrectness:
    """Guards that the eager evaluation does not corrupt metadata values.

    NOTE: these pass on unfixed code as well — the ``.tolist()`` calls force
    evaluation themselves, so they cannot detect the leak.  They exist to
    pin the *values* produced by filter/extend with the eager eval in place.
    Leak detection lives in ``TestReshuffleEvaluatesMetadata``.
    """

    def test_filter_values_correct(self):
        """After filter(), offset/left_padding values must be correct."""
        from mlx_lm.models.cache import BatchKVCache

        cache = [BatchKVCache([1, 3, 0]) for _ in range(2)]
        batch = _make_batch(cache)
        batch.filter([1, 2])
        for c in cache:
            assert c.offset.tolist() == [-3, 0]
            assert c.left_padding.tolist() == [3, 0]
        assert len(batch.uids) == 2

    def test_extend_values_correct(self):
        """After extend() of prefilled caches, concatenated values are correct."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        def prefill(c, bs):
            k = mx.zeros((bs, 2, 1, 4))
            c.update_and_fetch(k, k)

        ca, cb = BatchKVCache([0, 1]), BatchKVCache([2])
        prefill(ca, 2)
        prefill(cb, 1)
        a = _make_batch([ca])
        b = _make_batch([cb])
        a.extend(b)
        assert a.cache[0].offset.tolist() == [1, 0, -1]
        assert a.cache[0].left_padding.tolist() == [0, 1, 2]
        assert len(a.uids) == 3
