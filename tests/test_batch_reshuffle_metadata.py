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
        mx.eval(batch.y, inner_kv.keys, plain.keys, recurrent.cache[0])
        assert recurrent.left_padding.tolist() == [0]
        assert recurrent.lengths.tolist() == [4]

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
        mx.eval(batch.y, inner_kv.keys, plain.keys, recurrent.cache[0])
        assert recurrent.left_padding.tolist() == [0, 3]
        assert recurrent.lengths.tolist() == [4, 2]

    def test_rotating_kv_cache_layer_evaluated(self, monkeypatch):
        """``BatchRotatingKVCache`` layers (sliding-window attention) also
        rewrite ``offset``/``left_padding`` on filter/extend.  The class
        guards its *decode-step* writes itself via ``mx.depends`` in
        ``update_and_fetch``, but a reshuffle happens outside any decode
        step, so the sweep must cover it like any other layer."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchRotatingKVCache

        _assert_instrument_works()

        def live_rotating(left_padding):
            n = len(left_padding)
            c = BatchRotatingKVCache(max_size=8, left_padding=left_padding)
            k = mx.zeros((n, 2, 4, 4), mx.float16)
            c.update_and_fetch(k, k)
            batch = _make_batch([c])
            mx.eval(batch.y, c.keys, c.values)
            return batch

        batch = live_rotating([0, 1, 0])
        captured = _install_eval_spy(monkeypatch)
        batch.filter([1, 2])
        _assert_metadata_evaluated(batch.cache[0], captured, "rotating filter")
        mx.eval(batch.y, batch.cache[0].keys, batch.cache[0].values)
        _assert_no_pending_cache_arrays(batch, "rotating filter")

        incoming = live_rotating([2])
        captured.clear()
        batch.extend(incoming)
        _assert_metadata_evaluated(batch.cache[0], captured, "rotating extend")
        mx.eval(batch.y, batch.cache[0].keys, batch.cache[0].values)
        _assert_no_pending_cache_arrays(batch, "rotating extend")

    def test_churn_under_generation_stream(self):
        """Membership churn inside the scheduler's MLX execution context.

        ``MLLMBatchGenerator.next()`` runs every decode step — including the
        ``batch.filter()``/``batch.extend()`` reshuffles — under
        ``with mx.stream(MLLMBatchGenerator._stream)``, a dedicated
        generation stream created once per process.  This test replicates
        that exact context (same class attribute, same construction, same
        context manager) around sustained churn and asserts that after each
        cycle's normal ``y``/keys/values evaluation no cache array carries
        pending lazy work, and the surviving metadata values stay correct.
        No model forward pass is needed: the leak is a property of the
        reshuffle graph, not of inference."""
        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator

        _assert_instrument_works()
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())
        with mx.stream(MLLMBatchGenerator._stream):
            resident = _make_live_batch(3)
            for cycle in range(20):
                incoming = _make_live_batch(1)
                resident.extend(incoming)
                resident.filter([1, 2, 3])
                mx.eval(
                    resident.y,
                    *(c.keys for c in resident.cache),
                    *(c.values for c in resident.cache),
                )
                _assert_no_pending_cache_arrays(resident, f"stream cycle {cycle}")
                for c in resident.cache:
                    assert c.offset.tolist() == [4] * 3, f"stream cycle {cycle}"
                    assert c.left_padding.tolist() == [0] * 3, f"stream cycle {cycle}"

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


class TestGeneratorPathChurn:
    """Join/leave churn driven through the real ``MLLMBatchGenerator.next()``
    path — admission of queued requests, mid-batch extend, decode step,
    retirement, and filter all run the production ``_next()`` code.

    The fixture is deterministic and model-free: only the two operations
    that require a real model are replaced — ``_process_prompts`` builds a
    real ``MLLMBatch`` over real prefilled ``BatchKVCache`` layers, and
    ``_step`` performs a genuine ``update_and_fetch`` per layer (the same
    lazy ``offset += 1`` metadata write a production decode step makes)
    and returns deterministic tokens.  Everything else — the admission
    branch, the mid-batch extend branch, ``mx.async_eval`` of the step
    output, retirement bookkeeping, and ``batch.filter()`` — is the real
    generator code, running under the generator's own stream context.
    """

    LAYERS = 2
    HEADS = 2
    DIM = 4
    PREFILL = 4

    def _make_generator(self):
        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchStats,
        )

        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.active_batch = None
        gen.unprocessed_requests = []
        gen._pending_error_responses = []
        gen._prefill_progress = {}
        gen._aborted_request_ids = set()
        gen._stats = MLLMBatchStats()
        gen.stop_tokens = set()
        gen.prefix_cache = None
        gen.completion_batch_size = 16
        gen._require_uniform_mllm_draft = False
        gen._allow_mid_batch_extend = True

        def process_prompts(requests):
            from mlx_lm.models.cache import BatchKVCache

            n = len(requests)
            caches = []
            for _ in range(self.LAYERS):
                c = BatchKVCache([0] * n)
                k = mx.zeros((n, self.HEADS, self.PREFILL, self.DIM), mx.float16)
                c.update_and_fetch(k, k)
                caches.append(c)
            batch = _make_batch(caches)
            batch.uids = [r.uid for r in requests]
            batch.request_ids = [r.request_id for r in requests]
            batch.max_tokens = [r.max_tokens for r in requests]
            batch.requests = list(requests)
            batch.logprobs = [mx.zeros((4,)) for _ in requests]
            # prompt-path evaluation, as production's _eval_prompt_cache does
            mx.eval(batch.y, *(c.keys for c in caches), *(c.values for c in caches))
            return batch

        def step(
            input_tokens, cache, logits_processors=None, tokens=None, samplers=None
        ):
            n = input_tokens.shape[0]
            k = mx.zeros((n, self.HEADS, 1, self.DIM), mx.float16)
            for c in cache:
                c.update_and_fetch(k, k)
            return mx.ones((n,), mx.int32), [mx.zeros((4,)) for _ in range(n)]

        gen._process_prompts = process_prompts
        gen._step = step
        return gen

    def test_next_join_leave_churn_keeps_metadata_materialized(self):
        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        _assert_instrument_works()
        gen = self._make_generator()

        def make_request(uid, max_tokens):
            return MLLMBatchRequest(
                uid=uid,
                request_id=f"req-{uid}",
                prompt="x",
                max_tokens=max_tokens,
                is_text_only=True,
            )

        def check_step(context, expected_offsets, filtered):
            """Decode-style evaluation, then the pending-metadata check.

            Inside ``_next()`` the decode step runs after the extend branch
            but before retirement filtering.  A call that retires a request
            therefore ends on ``batch.filter()`` — whose sweep is the last
            thing to touch the caches — so pending metadata must be exactly
            zero.  A call that only admits ends on the decode step, whose
            lazy per-step ``offset += 1`` write legitimately remains (it is
            bounded and collapsed by the next reshuffle), so the check is a
            small constant bound instead.  On unfixed code the first
            retirement cycle fails the exact-zero check.  The value asserts
            run last: ``.tolist()`` forces evaluation, so putting them
            earlier would cure the very laziness under test.
            """
            batch = gen.active_batch
            mx.eval(
                batch.y,
                *(c.keys for c in batch.cache),
                *(c.values for c in batch.cache),
            )
            if filtered:
                _assert_no_pending_cache_arrays(batch, context)
            else:
                arrays = [c.offset for c in batch.cache] + [
                    c.left_padding for c in batch.cache
                ]
                # constant bound: production's prompt-path eval
                # (_cache_eval_tensors) covers keys/values only, so a fresh
                # batch legitimately carries its prefill metadata chain plus
                # one decode step's writes — a fixed amount per layer, never
                # per-cycle growth (the next reshuffle sweep collapses it)
                edges = _pending_edges(*arrays)
                assert edges <= 16 * self.LAYERS, (
                    f"{context}: {edges} pending metadata edges after a "
                    f"non-filtering step — beyond one prefill plus one "
                    f"decode step's worth"
                )
            # BatchKVCache right-aligns keys on extend: a mid-stream joiner
            # gets left_padding = max(offset) - its offset (alignment law)
            want = [expected_offsets[u] for u in batch.uids]
            want_lp = [max(want) - o for o in want]
            for c in batch.cache:
                assert c.offset.tolist() == want, context
                assert c.left_padding.tolist() == want_lp, context

        # three residents that never retire within the test
        expected = {}
        for uid in (0, 1, 2):
            gen.unprocessed_requests.append(make_request(uid, max_tokens=999))
            expected[uid] = self.PREFILL

        # admission: first next() builds the batch and runs one decode step
        responses = gen.next()
        assert sorted(r.uid for r in responses) == [0, 1, 2]
        for uid in (0, 1, 2):
            expected[uid] += 1
        check_step("admission", expected, filtered=False)

        # churn: each cycle one request joins (mid-batch extend inside
        # next()) and, two cycles later, retires by length (filter inside
        # next()); the batch never drains — the production failure mode
        retired = []
        for cycle in range(10):
            uid = 100 + cycle
            gen.unprocessed_requests.append(make_request(uid, max_tokens=2))
            expected[uid] = self.PREFILL
            responses = gen.next()
            cycle_retired = False
            for r in responses:
                expected[r.uid] += 1
                if r.finish_reason is not None:
                    assert r.finish_reason == "length"
                    retired.append(r.uid)
                    del expected[r.uid]
                    cycle_retired = True
            check_step(f"churn cycle {cycle}", expected, filtered=cycle_retired)
            assert not gen.unprocessed_requests, f"churn cycle {cycle}"
            joined = set(gen.active_batch.uids)
            assert uid in joined, f"churn cycle {cycle}: request never admitted"
            assert not (set(retired) & joined), f"churn cycle {cycle}"

        # both reshuffle directions really ran through next(): 10 joins,
        # and every joiner except the one still in flight retired (a joiner
        # gets its first token in the same call that admits it, so it
        # retires by length one cycle later)
        assert retired == [100 + c for c in range(9)]
        assert len(gen.active_batch) == 4


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
    """The rewritten metadata must be *materialized* by the operation itself
    (zero pending lazy edges immediately after the call — this half is red
    on unfixed code) and must still hold the *correct values* after the
    decode loop's own ``y``/keys/values evaluation runs on top (this half
    guards against an eager evaluation that materializes the wrong graph).
    """

    def _assert_metadata_materialized(self, cache, context):
        for c in cache:
            edges = _pending_edges(c.offset, c.left_padding)
            assert edges == 0, (
                f"{context}: metadata still pending ({edges} lazy edges) "
                f"immediately after the operation"
            )

    def test_filter_values_correct(self):
        """After filter(): metadata materialized by the call, values correct
        after a decode-style evaluation."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        _assert_instrument_works()
        cache = []
        for _ in range(2):
            c = BatchKVCache([1, 3, 0])
            k = mx.zeros((3, 2, 1, 4))
            c.update_and_fetch(k, k)
            cache.append(c)
        batch = _make_batch(cache)
        batch.filter([1, 2])
        self._assert_metadata_materialized(cache, "filter")
        mx.eval(batch.y, *(c.keys for c in cache), *(c.values for c in cache))
        for c in cache:
            assert c.offset.tolist() == [-2, 1]
            assert c.left_padding.tolist() == [3, 0]
        assert len(batch.uids) == 2

    def test_extend_values_correct(self):
        """After extend(): metadata materialized by the call, concatenated
        values correct after a decode-style evaluation."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        def prefill(c, bs):
            k = mx.zeros((bs, 2, 1, 4))
            c.update_and_fetch(k, k)

        _assert_instrument_works()
        ca, cb = BatchKVCache([0, 1]), BatchKVCache([2])
        prefill(ca, 2)
        prefill(cb, 1)
        a = _make_batch([ca])
        b = _make_batch([cb])
        a.extend(b)
        self._assert_metadata_materialized(a.cache, "extend")
        mx.eval(a.y, a.cache[0].keys, a.cache[0].values)
        assert a.cache[0].offset.tolist() == [1, 0, -1]
        assert a.cache[0].left_padding.tolist() == [0, 1, 2]
        assert len(a.uids) == 3

    def test_rotating_values_correct(self):
        """Same contract for ``BatchRotatingKVCache`` metadata."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchRotatingKVCache

        def live(left_padding):
            n = len(left_padding)
            c = BatchRotatingKVCache(max_size=8, left_padding=left_padding)
            k = mx.zeros((n, 2, 4, 4), mx.float16)
            c.update_and_fetch(k, k)
            return c

        _assert_instrument_works()
        a = _make_batch([live([0, 1])])
        b = _make_batch([live([2])])
        a.extend(b)
        self._assert_metadata_materialized(a.cache, "rotating extend")
        mx.eval(a.y, a.cache[0].keys, a.cache[0].values)
        assert a.cache[0].offset.tolist() == [4, 3, 2]
        assert a.cache[0].left_padding.tolist() == [0, 1, 2]

        a.filter([1, 2])
        self._assert_metadata_materialized(a.cache, "rotating filter")
        mx.eval(a.y, a.cache[0].keys, a.cache[0].values)
        assert a.cache[0].offset.tolist() == [3, 2]
        assert a.cache[0].left_padding.tolist() == [1, 2]
