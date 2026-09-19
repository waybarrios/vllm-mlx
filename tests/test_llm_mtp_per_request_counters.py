# SPDX-License-Identifier: Apache-2.0
"""Deterministic coverage for per-request MTP counters on the plain-text
LLM path (scheduler.py) -- the sibling gap to mllm_scheduler.py's
mtp_drafts/mtp_accepted, which is already wired end-to-end.

Covers:
- _install_mtp's per-UID attempt/accept delta tracking, driven through the
  real _mtp_step/_mtp_next closures with a small deterministic fake model
  (3-token vocab, no real weights) -- including a UID joining mid-stream,
  to prove drafts attributed to one request don't leak onto another.
- Scheduler._process_batch_responses correctly attributes drained deltas
  to the right Request and copies them onto RequestOutput, applying each
  UID's delta at most once per call.
"""

from collections import namedtuple

import pytest

try:
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


_Response = namedtuple(
    "_Response", ["uid", "token", "logprobs", "finish_reason", "cache_out"]
)


class _FakeActiveBatch:
    def __init__(self, uids, cache):
        self.uids = uids
        self.cache = cache

    def filter(self, keep_indices):
        self.uids = [self.uids[i] for i in keep_indices]


class _FakeMTPModel:
    """Deterministic 3-token-vocab model: primary always predicts token 0,
    the MTP head always drafts token 1. ``accept`` controls whether the
    verify pass's own prediction agrees with the draft (accept) or not
    (reject) -- vocab index 1 vs. index 0 at verify position 0.
    """

    def __init__(self, accept: bool):
        self._accept = accept

    def __call__(self, tokens, cache, return_hidden=True):
        batch, seq = tokens.shape
        if seq == 1:
            logits = mx.array([[[10.0, 0.0, 0.0]]] * batch)
            hidden = mx.zeros((batch, 1, 1))
            return logits, hidden
        # 2-token verify pass (primary, draft): position 0's argmax is
        # compared against the draft token to decide accept/reject.
        row0 = [0.0, 10.0, 0.0] if self._accept else [10.0, 0.0, 0.0]
        verify_logits = mx.array([[row0, [10.0, 0.0, 0.0]]] * batch)
        verify_hidden = mx.zeros((batch, 1, 1))
        return verify_logits, verify_hidden

    def mtp_forward(self, hidden, primary_tokens, mtp_cache=None):
        batch = primary_tokens.shape[0]
        return mx.array([[[0.0, 10.0, 0.0]]] * batch)


class _FakeBatchGenerator:
    """Minimal stand-in for mlx_lm's real BatchGenerator: just enough
    surface for _install_mtp to patch. ``_next`` mimics the real one's
    contract of advancing one primary token per active UID via
    ``self._step`` -- which _install_mtp replaces with ``_mtp_step``.
    """

    Response = _Response

    def __init__(self, uids, cache, stop_tokens=frozenset({1})):
        self.active_batch = _FakeActiveBatch(list(uids), cache)
        self.sampler = make_sampler(temp=0.0)
        self.stop_tokens = stop_tokens

    def _step(self, *args, **kwargs):
        raise NotImplementedError("replaced by _install_mtp")

    def _next(self):
        if self.active_batch is None:
            return []
        uids = list(self.active_batch.uids)
        input_tokens = mx.zeros((len(uids), 1), dtype=mx.int32)
        primary_tokens, logprobs = self._step(
            input_tokens, self.active_batch.cache, [None] * len(uids), None, None
        )
        primary_list = primary_tokens.tolist()
        return [
            self.Response(uid, primary_list[i], logprobs[i], None, None)
            for i, uid in enumerate(uids)
        ]


class TestInstallMtpPerUidDeltas:
    """_install_mtp's closure-level per-UID attempt/accept bookkeeping."""

    def test_attempt_recorded_even_when_verify_rejects(self):
        from vllm_mlx.scheduler import _install_mtp

        cache = []
        batch_gen = _FakeBatchGenerator(uids=[7], cache=cache)
        _install_mtp(batch_gen, model=_FakeMTPModel(accept=False), num_draft_tokens=1)

        batch_gen._next()
        drafts, accepted = batch_gen.drain_mtp_uid_deltas()

        assert drafts == {7: 1}
        assert accepted == {}

    def test_accept_only_recorded_once_draft_is_actually_emitted(self):
        from vllm_mlx.scheduler import _install_mtp

        cache = []
        batch_gen = _FakeBatchGenerator(uids=[1], cache=cache)
        _install_mtp(batch_gen, model=_FakeMTPModel(accept=True), num_draft_tokens=1)

        # Step 1: the draft is verified-accepted (deferred), not yet
        # surfaced as an output token -- accepted delta must stay empty.
        batch_gen._next()
        drafts_1, accepted_1 = batch_gen.drain_mtp_uid_deltas()
        assert drafts_1 == {1: 1}
        assert accepted_1 == {}

        # Step 2: the deferred draft from step 1 is emitted.
        batch_gen._next()
        drafts_2, accepted_2 = batch_gen.drain_mtp_uid_deltas()
        assert accepted_2 == {1: 1}

    def test_uid_joining_mid_stream_does_not_inherit_another_uids_count(self):
        """A UID joining mid-stream must not inherit an earlier UID's
        accepted-draft count, and must start accumulating its own drafts
        only from the step it actually joins."""
        from vllm_mlx.scheduler import _install_mtp

        cache = []
        batch_gen = _FakeBatchGenerator(uids=[1], cache=cache)
        _install_mtp(batch_gen, model=_FakeMTPModel(accept=True), num_draft_tokens=1)

        # Step 1: only uid 1 present.
        batch_gen._next()
        drafts_1, accepted_1 = batch_gen.drain_mtp_uid_deltas()
        assert drafts_1 == {1: 1}
        assert accepted_1 == {}

        # Step 2: uid 2 joins. Both attempt a draft this step; only uid 1's
        # step-1 draft is due for emission (uid 2 has no prior deferred
        # draft of its own yet).
        batch_gen.active_batch.uids.append(2)
        batch_gen._next()
        drafts_2, accepted_2 = batch_gen.drain_mtp_uid_deltas()

        assert drafts_2 == {1: 1, 2: 1}
        assert accepted_2 == {1: 1}
        assert 2 not in accepted_2


def _make_scheduler():
    """Mirrors the mock_model/mock_tokenizer Scheduler-construction idiom
    used across test_batching.py/test_memory_stability.py."""
    from unittest.mock import MagicMock

    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: list(range(len(x.split())))
    tokenizer.eos_token_id = 0
    return Scheduler(model, tokenizer, SchedulerConfig(enable_prefix_cache=False))


class TestProcessBatchResponsesAttributesMtpDeltas:
    """Scheduler._process_batch_responses must attribute drained per-UID
    MTP deltas to the correct Request and copy them onto RequestOutput.
    """

    def test_deltas_land_on_the_right_request_and_output(self):
        from vllm_mlx.request import Request, SamplingParams

        scheduler = _make_scheduler()

        req_a = Request(request_id="a", prompt="hi", sampling_params=SamplingParams())
        req_b = Request(request_id="b", prompt="yo", sampling_params=SamplingParams())
        req_a.output_token_ids = [1]  # skip _store_prompt_only_cache
        req_b.output_token_ids = [1]
        scheduler.running = {"a": req_a, "b": req_b}
        scheduler.uid_to_request_id = {1: "a", 2: "b"}

        class FakeBatchGenerator:
            def drain_mtp_uid_deltas(self):
                return {1: 2, 2: 1}, {1: 1}

        scheduler.batch_generator = FakeBatchGenerator()

        responses = [
            _Response(
                uid=1, token=5, logprobs=None, finish_reason="stop", cache_out=None
            ),
            _Response(
                uid=2, token=6, logprobs=None, finish_reason="stop", cache_out=None
            ),
        ]

        outputs, finished_ids = scheduler._process_batch_responses(responses)

        assert req_a.mtp_drafts == 2
        assert req_a.mtp_accepted == 1
        assert req_b.mtp_drafts == 1
        assert req_b.mtp_accepted == 0

        by_request = {o.request_id: o for o in outputs}
        assert by_request["a"].mtp_drafts == 2
        assert by_request["a"].mtp_accepted == 1
        assert by_request["b"].mtp_drafts == 1
        assert by_request["b"].mtp_accepted == 0

    def test_no_mtp_installed_leaves_counters_at_zero(self):
        """No batch_generator.drain_mtp_uid_deltas attribute (MTP not
        installed) must not crash and must leave counters at their
        dataclass default of 0."""
        from vllm_mlx.request import Request, SamplingParams

        scheduler = _make_scheduler()
        req = Request(request_id="a", prompt="hi", sampling_params=SamplingParams())
        req.output_token_ids = [1]
        scheduler.running = {"a": req}
        scheduler.uid_to_request_id = {1: "a"}

        class PlainBatchGenerator:
            pass

        scheduler.batch_generator = PlainBatchGenerator()

        responses = [
            _Response(
                uid=1, token=5, logprobs=None, finish_reason="stop", cache_out=None
            )
        ]

        outputs, _ = scheduler._process_batch_responses(responses)

        assert outputs[0].mtp_drafts == 0
        assert outputs[0].mtp_accepted == 0
